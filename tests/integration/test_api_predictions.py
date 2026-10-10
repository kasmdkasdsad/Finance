"""Forecasts, the stock model, stock reports, the market regime and the prediction ledger through the API."""

import functools
from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.providers import synthetic
from quantpulse.schemas.market import PriceHistory

from .conftest import _client, make_settings

FRIDAY_AFTER_CLOSE = datetime(2026, 9, 25, 20, 30, tzinfo=UTC)  # 16:30 New York
UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "LLY", "COST", "KO", "PG"]


@functools.lru_cache(maxsize=128)
def _full_history(symbol, interval):
    anchor = ConsistentFeed.ANCHOR
    return synthetic.synthetic_history(symbol, interval, anchor - timedelta(days=2400), anchor, anchor)


class ConsistentFeed:
    """A stand-in live vendor whose price path does not change as the test clock moves forward.

    It reads one fixed synthetic history (anchored far in the future) and only reveals bars that have
    closed by the clock's "now", so a prediction logged on Friday can be graded against the real
    (simulated) close a week later."""

    name = "replayfeed"
    ANCHOR = datetime(2027, 3, 31, 21, 0, tzinfo=UTC)

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def configured(self) -> bool:
        return True

    async def history(self, symbol, interval, start, end):
        now = self._clock.now()
        bars = [b for b in _full_history(symbol, interval).bars if start <= b.timestamp <= now]
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)

    async def quote(self, symbol):
        now = self._clock.now()
        h = await self.history(symbol, "1d", now - timedelta(days=10), now)
        last = h.bars[-1]
        return synthetic.synthetic_quote(symbol, now).model_copy(
            update={
                "price": last.close,
                "name": symbol,
                "timestamp": now,
                "previous_close": None,
                "change": None,
                "change_percent": None,
            }
        )


async def feed_client(tmp_path, clock, **overrides):
    settings = make_settings(tmp_path, enable_live_data=True, picks_universe=",".join(UNIVERSE), **overrides)
    async for c in _client(settings, clock):
        c.container.market._providers[:] = [ConsistentFeed(clock)]
        yield c


# ----------------------------------------------------------------------------- forecasts
async def test_forecast_endpoint_offline(api):
    r = await api.get("/api/v1/forecast/AAPL", params={"target": 250, "calibrate": True})
    assert r.status_code == 200, r.text
    body = r.json()
    f = body["data"]
    assert [h["days"] for h in f["horizons"]] == [5, 21, 63] and len(f["cone"]) == 63
    assert f["data_status"] == "synthetic" and body["meta"]["status"] == "synthetic"
    assert any("synthetic" in n.lower() for n in f["notes"])
    for h in f["horizons"]:
        b = h["band"]
        assert b["p05"] < b["p25"] < b["p50"] < b["p75"] < b["p95"]
        assert 0 <= h["prob_up"] <= 1 and 0 <= h["prob_above_target"] <= 1
    widths = [h["band"]["p95"] - h["band"]["p05"] for h in f["horizons"]]
    assert widths == sorted(widths)
    assert f["cone"][0]["date"] == "2026-09-25"  # 10:00 New York: today's close is the first step
    assert f["cone"][4]["date"] == f["horizons"][0]["target_date"] == "2026-10-01"
    month = f["horizons"][1]
    assert month["implied"] is not None and month["implied"]["data_status"] == "synthetic"
    assert 0 < month["implied"]["move_1sd"] < 1
    v = f["volatility"]
    assert v["method"] == "garch" and 0 < v["persistence"] < 1 and v["current_vol_annual"] > 0
    assert f["drift"]["beta"] is not None and f["drift"]["risk_free"] > 0
    # Earnings: past reactions are left out of the GARCH fit and the next one is simulated as a jump.
    e = f["earnings"]
    assert v["earnings_days_excluded"] >= 8 and e["events_used"] == v["earnings_days_excluded"]
    assert e["source"] == "estimated" and e["next_date"] > "2026-09-25" and e["typical_move"] > 0
    assert e["in_horizons"] == [h for h in (5, 21, 63) if h >= e["sessions_ahead"]]
    assert e["modelled"] is bool(e["in_horizons"])
    # Options (synthetic here, like the prices) are blended into the volatility.
    assert v["iv_weight"] == 0.5 and v["variance_premium"] == 1.1 and v["implied_vol_annual_21d"] > 0
    assert v["garch_vol_annual_21d"] > 0 and v["blended_vol_annual_21d"] > 0
    cal = f["calibration"]
    assert cal["horizon"] == 21 and cal["n"] > 50 and 0 <= cal["coverage_90"] <= 1
    assert sum(cal["pit_histogram"]) == pytest.approx(1.0)

    bad = await api.get("/api/v1/forecast/AAPL", params={"horizons": "0"})
    assert bad.status_code == 422
    assert (await api.get("/api/v1/forecast/AAPL", params={"horizons": "x"})).status_code == 422


# ----------------------------------------------------------------------------- the model lab
async def test_model_report_research_and_regime_offline(api):
    r = await api.get("/api/v1/model/report", params={"symbols": ",".join(UNIVERSE)})
    assert r.status_code == 200, r.text
    m = r.json()["data"]
    assert m["symbols"] == sorted(UNIVERSE) and m["horizon"] == 21
    assert m["data_status"] == "synthetic" and any("synthetic" in w.lower() for w in m["warnings"])
    assert m["oos"]["n_dates"] > 100 and -1 <= m["oos"]["mean_ic"] <= 1
    assert m["verdict"] and isinstance(m["has_skill"], bool)
    assert [x["rank"] for x in m["live"]] == list(range(1, len(UNIVERSE) + 1))
    assert all(0 < x["prob_outperform"] < 1 and 1 <= x["rating"] <= 10 for x in m["live"])
    probs = [b["probability"] for b in m["calibration"]]
    assert probs == sorted(probs)  # monotone by construction
    bt = m["backtest"]
    assert len(bt["dates"]) == len(bt["strategy"]) == bt["periods"] + 1 and bt["strategy"][0] == 1.0
    groups = {i["group"] for i in m["importance"]}
    assert len(m["importance"]) == 28 and groups == {"price", "earnings", "sector", "fundamental"}
    assert m["retrains"] >= 3 and len(m["lambdas"]) >= 3 and m["tree_sizes"]
    assert any(i["tree_importance"] is not None for i in m["importance"])
    # every model type is scored on the same out-of-sample dates; the configured one is live
    assert m["model_type"] == "ensemble" and m["model_label"] == "Ensemble (ridge + trees)"
    names = [c["name"] for c in m["comparison"]]
    assert names == ["ensemble", "ridge", "gbm", "baseline"]
    assert [c["chosen"] for c in m["comparison"]] == [True, False, False, False]
    assert m["comparison"][0]["mean_ic"] == pytest.approx(m["oos"]["mean_ic"])
    assert m["comparison"][3]["mean_ic"] == pytest.approx(m["baseline"]["mean_ic"])
    assert m["within_sector"] is not None and m["within_sector"]["n_dates"] == m["oos"]["n_dates"]
    # a fixed ticker list is flagged for survivorship bias; industries, earnings and fundamentals are used
    u = m["universe"]
    assert u["kind"] == "custom" and u["point_in_time"] is False and u["with_prices"] == len(UNIVERSE)
    assert any("survivorship" in w for w in m["warnings"])
    cov = m["coverage"]
    assert cov["sector_neutral"] is True and sum(cov["sectors"].values()) == len(UNIVERSE)
    assert cov["earnings_companies"] == len(UNIVERSE) and cov["fundamentals_companies"] == len(UNIVERSE)
    assert all(x["sector"] and x["sector_label"] for x in m["live"])
    again = (await api.get("/api/v1/model/report", params={"symbols": ",".join(UNIVERSE)})).json()["data"]
    assert again["computed_at"] == m["computed_at"]  # served from the cached run for this close

    res = (await api.get("/api/v1/model/research", params={"symbols": ",".join(UNIVERSE)})).json()["data"]
    assert res["horizons"] == [1, 5, 21, 63] and len(res["signals"]) == 28
    assert {"earn_reaction", "sector_mom_6_1", "book_to_market"} <= {x["feature"] for x in res["signals"]}
    assert res["correlation"]["mom_3m"]["mom_3m"] == pytest.approx(1.0)

    reg = (await api.get("/api/v1/market/regime")).json()["data"]
    assert reg["label"] in {"Uptrend", "Volatile uptrend", "Downtrend", "Stress"}
    assert reg["benchmark"] == "SPY" and len(reg["history"]) == 2 and 0 <= reg["volatility_percentile"] <= 1

    assert (await api.get("/api/v1/model/report", params={"horizon": 2})).status_code == 422


async def test_stock_report_offline(api):
    r = await api.get("/api/v1/stocks/AAPL/report")
    assert r.status_code == 200, r.text
    rep = r.json()["data"]
    assert rep["symbol"] == "AAPL" and len(rep["chart"]) == 252
    assert rep["technicals"]["sma200"] > 0 and 0 <= rep["technicals"]["rsi_14"] <= 100
    assert rep["forecast"]["horizons"][1]["days"] == 21
    assert rep["model"]["in_universe"] is True and 1 <= rep["model"]["rank"] <= 30
    assert rep["valuation"] is not None and rep["valuation"]["value_per_share"] > 0
    assert rep["track_record"]["resolved"] == 0
    assert len(rep["summary"]) >= 4 and "90% of simulated outcomes" in rep["summary"][0]

    outside = (await api.get("/api/v1/stocks/IBM/report", params={"valuation": False})).json()["data"]
    assert outside["model"]["in_universe"] is False and 1 <= outside["model"]["rank"] <= 31
    assert outside["valuation"] is None


# ----------------------------------------------------------------------------- picks
async def test_picks_methods(api):
    factors = (await api.get("/api/v1/picks/daily", params={"method": "factors", "forecast": False})).json()[
        "data"
    ]
    assert factors["method"] == "factors" and factors["model_verdict"] is None
    assert all(p["prob_outperform"] is None and p["low_21d"] is None for p in factors["picks"])
    assert all(p["rating"] == p["factor_rating"] for p in factors["picks"])

    model = (await api.get("/api/v1/picks/daily", params={"method": "model", "top_n": 5})).json()["data"]
    assert model["method"] == "model" and model["model_verdict"]
    ranks = [p["model_rank"] for p in model["picks"]]
    assert ranks == sorted(ranks) and ranks[0] == 1  # ranked by the model
    assert all(
        p["low_21d"] < p["price"] < p["high_21d"] and 0 <= p["prob_up_21d"] <= 1 for p in model["picks"]
    )

    auto = (await api.get("/api/v1/picks/daily", params={"top_n": 5})).json()["data"]
    assert auto["requested_method"] == "auto"
    expected = "blend" if auto["model_has_skill"] else "factors"
    assert auto["method"] == expected
    if expected == "factors":
        assert any("not shown reliable" in n for n in auto["notes"])
    assert all(p["prob_outperform"] is not None for p in auto["picks"])


# ----------------------------------------------------------------------------- the ledger
async def test_prediction_ledger_logs_grades_and_scores(tmp_path, mock_net):
    clock = FakeClock(FRIDAY_AFTER_CLOSE)
    async for api in feed_client(tmp_path, clock):
        logged = await api.post("/api/v1/predictions/log")
        assert logged.status_code == 200, logged.text
        out = logged.json()
        assert out["made_on"] == "2026-09-25" and out["data_status"] in {"live", "cached"}
        assert out["logged"] == len(UNIVERSE) * 3, out["skipped"]  # 5d + 21d forecasts, 21d model
        assert (await api.post("/api/v1/predictions/log")).json()["logged"] == 0  # once per day

        rows = (await api.get("/api/v1/predictions", params={"source": "forecast", "limit": 500})).json()
        five = [p for p in rows if p["horizon_days"] == 5]
        assert len(five) == len(UNIVERSE) and all(p["target_date"] == "2026-10-02" for p in five)
        assert all(p["q05"] < p["q50"] < p["q95"] and p["status"] == "open" for p in five)
        model_rows = (await api.get("/api/v1/predictions", params={"source": "model"})).json()
        assert sorted(p["rank"] for p in model_rows) == list(range(1, len(UNIVERSE) + 1))

        clock.advance((datetime(2026, 10, 2, 21, 0, tzinfo=UTC) - FRIDAY_AFTER_CLOSE).total_seconds())
        res = (await api.post("/api/v1/predictions/resolve")).json()
        assert res == {"resolved": len(UNIVERSE), "voided": 0, "pending": 0}
        graded = (await api.get("/api/v1/predictions", params={"status": "resolved"})).json()
        feed = ConsistentFeed(clock)
        for p in graded:
            h = await feed.history(p["symbol"], "1d", clock.now() - timedelta(days=30), clock.now())
            closes = {b.timestamp.astimezone(UTC).date().isoformat(): b.close for b in h.bars}
            assert p["realized_price"] == pytest.approx(closes["2026-10-02"])
            assert p["reference_price"] == pytest.approx(closes["2026-09-25"])
            assert p["realized_return"] == pytest.approx(p["realized_price"] / p["reference_price"] - 1)
            assert p["outcome_up"] == (p["realized_price"] > p["reference_price"])
            assert p["in_90"] == (p["q05"] <= p["realized_price"] <= p["q95"])
            assert p["benchmark_return"] is not None

        card = (await api.get("/api/v1/predictions/scorecard")).json()
        by = {(s["source"], s["horizon_days"]): s for s in card["sources"]}
        f5 = by[("forecast", 5)]
        assert f5["resolved"] == len(UNIVERSE) and f5["open"] == 0 and 0 <= f5["brier"] <= 1
        assert 0 <= f5["coverage_90"] <= 1 and sum(b["n"] for b in f5["calibration"]) == len(UNIVERSE)
        assert by[("forecast", 21)]["resolved"] == 0 and by[("forecast", 21)]["open"] == len(UNIVERSE)
        assert by[("model", 21)]["open"] == len(UNIVERSE)
        one = (await api.get("/api/v1/predictions/scorecard", params={"symbol": "aapl"})).json()
        assert one["symbol"] == "AAPL" and all(p["symbol"] == "AAPL" for p in one["recent"])


async def test_ledger_refuses_intraday_and_synthetic(tmp_path):
    clock = FakeClock(datetime(2026, 9, 25, 15, 0, tzinfo=UTC))  # 11:00 New York
    async for api in _client(make_settings(tmp_path, picks_universe=",".join(UNIVERSE[:4])), clock):
        intraday = await api.post("/api/v1/predictions/log")
        assert intraday.status_code == 422 and "after the 4 pm ET close" in intraday.json()["detail"]
        clock.advance(6 * 3600)  # 17:00 New York; offline, so every price is synthetic
        out = (await api.post("/api/v1/predictions/log")).json()
        assert out["logged"] == 0
        assert all("synthetic" in why for why in out["skipped"].values()), out["skipped"]
        assert (await api.get("/api/v1/predictions")).json() == []


async def test_ledger_scheduler(tmp_path, mock_net):
    clock = FakeClock(datetime(2026, 9, 25, 20, 0, tzinfo=UTC))  # 16:00 New York
    async for api in feed_client(tmp_path, clock, predictions_log_time="16:20"):
        service = api.container.predictions
        assert await service.run_scheduled() == "idle"  # before the log time
        clock.advance(25 * 60)
        assert (await service.run_scheduled()).startswith(f"logged {len(UNIVERSE) * 3} for 2026-09-25")
        clock.advance(60)
        assert await service.run_scheduled() == "idle"
        assert await api.container.poller.run_predictions() == "idle"
    async for api in _client(make_settings(tmp_path / "off", predictions_enabled=False), clock):
        assert await api.container.predictions.run_scheduled() == "disabled"


async def test_backfill_replays_history_point_in_time(tmp_path, mock_net):
    clock = FakeClock(FRIDAY_AFTER_CLOSE)
    async for api in feed_client(tmp_path, clock):
        r = await api.post("/api/v1/predictions/backfill", params={"wait": 600})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["forecast_rows"] > 12 * 100 and out["model_rows"] > 12 * 20, out
        assert out["last_date"] < "2026-09-25" and not out["skipped"]

        rows = (
            await api.get(
                "/api/v1/predictions", params={"origin": "backfill", "source": "forecast", "limit": 2000}
            )
        ).json()
        assert rows and all(p["origin"] == "backfill" and p["status"] == "resolved" for p in rows)
        assert all(p["model_version"].startswith("garch-t-fhs/2") for p in rows)
        feed = ConsistentFeed(clock)
        for p in rows[:: max(1, len(rows) // 25)]:
            h = await feed.history(p["symbol"], "1d", clock.now() - timedelta(days=2000), clock.now())
            closes = {b.timestamp.astimezone(UTC).date().isoformat(): b.close for b in h.bars}
            assert p["reference_price"] == pytest.approx(closes[p["made_on"]])
            assert p["realized_price"] == pytest.approx(
                closes[p["target_date"]]
            )  # graded against what happened
            assert p["in_90"] == (p["q05"] <= p["realized_price"] <= p["q95"])
            assert p["q05"] < p["q25"] < p["q50"] < p["q75"] < p["q95"]

        model_rows = (
            await api.get(
                "/api/v1/predictions", params={"origin": "backfill", "source": "model", "limit": 2000}
            )
        ).json()
        by_day: dict[str, list] = {}
        for p in model_rows:
            by_day.setdefault(p["made_on"], []).append(p)
        for day in by_day.values():
            assert sorted(p["rank"] for p in day) == list(range(1, len(day) + 1))
            probs = [p["prob_outperform"] for p in sorted(day, key=lambda p: p["rank"])]
            assert probs == sorted(
                probs, reverse=True
            )  # monotone calibration, better rank -> higher probability

        live = (await api.get("/api/v1/predictions/scorecard", params={"origin": "live"})).json()
        replay = (await api.get("/api/v1/predictions/scorecard", params={"origin": "backfill"})).json()
        assert live["sources"] == [] and replay["origin"] == "backfill"
        by = {(s["source"], s["horizon_days"]): s for s in replay["sources"]}
        assert by[("forecast", 21)]["resolved"] > 100 and 0.5 < by[("forecast", 21)]["coverage_90"] <= 1
        assert by[("model", 21)]["resolved"] == out["model_rows"] and by[("model", 21)]["brier"] is not None

        # Idempotent, replaceable, and it never blocks today's live logging.
        again = (await api.post("/api/v1/predictions/backfill", params={"wait": 600})).json()
        assert again["forecast_rows"] == 0 and again["model_rows"] == 0
        redo = (
            await api.post(
                "/api/v1/predictions/backfill", params={"wait": 600, "replace": True, "sources": "model"}
            )
        ).json()
        assert redo["replaced"] == out["model_rows"] and redo["model_rows"] == out["model_rows"]
        logged = (await api.post("/api/v1/predictions/log")).json()
        assert logged["logged"] == len(UNIVERSE) * 3, logged
        status = (await api.get("/api/v1/predictions/backfill")).json()
        assert status["job"]["status"] == "done" and status["result"]["model_rows"] == out["model_rows"]
        counts = {(c["origin"], c["source"]): c for c in status["counts"]}
        assert counts[("backfill", "forecast")]["resolved"] == out["forecast_rows"]
        assert counts[("live", "forecast")]["open"] == len(UNIVERSE) * 2


async def test_model_warmup_backs_off_after_a_failed_run(api, clock, monkeypatch):
    from quantpulse.core.errors import DomainError

    model = api.container.model

    async def broken(*args, **kwargs):
        raise DomainError("not enough history")

    monkeypatch.setattr(model, "_compute", broken)
    assert await model.warm() == "started"
    job = model.model_job()
    with pytest.raises(DomainError):
        await api.container.jobs.wait(job, 10)
    assert job.status == "failed" and job.error == "not enough history"
    assert (await model.warm()).startswith("last run failed")  # no restart loop every five minutes
    clock.advance(31 * 60)
    assert await model.warm() == "started"
