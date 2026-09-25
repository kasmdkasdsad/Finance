"""The S&P 500 universe end to end: point-in-time membership, former members, reference data and jobs.

A small stand-in index replaces the real S&P 500 (15 current members, one that joined late and two that
left: one acquired, one delisted) so the whole pipeline — bulk prices, SEC industries and earnings,
XBRL fundamentals, the walk-forward models and the background job — runs in seconds."""

import asyncio
import functools
from datetime import UTC, date, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.core.errors import ProviderNoData
from quantpulse.domain.universe import Constituent, IndexChange
from quantpulse.providers import synthetic
from quantpulse.schemas.market import PriceHistory
from quantpulse.schemas.reference import FrameFact
from quantpulse.services import reference as ref_mod
from quantpulse.services.facts import SPECS

from .conftest import _client, make_settings

NOW = datetime(2026, 9, 25, 20, 30, tzinfo=UTC)  # Friday, after the close
ANCHOR = datetime(2027, 3, 31, 21, 0, tzinfo=UTC)
CURRENT = [f"C{i:02d}" for i in range(14)] + ["NEWCO"]
GONE_UNTIL = datetime(2025, 3, 14, 21, 0, tzinfo=UTC)  # acquired two weeks after leaving the index
OLDCO_UNTIL = datetime(2024, 6, 28, 21, 0, tzinfo=UTC)
DELISTED = {"GONE": GONE_UNTIL, "OLDCO": OLDCO_UNTIL}
CHANGES = [
    IndexChange(date(2025, 3, 3), "NEWCO", "New Co", "GONE", "Gone Inc", "Acquired"),
    IndexChange(date(2024, 6, 3), "C13", "C13 Corp", "OLDCO", "Old Co", "Market capitalization change"),
]


@functools.lru_cache(maxsize=64)
def _full(symbol):
    return synthetic.synthetic_history(symbol, "1d", ANCHOR - timedelta(days=2400), ANCHOR, ANCHOR)


class BulkFeed:
    name = "bulk"

    def __init__(self, clock):
        self.clock = clock

    def configured(self):
        return True

    async def histories(self, symbols, interval, start, end):
        out = {}
        for s in symbols:
            last = min(end, self.clock.now(), DELISTED.get(s, ANCHOR))
            bars = [b for b in _full(s).bars if start <= b.timestamp <= last]
            if bars:
                out[s] = PriceHistory(symbol=s, interval=interval, bars=bars)
        return out

    async def history(self, symbol, interval, start, end):
        got = await self.histories([symbol], interval, start, end)
        if symbol not in got:
            raise ProviderNoData(self.name, f"no bars for {symbol}")
        return got[symbol]

    async def quote(self, symbol):
        now = self.clock.now()
        last = (await self.history(symbol, "1d", now - timedelta(days=10), now)).bars[-1]
        return synthetic.synthetic_quote(symbol, now).model_copy(
            update={
                "price": last.close,
                "timestamp": now,
                "previous_close": None,
                "change": None,
                "change_percent": None,
            }
        )


def _cik(symbol):
    return int(synthetic.synthetic_company_events(symbol, date(2020, 1, 1), NOW).profile.cik)


class FakeSec:
    """Industries and earnings for listed companies; delisted tickers are unknown (as on SEC's map)."""

    name = "sec_edgar"

    def __init__(self):
        self.frames = 0

    async def company_events(self, symbol, since):
        if symbol in DELISTED:
            raise ProviderNoData(self.name, f"{symbol} is not an SEC-registered ticker")
        return synthetic.synthetic_company_events(symbol, since, NOW)

    async def frame(self, taxonomy, tag, unit, period):
        self.frames += 1
        field = next(s.field for s in SPECS if s.tag == tag)
        year = int(period[2:6])
        rows = []
        for symbol in CURRENT:
            for f in getattr(synthetic.synthetic_company_facts(symbol, NOW), field):
                if f.end.year != year:
                    continue
                if len(period) > 6 and (f.end.month - 1) // 3 + 1 != int(period[7]):
                    continue
                rows.append(
                    FrameFact(cik=_cik(symbol), start=f.start, end=f.end, value=f.value, accn="0000-00")
                )
        # A published frame always has other filers in it.
        end = date(year, 12, 31) if len(period) == 6 else date(year, 3 * int(period[7]), 28)
        rows.append(FrameFact(cik=1, start=None, end=end, value=1.0, accn="0000-01"))
        return rows


async def _index_client(tmp_path, clock, **overrides):
    settings = make_settings(tmp_path, enable_live_data=True, model_universe="sp500", **overrides)
    async for c in _client(settings, clock):
        container = c.container
        container.market._providers[:] = [BulkFeed(clock)]
        fake_sec = FakeSec()
        container.reference._sec = fake_sec
        container.facts._sec = fake_sec
        constituents = [Constituent(s, f"{s} Corp", "Industrials", None, None, None) for s in CURRENT]
        async with container.db.session() as s:
            await ref_mod.repo.put_blob(
                s, ref_mod.MEMBERSHIP_KEY, ref_mod._snapshot_payload((constituents, CHANGES)), "wikipedia"
            )
        c.fake_sec = fake_sec
        yield c


async def test_sp500_universe_is_point_in_time(tmp_path, mock_net):
    clock = FakeClock(NOW)
    async for api in _index_client(tmp_path, clock, picks_universe="C00,C01,C02"):
        info = (await api.get("/api/v1/model/universe")).json()
        assert info["kind"] == "sp500" and info["current_members"] == 15 and info["changes_logged"] == 2
        assert info["bulk_prices"] is True and info["membership_status"] == "cached"

        r = await api.get("/api/v1/model/report")
        assert r.status_code == 200, r.text
        m = r.json()["data"]
        assert m["data_status"] in {"live", "cached"}
        u = m["universe"]
        assert u["kind"] == "sp500" and u["point_in_time"] is True
        assert u["with_prices"] == 17 and u["current_members"] == 15 and u["former_members"] == 2
        assert not any("survivorship" in w for w in m["warnings"])
        # only today's members are ranked; former members still shaped the history
        assert sorted(x["symbol"] for x in m["live"]) == sorted(CURRENT)
        assert {"GONE", "OLDCO"}.isdisjoint(m["symbols"])
        cov = m["coverage"]
        assert cov["earnings_companies"] == 15 and cov["fundamentals_companies"] == 15
        assert cov["sector_neutral"] is True and cov["fundamentals_status"] == "live"
        assert api.fake_sec.frames > 0

        # The rankings use the close of the last settled session, never a partial intraday bar.
        assert m["as_of"] == "2026-09-25"

        model = api.container.model
        spec = await model.universe(None)
        inputs = await model._inputs(spec, 1825)
        mask = inputs.eligible
        assert not mask.loc["2025-02-28", "NEWCO"] and mask.loc["2025-03-03", "NEWCO"]
        assert mask.loc["2025-02-28", "GONE"] and not mask.loc["2025-03-03", "GONE"]
        assert mask.loc["2024-05-31", "OLDCO"] and not mask.loc["2024-06-03", "OLDCO"]
        assert not mask.loc["2024-05-31", "C13"]

        # Picks ranked by the model draw on its whole universe, not just the three-stock picks list.
        picks = (await api.get("/api/v1/picks/daily", params={"method": "model", "top_n": 3})).json()["data"]
        model_top = [x["symbol"] for x in m["live"][:3]]
        assert [p["symbol"] for p in picks["picks"]] == model_top
        assert any("ranks 15 stocks" in n for n in picks["notes"]) and picks["universe_size"] > 3
        assert all(p["sector"] for p in picks["picks"])


async def test_long_runs_answer_202_with_progress(tmp_path, mock_net):
    clock = FakeClock(NOW)
    async for api in _index_client(tmp_path, clock, model_sync_wait_seconds=0):
        r = await api.get("/api/v1/model/report", params={"wait": 0})
        assert r.status_code == 202 and r.headers["location"].startswith("/api/v1/jobs/model-")
        job = r.json()
        assert job["status"] == "running" and job["kind"] == "model" and 0 <= job["progress"] < 1
        assert (await api.get("/api/v1/model/job")).json()["id"] == job["id"]

        # Picks and stock reports never hang on a training model: they say so and fall back.
        picks = (await api.get("/api/v1/picks/daily", params={"method": "model", "top_n": 3})).json()["data"]
        assert any("still training" in n for n in picks["notes"])

        for _ in range(600):
            state = (await api.get(f"/api/v1/jobs/{job['id']}")).json()
            if state["status"] != "running":
                break
            await asyncio.sleep(0.1)
        assert state["status"] == "done" and state["progress"] == 1.0, state
        assert any(j["id"] == job["id"] for j in (await api.get("/api/v1/jobs")).json())
        done = await api.get("/api/v1/model/report", params={"wait": 0})
        assert done.status_code == 200 and done.json()["data"]["universe"]["kind"] == "sp500"
        assert (await api.get("/api/v1/jobs/nope")).status_code == 404


async def test_previous_close_keeps_serving_while_the_next_run_trains(tmp_path, mock_net):
    clock = FakeClock(NOW)
    async for api in _index_client(tmp_path, clock):
        first = (await api.get("/api/v1/model/report")).json()["data"]
        clock.advance((datetime(2026, 9, 28, 20, 30, tzinfo=UTC) - NOW).total_seconds())  # Monday close
        stale = await api.get("/api/v1/model/report", params={"wait": 0})
        assert stale.status_code == 200 and stale.json()["data"]["as_of"] == first["as_of"] == "2026-09-25"
        job = api.container.model.model_job()
        assert job is not None and not job.done
        with pytest.raises(Exception):  # noqa: B017 - forced refreshes never fall back to an old run
            await api.container.model.report(wait=0, force=True)
        await api.container.jobs.wait(job, 120)
        fresh = (await api.get("/api/v1/model/report", params={"wait": 0})).json()["data"]
        assert fresh["as_of"] == "2026-09-28"
