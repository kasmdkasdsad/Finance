"""Paper-trading sandbox through the API: accounts, orders, the learning agent, training and the scheduler."""

from datetime import UTC, datetime

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.providers import synthetic

from .conftest import NOW, _client, make_settings

BASE = "/api/v1/sandbox/accounts"
SMALL_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]
MONDAY_10AM_NY = datetime(2026, 9, 28, 14, 0, tzinfo=UTC)


class ReplayFeed:
    """A stand-in for a live market-data vendor: deterministic prices that the gateway treats as LIVE,
    so the non-synthetic code paths can be exercised without network access."""

    name = "replayfeed"

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def configured(self) -> bool:
        return True

    async def quote(self, symbol):
        q = synthetic.synthetic_quote(symbol, self._clock.now())
        return q.model_copy(update={"name": symbol})

    async def history(self, symbol, interval, start, end):
        return synthetic.synthetic_history(symbol, interval, start, end, self._clock.now())


@pytest.fixture
async def feed_api(tmp_path, clock, mock_net):
    """Live-enabled API whose only market provider is :class:`ReplayFeed` (everything else is blocked)."""
    async for client in _client(make_settings(tmp_path, enable_live_data=True), clock):
        client.container.market._providers[:] = [ReplayFeed(clock)]
        yield client


async def _create(client, name="Agent", **fields):
    body = {"name": name, "strategy": {"universe": SMALL_UNIVERSE, "top_k": 3}, **fields}
    r = await client.post(BASE, json=body)
    assert r.status_code == 201, r.text
    return r.json()


async def _journal_kinds(client, account_id):
    return [e["kind"] for e in (await client.get(f"{BASE}/{account_id}/journal")).json()]


# ----------------------------------------------------------------------------- accounts
async def test_account_crud_and_validation(api):
    acct = await _create(api, "Alpha bot", starting_cash=50_000, allow_synthetic=True)
    assert acct["cash"] == acct["starting_cash"] == 50_000 and acct["mode"] == "agent"
    assert acct["universe"] == SMALL_UNIVERSE and acct["periods_learned"] == 0
    assert acct["factor_weights"] == pytest.approx(acct["prior_weights"])
    assert sum(acct["factor_weights"].values()) == pytest.approx(1.0)

    dup = await api.post(BASE, json={"name": "ALPHA   BOT"})
    assert dup.status_code == 422 and "already exists" in dup.json()["detail"]
    bad = await api.post(BASE, json={"name": "x", "strategy": {"weight_floor": 0.5}})
    assert bad.status_code == 422
    assert (
        await api.post(BASE, json={"name": "x", "strategy": {"universe": ["A", "a", "A"]}})
    ).status_code == 422
    assert (await api.post(BASE, json={"name": "x", "starting_cash": 0})).status_code == 422

    patched = await api.patch(
        f"{BASE}/{acct['id']}", json={"name": "Beta bot", "strategy": {"top_k": 2, "slippage_bps": 10}}
    )
    assert patched.status_code == 200
    p = patched.json()
    assert p["name"] == "Beta bot" and p["strategy"]["top_k"] == 2 and p["strategy"]["slippage_bps"] == 10
    assert p["universe"] == api.container.settings.picks_universe  # strategy replaced → default universe

    listed = (await api.get(BASE)).json()
    assert [a["name"] for a in listed] == ["Beta bot"]
    assert "update" in await _journal_kinds(api, acct["id"])

    assert (await api.delete(f"{BASE}/{acct['id']}")).status_code == 204
    assert (await api.get(f"{BASE}/{acct['id']}")).status_code == 404
    assert (await api.post(f"{BASE}/{acct['id']}/step")).status_code == 404
    assert (await api.get(f"{BASE}/999/journal")).status_code == 404


# ----------------------------------------------------------------------------- manual orders
async def test_manual_orders_and_pnl(api):
    acct = await _create(api, "Manual", mode="manual", starting_cash=10_000, allow_synthetic=True)
    url = f"{BASE}/{acct['id']}"

    both = await api.post(
        f"{url}/orders", json={"symbol": "AAPL", "side": "buy", "quantity": 1, "notional": 5}
    )
    assert both.status_code == 422
    buy = await api.post(f"{url}/orders", json={"symbol": "aapl", "side": "buy", "notional": 2_000})
    assert buy.status_code == 201, buy.text
    t = buy.json()
    assert t["symbol"] == "AAPL" and t["source"] == "manual" and t["data_status"] == "synthetic"
    assert t["price"] == pytest.approx(t["reference_price"] * 1.0005)  # 5 bps slippage against the buyer
    assert t["notional"] <= 2_000 + 1e-6

    too_big = await api.post(f"{url}/orders", json={"symbol": "AAPL", "side": "sell", "quantity": 1e6})
    assert too_big.status_code == 422 and "short selling is disabled" in too_big.json()["detail"]
    for too_much in ({"notional": 1e9}, {"quantity": 10_000}):
        no_cash = await api.post(f"{url}/orders", json={"symbol": "MSFT", "side": "buy", **too_much})
        assert no_cash.status_code == 422 and "insufficient cash" in no_cash.json()["detail"]
    msft = await api.post(f"{url}/orders", json={"symbol": "MSFT", "side": "buy", "notional": 3_000})
    assert msft.status_code == 201

    summary = (await api.get(url)).json()
    d = summary["data"]
    held = {p["symbol"]: p for p in d["positions"]}
    assert set(held) == {"AAPL", "MSFT"}
    perf = d["performance"]
    assert perf["cash"] >= 0 and perf["equity"] == pytest.approx(perf["cash"] + perf["invested"])
    assert perf["trades"] == 2 and d["data_status"] == "synthetic"
    assert sum(p["weight"] for p in d["positions"]) == pytest.approx(perf["invested"] / perf["equity"])

    sell = await api.post(
        f"{url}/orders", json={"symbol": "AAPL", "side": "sell", "quantity": held["AAPL"]["quantity"]}
    )
    assert sell.status_code == 201
    s = sell.json()
    # Same quote, so the round trip loses exactly the slippage on both legs.
    assert s["realized_pnl"] == pytest.approx((s["price"] - t["price"]) * t["quantity"])
    assert s["realized_pnl"] < 0
    trades = (await api.get(f"{url}/trades")).json()
    assert [x["side"] for x in trades] == ["sell", "buy", "buy"]  # newest first
    assert (await api.post(f"{url}/step")).status_code == 422  # manual accounts are not run by the agent


async def test_synthetic_prices_are_refused_by_default(api):
    acct = await _create(api, "Careful")
    url = f"{BASE}/{acct['id']}"
    refused = await api.post(f"{url}/orders", json={"symbol": "AAPL", "side": "buy", "quantity": 1})
    assert refused.status_code == 409 and refused.json()["error"] == "synthetic_data"

    step = (await api.post(f"{url}/step")).json()
    assert step["executed"] is False and step["data_status"] == "synthetic"
    assert "only 0 symbol(s) have usable live data" in step["skipped_reason"]
    assert all("synthetic" in why for why in step["excluded"].values())
    await api.post(f"{url}/step")  # the same skip is journalled only once per day
    kinds = await _journal_kinds(api, acct["id"])
    assert kinds.count("skip") == 1
    summary = (await api.get(url)).json()["data"]
    assert summary["performance"]["equity"] == 100_000 and summary["positions"] == []
    equity = (await api.get(f"{url}/equity")).json()
    assert len(equity) == 1 and equity[0]["benchmark_price"] is None  # no fake benchmark baseline


# ----------------------------------------------------------------------------- the agent
async def test_agent_trades_and_learns_across_days(feed_api, clock):
    api = feed_api
    acct = await _create(api, "Learner")
    url = f"{BASE}/{acct['id']}"

    first = (await api.post(f"{url}/step")).json()
    assert first["executed"] is True and first["data_status"] == "live"
    assert first["lessons"] == []  # nothing to learn from yet
    assert 1 <= len(first["targets"]) <= 3
    assert all(w == pytest.approx(0.25) for w in first["targets"].values())  # min(max_position, 1/top_k)
    assert {t["symbol"] for t in first["trades"]} == set(first["targets"])
    assert all(t["side"] == "buy" and t["source"] == "agent" for t in first["trades"])
    assert len(first["candidates"]) == len(SMALL_UNIVERSE)
    assert first["cash"] >= 0.02 * first["equity"] - 1e-6  # the cash buffer is respected

    again = (await api.post(f"{url}/step")).json()
    assert again["executed"] is False and "already traded" in again["skipped_reason"]
    forced = (await api.post(f"{url}/step", params={"force": True})).json()
    assert forced["executed"] is True and forced["lessons"] == []  # no same-day learning
    assert forced["trades"] == []  # same prices, same targets → nothing to do

    clock.advance((MONDAY_10AM_NY - NOW).total_seconds())
    monday = (await api.post(f"{url}/step")).json()
    assert monday["executed"] is True and monday["trading_day"] == "2026-09-28"
    assert {lesson["factor"] for lesson in monday["lessons"]} <= set(acct["prior_weights"])
    assert len(monday["lessons"]) >= 4
    assert all(-1 <= lesson["ic"] <= 1 and lesson["observations"] == 8 for lesson in monday["lessons"])
    assert monday["weights_after"] != monday["weights_before"]
    assert sum(monday["weights_after"].values()) == pytest.approx(
        1.0, abs=1e-5
    )  # six weights rounded to 1e-6

    account = (await api.get(url)).json()["data"]["account"]
    assert account["periods_learned"] == 1 and account["last_decision_on"] == "2026-09-28"
    assert account["factor_weights"] == pytest.approx(monday["weights_after"], abs=1e-6)
    assert set(account["ic_ema"]) == {lesson["factor"] for lesson in monday["lessons"]}

    kinds = await _journal_kinds(api, acct["id"])
    assert kinds[:2] == ["decision", "lesson"] and kinds[-1] == "created"
    lesson_entry = (await api.get(f"{url}/journal")).json()[1]
    assert "Scored the 2026-09-25 decision" in lesson_entry["summary"]

    summary = (await api.get(url)).json()["data"]
    perf = summary["performance"]
    assert perf["benchmark_return"] is not None and perf["snapshots"] == 4  # created + 3 steps
    assert summary["data_status"] in {"live", "cached"}
    assert {p["symbol"] for p in summary["positions"]} == set(monday["targets"])


async def test_agent_can_trade_on_the_stock_model(feed_api):
    api = feed_api
    acct = await _create(
        api, "Model trader", strategy={"universe": SMALL_UNIVERSE, "top_k": 3, "signal": "model"}
    )
    assert acct["strategy"]["signal"] == "model"
    step = (await api.post(f"{BASE}/{acct['id']}/step")).json()
    assert step["executed"] is True and step["data_status"] == "live", step
    assert step["lessons"] == [] and 1 <= len(step["targets"]) <= 3
    report = (await api.get("/api/v1/model/report", params={"symbols": ",".join(SMALL_UNIVERSE)})).json()[
        "data"
    ]
    model_top = [x["symbol"] for x in report["live"] if x["z"] > 0][:3]
    assert sorted(step["targets"]) == sorted(model_top)  # the agent holds exactly the model's top names
    assert [c["symbol"] for c in step["candidates"]] == [x["symbol"] for x in report["live"]][:8]
    decision = (await api.get(f"{BASE}/{acct['id']}/journal")).json()[0]
    assert decision["kind"] == "decision" and decision["details"]["signal"] == "model"


async def test_agent_skips_non_trading_days_unless_forced(feed_api, clock):
    api = feed_api
    acct = await _create(api, "Weekend")
    clock.advance(86_400)  # Saturday
    r = (await api.post(f"{BASE}/{acct['id']}/step")).json()
    assert r["executed"] is False and "not a NYSE trading day" in r["skipped_reason"]
    forced = (await api.post(f"{BASE}/{acct['id']}/step", params={"force": True})).json()
    assert forced["executed"] is True and forced["trading_day"] == "2026-09-26"


async def test_reset_keeps_or_forgets_learning(feed_api, clock):
    api = feed_api
    acct = await _create(api, "Resettable")
    url = f"{BASE}/{acct['id']}"
    await api.post(f"{url}/step")
    clock.advance((MONDAY_10AM_NY - NOW).total_seconds())
    learned = (await api.post(f"{url}/step")).json()["weights_after"]

    kept = (await api.post(f"{url}/reset", params={"keep_learning": True})).json()
    assert kept["cash"] == 100_000 and kept["periods_learned"] == 1 and kept["last_decision_on"] is None
    assert kept["factor_weights"] == pytest.approx(learned, abs=1e-6)
    assert (await api.get(f"{url}/trades")).json() == []
    assert await _journal_kinds(api, acct["id"]) == ["reset"]

    fresh = (await api.post(f"{url}/reset")).json()
    assert fresh["periods_learned"] == 0 and fresh["factor_weights"] == pytest.approx(fresh["prior_weights"])
    assert (await api.get(url)).json()["data"]["positions"] == []


# ----------------------------------------------------------------------------- training
async def test_walk_forward_training_applies_learned_weights(feed_api):
    api = feed_api
    acct = await _create(api, "Student")
    r = await api.post(f"{BASE}/{acct['id']}/train", json={"lookback_days": 730, "rebalance_every": 5})
    assert r.status_code == 200, r.text
    rep = r.json()["data"]
    assert rep["applied"] is True and rep["data_status"] == "live" and rep["warnings"] == []
    assert rep["symbols"] == SMALL_UNIVERSE and rep["benchmark"] == "SPY"
    assert rep["decisions"] >= 40 and rep["trades"] > 0
    assert len(rep["equity_curve"]) == rep["trading_days"]
    assert rep["equity_curve"][0]["strategy"] == rep["equity_curve"][0]["benchmark"] == 100_000
    assert len(rep["weights_history"]) == rep["decisions"]
    assert set(rep["mean_ic"]) == set(rep["prior_weights"])
    assert sum(rep["learned_weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert rep["learned_weights"] != rep["prior_weights"]
    assert rep["strategy"]["max_drawdown"] <= 0 and rep["benchmark_metrics"]["total_return"] is not None

    account = (await api.get(BASE)).json()[0]
    assert account["factor_weights"] == pytest.approx(rep["learned_weights"], abs=1e-6)
    assert account["periods_learned"] == rep["decisions"] - 1  # the first decision has nothing to score
    assert "train" in await _journal_kinds(api, acct["id"])
    again = (await api.post(f"{BASE}/{acct['id']}/train", json={"lookback_days": 730})).json()["data"]
    account = (await api.get(BASE)).json()[0]
    assert account["periods_learned"] == again["decisions"] - 1  # retraining replaces, never double-counts

    short = await api.post(f"{BASE}/{acct['id']}/train", json={"lookback_days": 450, "rebalance_every": 21})
    assert short.status_code == 422 and "increase lookback_days" in short.json()["detail"]
    assert (await api.post(f"{BASE}/{acct['id']}/train", json={"lookback_days": 100})).status_code == 422


async def test_training_on_synthetic_prices_is_not_applied(api):
    acct = await _create(api, "Offline student")
    rep = (await api.post(f"{BASE}/{acct['id']}/train", json={"lookback_days": 730})).json()["data"]
    assert rep["applied"] is False and rep["data_status"] == "synthetic"
    assert any("NOT applied" in w for w in rep["warnings"])
    account = (await api.get(BASE)).json()[0]
    assert account["factor_weights"] == pytest.approx(account["prior_weights"])


# ----------------------------------------------------------------------------- scheduler
async def test_scheduler_trades_once_per_day_and_marks_after_close(tmp_path, mock_net):
    clock = FakeClock(datetime(2026, 9, 25, 13, 0, tzinfo=UTC))  # Friday 09:00 New York
    async for api in _client(make_settings(tmp_path, enable_live_data=True), clock):
        api.container.market._providers[:] = [ReplayFeed(clock)]
        sandbox = api.container.sandbox
        agent_acct = await _create(api, "Auto")
        manual = await _create(api, "Hands-on", mode="manual")
        idle = await _create(api, "Paused", auto_trade=False)

        assert await sandbox.run_scheduled() == "idle (3 account(s))"
        clock.advance(3600)  # 10:00
        assert await sandbox.run_scheduled() == f"#{agent_acct['id']} traded"
        clock.advance(60)
        assert await sandbox.run_scheduled() == "idle (3 account(s))"
        clock.advance(6 * 3600 + 5 * 60)  # 16:06
        report = await sandbox.run_scheduled()
        assert report == ", ".join(f"#{a['id']} marked" for a in (agent_acct, manual, idle))
        assert await sandbox.run_scheduled() == "idle (3 account(s))"
        assert len((await api.get(f"{BASE}/{agent_acct['id']}/equity")).json()) == 3
        assert (await api.get(f"{BASE}/{idle['id']}/trades")).json() == []

        clock.advance(86_400)  # Saturday
        assert await sandbox.run_scheduled() == "market closed today"
        assert await api.container.poller.run_sandbox() == "market closed today"


async def test_scheduler_backs_off_after_a_skip(tmp_path):
    clock = FakeClock(NOW)
    async for api in _client(make_settings(tmp_path), clock):  # offline: synthetic data only
        acct = await _create(api, "Offline auto")
        sandbox = api.container.sandbox
        assert await sandbox.run_scheduled() == f"#{acct['id']} skipped"
        assert await sandbox.run_scheduled() == "idle (1 account(s))"  # backing off
        clock.advance(16 * 60)
        assert await sandbox.run_scheduled() == f"#{acct['id']} skipped"
        assert (await _journal_kinds(api, acct["id"])).count("skip") == 1


async def test_scheduler_can_be_disabled(tmp_path, clock):
    async for api in _client(make_settings(tmp_path, sandbox_scheduler_enabled=False), clock):
        assert await api.container.sandbox.run_scheduled() == "disabled"
