"""The Brain owns the Alpaca paper account (``QP_BRAIN_MODE=paper_execution``): end to end through the HTTP
API, the real ``alpaca-py`` SDK and the fake Alpaca paper API.

Every Brain order must go through the trading service — reconciliation, fresh quotes, the risk engine, the
order manager and every trading switch — and nothing may be sent unless all of them allow it: the Brain
kill switch, a dry run, a disabled trading switch, a paper-endpoint mismatch, entry halts and restarts are
all exercised against the fake's order log.
"""

from datetime import timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.db.models import BrokerOrderRow
from quantpulse.providers.alpaca_trading import NotPaperTrading
from quantpulse.services.order_manager import BRAIN
from tests.fakes.alpaca_paper import FakeAlpacaPaper

from .conftest import NOW
from .test_brain_cycle import BRAIN as API
from .test_brain_cycle import brain_client, by, only_reads, run_cycle, with_stock_model

TRADING = "/api/v1/trading"
ENABLED = {"alpaca_trading_enabled": True, "trading_dry_run": False}
OWNS = {"brain_mode": "paper_execution", "brain_use_stock_model": True}


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def posts(fake: FakeAlpacaPaper) -> list[str]:
    return [path for method, path in fake.log if method == "POST"]


def brain_orders(fake: FakeAlpacaPaper) -> list[dict]:
    return [o for o in fake.submitted() if o["client_order_id"].startswith("qp-brain-")]


def sent(cycle: dict) -> list[dict]:
    return [d for d in cycle["decisions"] if (d.get("execution") or {}).get("sent")]


async def test_the_brain_trades_its_account_through_the_trading_service(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        status = (await api.get(f"{TRADING}/status")).json()
        assert status["owner"] == "brain" and status["mode"] == "paper" and "Brain" in status["mode_banner"]
        cycle = await run_cycle(api)
        orders = brain_orders(api.fake)
        assert orders and orders == api.fake.submitted()  # only Brain orders, nothing else
        done = sent(cycle)
        assert {d["subject"] for d in done} == {o["symbol"] for o in orders}
        assert cycle["summary"]["orders_sent"] == len(done)
        assert "owned by the Brain" in cycle["portfolio"]["owner"]
        for d in done:
            ex = d["execution"]
            assert d["action"] in ("buy", "increase") and d["status"] == "filled"
            assert ex["client_order_id"].startswith("qp-brain-") and ex["alpaca_order_id"]
            assert ex["filled_qty"] == d["quantity"] and ex["risk"] == "approved"
            assert {c["name"] for c in ex["checks"]} >= {"live_data", "position_limit", "buying_power"}
        # the trading service's own record: one cycle, orders tagged as the Brain's, every step audited
        tc = (await api.get(f"{TRADING}/cycles/{done[0]['execution']['trading_cycle_id']}")).json()
        assert (
            tc["trigger"] == "brain:manual" and tc["mode"] == "paper" and tc["cycle_key"].startswith("brain-")
        )
        assert {t["symbol"] for t in tc["trades"]} == {d["subject"] for d in done}
        async with api.container.db.session() as s:
            from sqlalchemy import select

            rows = (await s.scalars(select(BrokerOrderRow))).all()
        assert rows and all(r.strategy == BRAIN and r.cycle_id == tc["id"] for r in rows)
        events = {e["kind"] for e in (await api.get(f"{TRADING}/events")).json()}
        assert {"trade_proposed", "risk_approved", "order_filled"} <= events
        # a manual Brain paper cycle arms scheduled (supervisor) cycles, like a manual strategy cycle
        assert (await api.get(f"{TRADING}/status")).json()["scheduler_armed"]
        assert (await api.get(f"{API}/execution")).json()["blockers_scheduled"] == []


async def test_nothing_is_sent_unless_every_trading_switch_allows_it(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    for name, switches, why in (
        ("disabled", {}, "QP_ALPACA_TRADING_ENABLED=false"),
        ("dry", {"alpaca_trading_enabled": True, "trading_dry_run": True}, "QP_TRADING_DRY_RUN=true"),
        ("killed", {**ENABLED, "trading_kill_switch": True}, "kill switch ON"),
        ("brainkilled", {**ENABLED, "brain_kill_switch": True}, "Brain kill switch ON"),
    ):
        async for api in brain_client(tmp_path / name, clock, **OWNS, **switches):
            price = api.feed.live_price("DNA")
            api.fake.hold("DNA", 10, price / 0.85, price)  # past its stop: a protective exit every time
            cycle = await run_cycle(api)
            trades = [d for d in cycle["decisions"] if d["quantity"]]
            assert trades, name  # it had decisions it would have executed
            for d in trades:
                assert not d["execution"]["sent"] and why in d["execution"]["reason"], (name, d["execution"])
            assert cycle["summary"]["orders_sent"] == 0 and only_reads(api.fake), name


async def test_the_brain_kill_switch_stops_new_brain_orders_at_once(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        r = await api.post(f"{API}/kill-switch", json={"active": True, "reason": "test"})
        assert r.status_code == 200 and r.json()["active"] and r.json()["source"] == "runtime"
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] == 0 and posts(api.fake) == []
        assert all(
            "Brain kill switch ON" in d["execution"]["reason"] for d in cycle["decisions"] if d["quantity"]
        )
        ex = (await api.get(f"{API}/execution")).json()
        assert ex["brain_kill_switch"]["active"] and any(
            "Brain kill switch" in b for b in ex["blockers_manual"]
        )
        # the strategy's trading kill switch is separate and untouched
        assert not (await api.get(f"{TRADING}/status")).json()["kill_switch"]["active"]
        await api.post(f"{API}/kill-switch", json={"active": False})
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] > 0 and brain_orders(api.fake)
        kinds = [e["kind"] for e in (await api.get(f"{TRADING}/events")).json()]
        assert "brain_kill_switch_activated" in kinds and "brain_kill_switch_released" in kinds


async def test_an_env_brain_kill_switch_cannot_be_released_from_the_api(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, brain_kill_switch=True, **OWNS, **ENABLED):
        r = await api.post(f"{API}/kill-switch", json={"active": False})
        assert r.status_code == 422 and "QP_BRAIN_KILL_SWITCH" in r.json()["detail"]
        ks = (await api.get(f"{API}/kill-switch")).json()
        assert ks["active"] and ks["source"] == "env"


async def test_the_strategy_is_only_a_dry_run_while_the_brain_owns_the_account(tmp_path):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        cycle = (await api.post(f"{TRADING}/run")).json()
        assert cycle["mode"] == "dry_run" and cycle["trades"]  # it still plans (a shadow for comparison)
        assert any("the Brain owns the Alpaca paper account" in n for n in cycle["notes"])
        assert posts(api.fake) == []
        assert not (await api.get(f"{TRADING}/status")).json()["scheduler_armed"]  # a dry run arms nothing


async def test_a_brain_order_is_never_sent_twice_in_a_slot_even_after_a_restart(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "accept"  # orders rest unfilled
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        first = await run_cycle(api)
        cids = {o["client_order_id"] for o in brain_orders(fake)}
        assert cids and first["summary"]["orders_sent"] == len(cids)
        second = await run_cycle(api)  # the orders are still working: nothing new for those symbols
        assert {o["client_order_id"] for o in brain_orders(fake)} == cids
        assert second["summary"]["orders_sent"] == 0
        for o in list(fake.orders.values()):  # canceled at Alpaca: the Brain wants the same buys again
            fake._cancel(o)
    clock.advance(60)
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):  # a restart, same database
        third = await run_cycle(api)
        assert {o["client_order_id"] for o in brain_orders(fake)} == cids  # same slot: never resent
        assert third["summary"]["orders_sent"] == 0
        again = [d for d in third["decisions"] if d["status"] == "duplicate_prevented"]
        assert again and all(d["execution"]["client_order_id"] in cids for d in again)
        kinds = [e["kind"] for e in (await api.get(f"{TRADING}/events")).json()]
        assert "duplicate_prevented" in kinds


async def test_a_paper_endpoint_mismatch_fails_closed(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        broker = api.container.trading.broker

        def not_paper() -> str:
            raise NotPaperTrading("refusing to trade: the client points at https://api.alpaca.markets")

        monkeypatch.setattr(broker, "verify_paper_client", not_paper)
        cycle = await run_cycle(api)  # the pre-trade audit stops it before anything is handed on
        assert cycle["summary"]["orders_sent"] == 0 and posts(api.fake) == []
        stopped = [d for d in cycle["decisions"] if d["quantity"] and d["action"] in ("buy", "increase")]
        assert stopped and all(
            "execution audit failed: paper_endpoint" in d["execution"]["reason"] for d in stopped
        )
        audit = (await api.get(f"{API}/execution-audit")).json()["latest"]
        assert not audit["ok"] and any(
            f.startswith("paper_endpoint: refusing to trade") for f in audit["failed"]
        )
        # and the trading service fails closed on its own, whoever calls it
        from quantpulse.services.trading import BrainOrder

        order = BrainOrder("UPA", "buy", 1, api.feed.live_price("UPA"), "buy", "test")
        tc = await api.container.trading.run_brain([order], brain_cycle_id=cycle["id"], scheduled=False)
        assert tc.status == "failed" and "refusing to trade" in (tc.error or "") and posts(api.fake) == []


async def test_entry_halts_hold_back_buys_but_let_exits_through(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        price = api.feed.live_price("DNA")
        fake.hold("DNA", 10, price / 0.85, price)  # 15% under water: past its 8% stop
        fake.last_equity = fake.equity() / 0.95  # down 5% on the day: the 4% daily loss limit is hit
        cycle = await run_cycle(api)
        assert "daily_loss" in cycle["summary"]["entry_halts"]
        d = by(cycle["decisions"])
        assert d["DNA"]["action"] == "close" and d["DNA"]["execution"]["sent"]
        buys = [x for x in cycle["decisions"] if x["action"] in ("buy", "increase") and x["quantity"]]
        assert all(x["status"] == "halted" and "daily loss" in x["execution"]["reason"] for x in buys)
        assert [(o["symbol"], o["side"]) for o in brain_orders(fake)] == [("DNA", "sell")]
        assert "DNA" not in fake.positions


async def test_nothing_is_sent_while_the_market_is_closed(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] == 0 and posts(fake) == []
        for d in cycle["decisions"]:
            if d["quantity"]:
                assert "market is closed" in d["execution"]["reason"]


async def test_a_passing_execution_audit_lets_the_supervisor_trade_on_its_own(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        assert not (await api.get(f"{TRADING}/status")).json()["scheduler_armed"]
        sup = api.container.brain.supervisor
        assert "cycle" in await sup.tick()  # the scheduled full cycle: no click anywhere
        cycle = (await api.get(f"{API}/cycles")).json()[0]
        assert cycle["trigger"] == "supervisor: scheduled" and brain_orders(api.fake)
        audit = (await api.get(f"{API}/execution-audit")).json()["latest"]
        assert audit["ok"] and audit["purpose"] == "pre_trade" and audit["failed"] == []
        names = {c["name"] for c in audit["checks"]}
        assert {"paper_endpoint", "paper_key", "trading_kill_switch", "brain_kill_switch", "environment",
                "reconciliation", "market_open", "clock_skew", "market_data"} <= names  # fmt: skip
        assert (
            audit["endpoint"] == "https://paper-api.alpaca.markets"
            and audit["live_trading_possible"] is False
        )
        assert audit["orders"] and all(o["consensus"] and o["reasons"] for o in audit["orders"])
        assert audit["risk_limits"]["max_spread_bps"] == 30 and audit["agents"]["registered"] >= 17
        assert audit["outcome"]["orders_sent"] == len(brain_orders(api.fake))
        status = (await api.get(f"{TRADING}/status")).json()
        assert status["scheduler_armed"]  # armed by the audit
        events = [e["kind"] for e in (await api.get(f"{TRADING}/events")).json()]
        assert "brain_execution_audit" in events and "paper_armed" in events


async def test_a_failing_execution_audit_sends_nothing_and_arms_nothing(tmp_path, monkeypatch):
    from quantpulse.brain import execution

    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        # only the audit sees the drift here, so it is the audit that must stop the orders
        monkeypatch.setattr(execution, "env_file_drift", lambda s: ["QP_TRADING_DRY_RUN: .env says true"])
        sup = api.container.brain.supervisor
        # at start-up the environment check is a resume gate: supervision does not even begin
        waiting = await sup.tick()
        assert waiting.startswith("waiting: startup recovery has not passed") and "environment" in waiting
        assert (await api.get(f"{API}/cycles")).json() == [] and posts(api.fake) == []
        sup._recovered = True  # the drift appears after a clean start: the pre-trade audit is what stops it
        await sup.tick()
        assert posts(api.fake) == []
        assert not (await api.get(f"{TRADING}/status")).json()["scheduler_armed"]
        audit = (await api.get(f"{API}/execution-audit")).json()["latest"]
        assert not audit["ok"] and audit["failed"] == ["environment: QP_TRADING_DRY_RUN: .env says true"]
        cycle = (await api.get(f"{API}/cycles/{(await api.get(f'{API}/cycles')).json()[0]['id']}")).json()
        stopped = [d for d in cycle["decisions"] if d["action"] in ("buy", "increase") and d["quantity"]]
        assert stopped and all(d["execution"]["reason"].startswith("execution audit failed") for d in stopped)
        monkeypatch.setattr(
            execution, "env_file_drift", lambda s: []
        )  # fixed (a restart): the next cycle trades
        clock.advance(31 * 60)
        await api.container.brain.supervisor.tick()
        assert brain_orders(api.fake) and (await api.get(f"{TRADING}/status")).json()["scheduler_armed"]


async def test_the_first_tick_after_a_restart_closes_interrupted_cycles_and_reconciles(tmp_path):
    from quantpulse.db.models import BrainCycleRow

    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        async with api.container.db.session() as s:
            row = BrainCycleRow(kind="full", trigger="supervisor: scheduled", session="market_open",
                                mode="paper_execution", status="running", started_at=NOW - timedelta(minutes=3))  # fmt: skip
            s.add(row)
            await s.flush()
            stuck = row.id
        sup = api.container.brain.supervisor
        assert "startup_recovery" in await sup.tick()
        old = (await api.get(f"{API}/cycles/{stuck}")).json()
        assert old["status"] == "failed" and "interrupted by a restart" in old["error"]
        events = (await api.get(f"{TRADING}/events")).json()
        assert any(e["kind"] == "reconciliation_completed" and "(startup)" in e["message"] for e in events)
        assert "startup_recovery" not in await sup.tick()  # once per start


async def test_every_position_gets_a_thesis_reconciled_with_alpaca(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        api.fake.hold("UPC", 25, 70.0, api.feed.live_price("UPC"))  # on the account when the Brain takes over
        first = await run_cycle(api)
        bought = {d["subject"] for d in sent(first) if d["action"] == "buy"}
        assert bought
        clock.advance(31 * 60)
        await run_cycle(api)
        pos = (await api.get(f"{API}/positions")).json()
        held = {p["symbol"]: p for p in pos["open"]}
        assert held["UPC"]["origin"] == "inherited" and "took the account over" in held["UPC"]["thesis"]
        for sym in bought:
            t = held[sym]
            assert t["origin"] == "brain" and t["entry_order_id"].startswith("qp-brain-")
            assert t["thesis"] and t["supporting"] and t["stop_price"] and t["horizon_days"] and t["regime"]
            assert (
                t["expected_return"] is None and t["target_price"] is None
            )  # uncalibrated: nothing invented
            assert t["check"]["status"] in ("intact", "weakening", "broken") and t["weight"] > 0
            assert t["entry_decision_id"] is not None
        gone = sorted(bought)[0]  # closed by hand at Alpaca
        qty = api.fake.positions.pop(gone)["qty"]
        api.fake.cash += qty * api.fake.price(gone)
        clock.advance(31 * 60)
        cycle = await run_cycle(api)
        assert gone in cycle["portfolio"]["theses"]["closed"]
        closed = {p["symbol"]: p for p in (await api.get(f"{API}/positions")).json()["closed"]}
        assert closed[gone]["exit_reason"].startswith("closed outside the Brain")


async def test_an_unexpected_position_halts_new_positions_until_it_is_adopted(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS):  # trading disabled: nothing is sent
        await run_cycle(api)  # the Brain takes the (empty) account over
        fake.hold("MIDC", 10, api.feed.live_price("MIDC"))  # then someone buys by hand
        clock.advance(31 * 60)
        cycle = await run_cycle(api)
        assert "unexpected_exposure" in cycle["summary"]["entry_halts"]
        assert (await api.get(f"{API}/positions")).json()["unexpected"] == ["MIDC"]
        halts = (await api.get(f"{API}/execution")).json()["last_cycle"]["entry_halts"]
        assert any(h["code"] == "unexpected_exposure" and "MIDC" in h["reason"] for h in halts)
        r = await api.post(f"{API}/positions/MIDC/adopt")
        assert r.status_code == 200 and r.json()["adopt"] == ["MIDC"]
        clock.advance(31 * 60)
        cycle = await run_cycle(api)
        assert "unexpected_exposure" not in cycle["summary"]["entry_halts"]
        pos = (await api.get(f"{API}/positions")).json()
        assert pos["unexpected"] == [] and {p["symbol"]: p["origin"] for p in pos["open"]} == {
            "MIDC": "adopted"
        }
        assert posts(fake) == []


async def test_a_key_that_is_not_a_paper_key_fails_closed(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, alpaca_api_key_id="AKNOTAPAPERKEY", **OWNS, **ENABLED):
        price = api.feed.live_price("DNA")
        api.fake.hold("DNA", 10, price / 0.85, price)  # even a protective exit waits
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] == 0 and posts(api.fake) == []
        trades = [d for d in cycle["decisions"] if d["quantity"]]
        assert trades and all("does not look like a paper key" in d["execution"]["reason"] for d in trades)
        ex = (await api.get(f"{API}/execution")).json()
        assert any("paper key" in b for b in ex["blockers_manual"])
        assert "AKNOTAPAPERKEY" not in str(ex)  # the key itself is never shown
