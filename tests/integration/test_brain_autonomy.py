"""Autonomous Brain paper trading, end to end on the fake Alpaca paper API and the fake market feed.

The Brain owns the Alpaca paper account and trades it by itself: the supervisor runs its cycles, a pre-trade
execution audit arms execution only when every gate passes, and every order still goes through the trading
service — reconciliation, fresh quotes, the risk engine and the order manager — with the kill switches,
paper endpoint, key, dry run and environment checked again immediately before each submission.

These tests exercise what can stop an order at each of those points, what happens to orders after they are
sent (partial fills, refusals), the near-close review, restart recovery, NO TRADE when the evidence is not
there, and one simulated trading day from pre-market to after hours. No real order is ever sent: the fake
refuses any host but ``paper-api.alpaca.markets``.
"""

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.core.market_calendar import earnings_overnight
from quantpulse.services.trading import BrainOrder, TradingService
from tests.fakes.alpaca_paper import FakeAlpacaPaper

from .conftest import NOW
from .test_brain_cycle import BRAIN as API
from .test_brain_cycle import brain_client, run_cycle, with_stock_model
from .test_brain_execution import ENABLED, OWNS, TRADING, brain_orders, posts, sent

ORDERS = "/v2/orders"


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def order_posts(fake: FakeAlpacaPaper) -> int:
    return sum(1 for m, p in fake.log if m == "POST" and p == ORDERS)


def ny(hour: int, minute: int, day: int = 25) -> datetime:
    """A time on a September 2026 day in New York (EDT, UTC-4)."""
    return datetime(2026, 9, day, hour + 4, minute, tzinfo=UTC)


def set_clock(clock: FakeClock, moment: datetime) -> None:
    clock.advance((moment - clock.now()).total_seconds())


async def executions(api) -> list[dict]:
    return (await api.get(f"{API}/executions")).json()["executions"]


# --------------------------------------------------------------------------- the last gate
async def test_a_kill_switch_flipped_mid_cycle_stops_the_order_at_submission(tmp_path, monkeypatch):
    """The gate and the audit passed; the switch goes on while the cycle runs: nothing reaches Alpaca."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        original = TradingService._risk_and_submit
        flipped: list[str] = []

        async def flip_then_submit(self, t, *args, _done=flipped, _submit=original, **kwargs):
            if not _done:  # someone presses STOP BRAIN ORDERS as the first order is about to go
                await self.set_brain_kill_switch(True, "pressed mid-cycle", False)
                _done.append(t.symbol)
            return await _submit(self, t, *args, **kwargs)

        monkeypatch.setattr(TradingService, "_risk_and_submit", flip_then_submit)
        cycle = await run_cycle(api)
        assert flipped, "the cycle must have reached the submission step"
        assert order_posts(api.fake) == 0 and cycle["summary"]["orders_sent"] == 0
        stopped = [d for d in cycle["decisions"] if d["status"] == "blocked_at_submit"]
        assert stopped and all("Brain kill switch ON" in d["execution"]["error"] for d in stopped)
        events = [
            e for e in (await api.get(f"{TRADING}/events")).json() if e["kind"] == "order_blocked_at_submit"
        ]
        assert events and "Brain kill switch ON" in events[0]["message"]
        assert cycle["summary"]["decision"]["outcome"] == "no_trade"
        assert await executions(api) == []  # never sent: nothing to track


async def test_env_drift_after_start_blocks_new_orders_at_submission(tmp_path, monkeypatch):
    from quantpulse.services import trading as trading_module

    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        original = TradingService._risk_and_submit

        async def drift_then_submit(
            self, t, *args, _submit=original, **kwargs
        ):  # .env edited while the server runs
            monkeypatch.setattr(
                trading_module, "env_file_drift", lambda s: ["QP_TRADING_DRY_RUN: .env says true"]
            )
            return await _submit(self, t, *args, **kwargs)

        monkeypatch.setattr(TradingService, "_risk_and_submit", drift_then_submit)
        cycle = await run_cycle(api)
        assert order_posts(api.fake) == 0 and cycle["summary"]["orders_sent"] == 0
        stopped = [d for d in cycle["decisions"] if d["status"] == "blocked_at_submit"]
        assert stopped and all("QP_TRADING_DRY_RUN" in d["execution"]["error"] for d in stopped)


# --------------------------------------------------------------------------- after the order is sent
async def test_partial_fills_are_tracked_to_their_final_state(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "partial"  # Alpaca fills half at once, the rest rests
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        cycle = await run_cycle(api)
        done = sent(cycle)
        assert done and cycle["summary"]["decision"]["outcome"] == "traded"
        rows = {r["client_order_id"]: r for r in await executions(api)}
        assert set(rows) == {d["execution"]["client_order_id"] for d in done}
        for r in rows.values():
            assert r["status"] == "partially_filled" and r["partial"] and not r["final"]
            assert 0 < r["filled_qty"] < r["qty"] and r["brain_cycle_id"] == cycle["id"]
            assert r["quote_price"] and r["spread_bps"] is not None and r["quote_age_s"] is not None
            assert (
                r["submit_latency_ms"] is not None and r["reason"] and r["consensus"]["stance"] == "bullish"
            )
        for cid in rows:
            fake.complete(cid)  # the rest fills later
        clock.advance(60)
        await api.container.brain.supervisor._reconcile("test")  # what the supervisor does every 5 minutes
        for r in await executions(api):
            assert r["status"] == "filled" and r["final"] and r["filled_qty"] == r["qty"]
            assert r["slippage_bps"] is not None and r["grade"] in ("good", "fair", "poor")
            assert r["seconds_to_fill"] is not None


async def test_an_order_alpaca_refuses_is_recorded_not_retried_in_its_slot(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    fake.default_mode = "reject"
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        cycle = await run_cycle(api)
        refused = [d for d in cycle["decisions"] if d["status"] == "rejected"]
        assert refused and fake.orders == {} and cycle["summary"]["orders_sent"] == 0
        assert cycle["summary"]["decision"]["outcome"] == "no_trade"
        rows = await executions(api)
        assert {r["symbol"] for r in rows} == {d["subject"] for d in refused}
        assert all(r["status"] == "rejected" and r["final"] and r["filled_qty"] == 0 for r in rows)
        tries = order_posts(fake)
        again = await run_cycle(api)  # same slot: the same order is not sent again
        assert order_posts(fake) == tries and again["summary"]["orders_sent"] == 0


async def test_the_risk_engine_still_refuses_what_the_brain_asks_for(tmp_path):
    """A Brain order the risk engine would not allow (far above the position limit) is refused, not sent."""
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        price = api.feed.live_price("UPA")
        huge = BrainOrder(
            "UPA", "buy", round(60_000 / price), price, "buy", "test: 60% of equity in one name"
        )
        tc = await api.container.trading.run_brain([huge], brain_cycle_id=None, scheduled=False)
        [t] = tc.trades
        assert t.stage == "risk_rejected" and not t.approved and order_posts(api.fake) == 0
        failed = {c.name for c in t.checks if not c.passed}
        assert "position_limit" in failed or "order_notional" in failed, failed


# --------------------------------------------------------------------------- no trade is a decision
async def test_without_enough_evidence_the_brain_chooses_no_trade(tmp_path):
    """Prices alone (no stock model, no second source) are not enough for a confident view: no order."""
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **{**OWNS, "brain_use_stock_model": False}, **ENABLED):
        cycle = await run_cycle(api)
        assert cycle["summary"]["orders_sent"] == 0 and order_posts(api.fake) == 0
        assert not [d for d in cycle["decisions"] if d["action"] in ("buy", "increase") and d["quantity"]]
        why = cycle["summary"]["decision"]
        assert why["outcome"] == "no_trade" and why["headline"].startswith("no trade")
        assert why["reasons"] and why["orders"] == []
        assert (await api.get(f"{API}/execution-audit")).json()["latest"] is None  # nothing to audit


def test_earnings_before_the_next_session_counts_the_weekend():
    friday = ny(15, 35)
    assert earnings_overnight(friday, 0) and earnings_overnight(
        friday, 3
    )  # tonight, or Monday before the open
    assert not earnings_overnight(friday, 4) and not earnings_overnight(friday, None)
    tuesday = ny(15, 35, 29)
    assert earnings_overnight(tuesday, 1) and not earnings_overnight(tuesday, 2)
    assert not earnings_overnight(tuesday, -1)


# --------------------------------------------------------------------------- the supervisor
async def test_overlapping_ticks_never_run_the_same_work_twice(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        sup = api.container.brain.supervisor
        first, second = await asyncio.gather(sup.tick(), sup.tick())  # a duplicate scheduler event
        assert "cycle" in first and second.startswith("busy")
        cycles = (await api.get(f"{API}/cycles")).json()
        assert len(cycles) == 1
        cids = [o["client_order_id"] for o in brain_orders(api.fake)]
        assert cids and len(cids) == len(set(cids))


async def test_the_near_close_review_halves_a_holding_before_overnight_earnings_once(tmp_path, monkeypatch):
    from quantpulse.core.gateway import Resolved
    from quantpulse.schemas.common import DataStatus, Provenance
    from quantpulse.schemas.reference import CompanyEvents, CompanyProfile
    from quantpulse.services.reference import ReferenceService

    with_stock_model(monkeypatch)
    prov = Provenance(status=DataStatus.LIVE, provider="test", as_of=NOW, fetched_at=NOW)

    async def events(self, symbol, *, force_refresh=False):
        profile = CompanyProfile(symbol=symbol, cik="0", name=symbol, sector="12", sector_label="Other")
        return Resolved(CompanyEvents(profile=profile, earnings=[], earnings_since=date(2024, 1, 1)), prov)

    async def next_earnings(self, symbol):  # UPA reports Monday before the open; the rest in November
        return (date(2026, 9, 28), "scheduled") if symbol == "UPA" else (date(2026, 11, 20), "estimated")

    monkeypatch.setattr(ReferenceService, "events", events)
    monkeypatch.setattr(ReferenceService, "next_earnings", next_earnings)
    clock = FakeClock(ny(15, 10))  # Friday afternoon
    fake = FakeAlpacaPaper(clock=clock)
    async for api in brain_client(
        tmp_path, clock, fake=fake, brain_catalyst_analysis=True, **OWNS, **ENABLED
    ):
        fake.hold("UPA", 40, 60.0, api.feed.live_price("UPA"))  # on the account when the Brain takes over
        sup = api.container.brain.supervisor
        assert "near_close" not in await sup.tick()  # 15:10: not yet
        assert not [o for o in brain_orders(fake) if o["symbol"] == "UPA"]
        set_clock(clock, ny(15, 31))
        done = await sup.tick()
        assert "near_close" in done
        sells = [o for o in brain_orders(fake) if o["symbol"] == "UPA" and o["side"] == "sell"]
        assert len(sells) == 1 and float(sells[0]["qty"]) == 20  # half, before the weekend's release
        [day] = (await api.get(f"{API}/sessions")).json()["sessions"]
        record = day["near_close"]
        assert "UPA" in record["earnings_overnight"] and "UPA" in record["reduced"], record
        upa = next(h for h in record["holdings"] if h["symbol"] == "UPA")
        assert upa["decision"] == "de_risk" and upa["sent"] and upa["why"][0].startswith("overnight:")
        set_clock(clock, ny(15, 41))  # the next full cycle, still in the last half hour: not halved again
        assert "cycle" in await sup.tick()
        assert len([o for o in brain_orders(fake) if o["symbol"] == "UPA" and o["side"] == "sell"]) == 1
        ledger = [r for r in await executions(api) if r["symbol"] == "UPA"]
        assert ledger and ledger[0]["action"] == "de_risk" and ledger[0]["reason"].startswith("overnight:")


async def test_a_restart_runs_the_startup_audit_before_anything_else(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):
        await api.container.brain.supervisor.tick()
        first = len(brain_orders(fake))
        assert first
    clock.advance(5 * 60)
    async for api in brain_client(tmp_path, clock, fake=fake, **OWNS, **ENABLED):  # the process restarted
        done = await api.container.brain.supervisor.tick()
        assert done.split(", ")[0] == "startup_recovery"
        history = (await api.get(f"{API}/execution-audit")).json()["history"]
        startup = next(a for a in history if a["purpose"] == "startup")
        assert startup["ok"] and {"account", "reconciliation", "environment"} <= {
            c["name"] for c in startup["checks"]
        }
        events = [e["kind"] for e in (await api.get(f"{TRADING}/events")).json()]
        assert "reconciliation_completed" in events and "brain_execution_audit" in events
        # the orders already sent are known (reconciled), not sent again
        cids = [o["client_order_id"] for o in brain_orders(fake)]
        assert len(cids) == len(set(cids))


async def test_live_endpoint_variables_change_nothing(tmp_path, monkeypatch):
    """The SDK's own URL variables point at the live API: the Brain still only reaches the paper host
    (the fake raises on any other host, so a single request elsewhere would fail this test)."""
    for name in ("APCA_API_BASE_URL", "ALPACA_BASE_URL", "QP_ALPACA_BASE_URL"):
        monkeypatch.setenv(name, "https://api.alpaca.markets")
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED):
        await api.container.brain.supervisor.tick()
        assert brain_orders(api.fake)
        audit = (await api.get(f"{API}/execution-audit")).json()["latest"]
        assert (
            audit["endpoint"] == "https://paper-api.alpaca.markets"
            and audit["live_trading_possible"] is False
        )


# --------------------------------------------------------------------------- one trading day
async def test_a_simulated_trading_day(tmp_path, monkeypatch):
    """Pre-market to after hours, driven only by the supervisor's ticks, as the desktop launcher's server
    would run it: prepared before the open, traded while the market is open (only then), reviewed near the
    close, reconciled, graded and recorded after it."""
    with_stock_model(monkeypatch)
    clock = FakeClock(ny(8, 45))
    fake = FakeAlpacaPaper(clock=clock)
    fake.market_open = False
    async for api in brain_client(tmp_path, clock, fake=fake, brain_cycle_minutes=60, **OWNS, **ENABLED):
        sup = api.container.brain.supervisor
        log: list[tuple[str, str, int]] = []

        async def tick_at(moment: datetime, market_open: bool, _sup=sup, _log=log) -> str:
            set_clock(clock, moment)
            fake.market_open = market_open
            done = await _sup.tick()
            _log.append((moment.astimezone(UTC).strftime("%H:%M"), done, order_posts(fake)))
            return done

        # pre-market: verify, prepare — nothing is sent
        done = await tick_at(ny(8, 45), False)
        assert {"startup_recovery", "premarket_check", "premarket"} <= set(done.split(", "))
        assert order_posts(fake) == 0
        # the session: a full cycle every hour, monitoring and reconciliation in between
        moment = ny(9, 35)
        while moment <= ny(15, 50):
            await tick_at(moment, True)
            moment += timedelta(minutes=20)
        opened = order_posts(fake)
        assert opened > 0, log  # the Brain traded by itself: no click, no manual cycle
        assert any("near_close" in d for _, d, _ in log)
        # after the close: nothing is sent any more
        done = await tick_at(ny(16, 45), False)
        assert {"session_close", "learn", "trade_lessons", "review", "improve"} <= set(done.split(", "))
        assert order_posts(fake) == opened
        await tick_at(ny(17, 30), False)
        assert order_posts(fake) == opened

        # every order: a Brain order with a unique id, sent while the market was open, in the ledger
        cids = [o["client_order_id"] for o in brain_orders(fake)]
        assert cids == [o["client_order_id"] for o in fake.submitted()] and len(cids) == len(set(cids))
        for o in fake.orders.values():
            t = datetime.fromisoformat(o["submitted_at"].replace("Z", "+00:00"))
            assert ny(9, 30) <= t <= ny(16, 0), o["submitted_at"]
        ledger = await executions(api)
        assert {r["client_order_id"] for r in ledger} == set(cids)
        assert all(r["final"] and r["status"] == "filled" and r["grade"] for r in ledger)
        # the day's record: pre-market check, near-close decision state, close
        [day] = (await api.get(f"{API}/sessions")).json()["sessions"]
        assert day["premarket"] and day["near_close"] and day["close"]
        assert day["orders_sent"] == len(cids) and day["cycles"] >= 6
        # every cycle says why it traded or why it did not
        cycles = (await api.get(f"{API}/cycles", params={"limit": 100})).json()
        for c in cycles:
            detail = (await api.get(f"{API}/cycles/{c['id']}")).json()
            assert (
                detail["summary"]["decision"]["outcome"] in ("traded", "no_trade")
                and detail["summary"]["decision"]["headline"]
            )
        assert sum(1 for c in cycles if c["trigger"] == "supervisor: scheduled") >= 6
        # the audit that armed it, and the account at the end of the day
        audit = (await api.get(f"{API}/execution-audit")).json()["latest"]
        assert audit["ok"] and audit["endpoint"] == "https://paper-api.alpaca.markets"
        assert (await api.get(f"{TRADING}/status")).json()["owner"] == "brain"
        positions = (await api.get(f"{API}/positions")).json()
        assert {p["symbol"] for p in positions["open"]} == set(fake.positions)
        assert posts(fake) == [ORDERS] * opened  # the only writes were paper orders
