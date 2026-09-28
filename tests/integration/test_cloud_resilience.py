"""Order idempotency, restart recovery and graceful shutdown in the cloud — against the fake Alpaca paper API.

The same logical decision must never reach Alpaca twice, whatever interrupts it: a timeout, a dropped
connection, a crash at any stage, a supervisor takeover. A restarted or newly elected supervisor recovers
(reconciles, audits) before anything else. A shutdown sends no new order, lets the running work finish,
reconciles, records itself and hands the lease over. Nothing here touches a real Alpaca account.
"""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import select

from quantpulse.core.clock import FakeClock
from quantpulse.db.models import BrainCycleRow, BrokerOrderRow, TradingEventRow
from quantpulse.services.container import LIFECYCLE_KEY
from quantpulse.services.order_manager import SUBMIT_FAILED, SUBMIT_UNKNOWN

from .conftest import NOW
from .test_brain_cycle import with_stock_model
from .test_brain_execution import API, brain_orders
from .test_render_deploy import Crash, order_posts, shared, start


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def unique(fake) -> list[str]:
    cids = [o["client_order_id"] for o in brain_orders(fake)]
    assert len(cids) == len(set(cids)), cids
    return cids


def post_bodies(fake) -> list[str]:
    return [b.get("client_order_id") for b in fake.bodies]


async def orders(container) -> dict[str, BrokerOrderRow]:
    async with container.db.session() as s:
        return {r.client_order_id: r for r in (await s.scalars(select(BrokerOrderRow))).all()}


async def events(container, kind: str) -> list[TradingEventRow]:
    async with container.db.session() as s:
        return list((await s.scalars(select(TradingEventRow).where(TradingEventRow.kind == kind))).all())


# --------------------------------------------------------------------------- idempotency
async def test_decision_order_timeout_retry_never_sends_twice(tmp_path, monkeypatch):
    """Alpaca accepted the orders but every answer was lost (a timeout): each is found by its client id and
    adopted; running the same slot again sends nothing new."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "timeout"
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        await api.container.brain.supervisor.tick()
        cids = unique(fake)
        assert cids
        rows = await orders(api.container)
        assert all(
            rows[c].status == "filled" and rows[c].alpaca_order_id for c in cids
        )  # adopted, not unknown
        fake.default_mode = "fill"
        attempts = len(fake.bodies)
        await api.post(f"{API}/run")  # a retry of the same slot (a double click, a re-run after the timeout)
        assert unique(fake) == cids and len(fake.bodies) == attempts  # not even an attempt
    finally:
        await gen.stop()


async def test_a_dropped_connection_is_never_resent_and_settles_as_failed(tmp_path, monkeypatch):
    """The request never reached Alpaca and the answer is unknown: the order stays unknown (never resent) until
    reconciliation, after its grace period, confirms Alpaca never had it."""
    import requests

    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    real = fake._submit
    tried: list[str] = []

    def drop(request, body):
        tried.append(body["client_order_id"])
        raise requests.exceptions.ConnectionError("connection reset before Alpaca read the request")

    monkeypatch.setattr(fake, "_submit", drop)
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        await api.container.brain.supervisor.tick()
        assert tried and fake.orders == {}
        rows = await orders(api.container)
        assert all(rows[c].status == SUBMIT_UNKNOWN for c in tried)
        monkeypatch.setattr(fake, "_submit", real)
        await api.post(f"{API}/run")  # the same slot again: the unknown ones are not tried again
        assert not set(tried) & set(post_bodies(fake))
        clock.advance(180)  # past the grace period: Alpaca never had them
        await api.container.trading.reconcile("test")
        rows = await orders(api.container)
        assert all(rows[c].status == SUBMIT_FAILED for c in tried)
    finally:
        await gen.stop()


# --------------------------------------------------------------------------- crashes and recovery
async def test_a_leader_that_crashes_in_the_decision_stage_is_replaced_safely(tmp_path, monkeypatch):
    """Leader A dies after deciding, before executing. B waits for A's lease to lapse (never earlier), knows A
    did not hand over, closes A's interrupted cycle, recovers first, and sends each order once."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    a_gen, a = await start(tmp_path, clock, fake, feed)
    real_execute = a.container.brain.executor.execute

    async def die(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(a.container.brain.executor, "execute", die)
    with pytest.raises(Crash):
        await a.container.brain.supervisor.tick()
    assert order_posts(fake) == 0
    a.container.lease.stop_heartbeat()  # the process is gone
    b_gen, b = await start(tmp_path, clock, fake, feed)
    try:
        assert (await b.container.brain.supervisor.tick()).startswith("standby")  # A's lease is still live
        clock.advance(a.container.lease.ttl.total_seconds() + 1)
        done = await b.container.brain.supervisor.tick()
        assert done.split(", ")[0] == "startup_recovery"
        sup = b.container.brain.supervisor
        assert sup.takeover and sup.takeover["previous_holder"] == a.container.lease.holder
        async with b.container.db.session() as s:
            cycles = (await s.scalars(select(BrainCycleRow).order_by(BrainCycleRow.id))).all()
        assert cycles[0].status == "failed" and "interrupted" in (cycles[0].error or "")
        assert "running" not in {c.status for c in cycles}
        unique(fake)
        status = (await b.get(f"{API}/cloud-status")).json()
        assert status["supervisor"]["role"] == "leader"
    finally:
        await b_gen.stop()
        monkeypatch.setattr(a.container.brain.executor, "execute", real_execute)
        await a_gen.stop()


async def test_a_crash_during_the_startup_reconciliation_is_recovered_by_the_next_leader(
    tmp_path, monkeypatch
):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "accept"  # working orders to reconcile
    gen, api = await start(tmp_path, clock, fake, feed)
    await api.container.brain.supervisor.tick()
    working = unique(fake)
    assert working
    await gen.stop()
    clock.advance(60)
    a_gen, a = await start(tmp_path, clock, fake, feed)
    real = a.container.trading.orders.reconcile

    async def die():
        raise Crash()

    monkeypatch.setattr(a.container.trading.orders, "reconcile", die)
    with pytest.raises(Crash):
        await a.container.brain.supervisor.tick()  # dies while reconciling at start-up: nothing sent
    assert unique(fake) == working
    a.container.lease.stop_heartbeat()
    b_gen, b = await start(tmp_path, clock, fake, feed)
    try:
        clock.advance(a.container.lease.ttl.total_seconds() + 1)
        done = await b.container.brain.supervisor.tick()
        assert done.split(", ")[0] == "startup_recovery"
        assert b.container.trading.last_reconciled_at is not None
        assert set(working) <= set(unique(fake))  # the working orders are kept, none sent again
    finally:
        await b_gen.stop()
        monkeypatch.setattr(a.container.trading.orders, "reconcile", real)
        await a_gen.stop()


async def test_a_leader_that_lost_the_lease_recovers_again_before_leading_again(tmp_path, monkeypatch):
    """A long pause: A's lease lapsed and B led meanwhile. When A leads again it trusts nothing it remembers:
    the startup recovery runs again first."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    a_gen, a = await start(tmp_path, clock, fake, feed)
    b_gen, b = await start(tmp_path, clock, fake, feed)
    try:
        assert "startup_recovery" in await a.container.brain.supervisor.tick()
        a.container.lease.stop_heartbeat()
        clock.advance(a.container.lease.ttl.total_seconds() + 1)  # A paused past its lease
        assert "startup_recovery" in await b.container.brain.supervisor.tick()
        assert (await a.container.brain.supervisor.tick()).startswith("standby")
        await b.container.lease.release()  # B hands over (its own deploy)
        again = await a.container.brain.supervisor.tick()
        assert again.split(", ")[0] == "startup_recovery"
        assert a.container.brain.supervisor.takeover is None  # B handed over cleanly
        unique(fake)
    finally:
        await b_gen.stop()
        await a_gen.stop()


# --------------------------------------------------------------------------- graceful shutdown
async def test_a_shutdown_during_an_active_cycle_sends_no_new_order_and_hands_over_cleanly(
    tmp_path, monkeypatch
):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    c = api.container
    sup = c.brain.supervisor
    real = sup._tick

    async def slow_tick():
        await asyncio.sleep(0.3)  # SIGTERM arrives while the cycle is analysing
        return await real()

    monkeypatch.setattr(sup, "_tick", slow_tick)
    tick = asyncio.create_task(sup.tick())
    await asyncio.sleep(0.05)
    holder = c.lease.holder
    await gen.stop()  # the real shutdown: SIGTERM on Render
    assert tick.done() and not tick.cancelled() and "cycle" in tick.result()  # allowed to finish
    assert order_posts(fake) == 0  # ... but it sent nothing new
    blocked = await events(c, "order_blocked_at_submit")
    assert blocked and all("shutting down" in e.message for e in blocked)
    b_gen, b = await start(tmp_path, clock, fake, feed)  # the next instance
    try:
        info = await b.container.lease.info()
        assert info["holder"] == holder and info["released"] and not info["live"]  # handed over, not lapsed
        life = await b.container.brain.store.get_state(LIFECYCLE_KEY)
        mine = life[holder]
        assert mine["clean"] and mine["stopped_at"] and mine["drained"] and mine["reconciled"] == "done"
        clock.advance(b.container.settings.brain_cycle_minutes * 60)  # the next cycle slot
        done = await b.container.brain.supervisor.tick()
        assert done.split(", ")[0] == "startup_recovery" and b.container.brain.supervisor.takeover is None
        assert order_posts(fake) > 0  # the new leader decides again, on fresh data, and trades on its own
        unique(fake)
    finally:
        await b_gen.stop()


async def test_an_order_already_on_its_way_at_shutdown_completes_and_is_recorded(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    c = api.container
    broker = c.broker
    real_submit = broker.submit
    in_flight, go = asyncio.Event(), asyncio.Event()

    async def slow_submit(spec):
        if not in_flight.is_set():
            in_flight.set()
            await go.wait()  # the first order is on the wire when SIGTERM arrives
        return await real_submit(spec)

    monkeypatch.setattr(broker, "submit", slow_submit)
    tick = asyncio.create_task(c.brain.supervisor.tick())
    await asyncio.wait_for(in_flight.wait(), 30)
    stopping = asyncio.create_task(gen.stop())
    await asyncio.sleep(0.05)
    assert c.trading.stopping
    go.set()
    await stopping
    assert tick.done() and not tick.cancelled()
    sent = unique(fake)
    assert len(sent) == 1  # the one in flight completed; every later one was held back
    b_gen, b = await start(tmp_path, clock, fake, feed)
    try:
        rows = await orders(b.container)
        assert rows[sent[0]].status == "filled" and rows[sent[0]].alpaca_order_id
    finally:
        await b_gen.stop()


# --------------------------------------------------------------------------- the last gate
async def test_the_market_closing_during_a_cycle_stops_its_orders(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)  # 10:00 in New York
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    executor = api.container.brain.executor
    real = executor.execute

    async def late(*args, **kwargs):
        clock.advance(timedelta(hours=6, minutes=1).total_seconds())  # the cycle ran past 16:00
        return await real(*args, **kwargs)

    monkeypatch.setattr(executor, "execute", late)
    try:
        await api.container.brain.supervisor.tick()
        assert order_posts(fake) == 0
        blocked = await events(api.container, "order_blocked_at_submit")
        assert all("market is closed" in e.message for e in blocked)
    finally:
        await gen.stop()


@pytest.mark.parametrize("which", ["brain", "trading"])
async def test_a_kill_switch_thrown_immediately_before_submission_stops_the_order(
    tmp_path, monkeypatch, which
):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    trading = api.container.trading
    real = trading.pre_submit_blockers

    async def thrown_now(owner, **kw):
        if which == "brain":
            await trading.set_brain_kill_switch(True, "pressed on the phone", cancel_open_orders=False)
        else:
            await trading.set_kill_switch(True, "pressed on the phone", cancel_open_orders=False)
        return await real(owner, **kw)

    monkeypatch.setattr(trading, "pre_submit_blockers", thrown_now)
    try:
        await api.container.brain.supervisor.tick()
        assert order_posts(fake) == 0
        blocked = await events(api.container, "order_blocked_at_submit")
        assert blocked and all("kill switch ON" in e.message for e in blocked)
    finally:
        await gen.stop()


async def test_a_kill_switch_thrown_mid_cycle_stops_every_order_of_the_cycle(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    executor = api.container.brain.executor
    real = executor.execute

    async def thrown(*args, **kwargs):
        await api.post(f"{API}/kill-switch", json={"active": True, "reason": "mid-cycle"})
        return await real(*args, **kwargs)

    monkeypatch.setattr(executor, "execute", thrown)
    try:
        await api.container.brain.supervisor.tick()
        assert order_posts(fake) == 0
    finally:
        await gen.stop()
