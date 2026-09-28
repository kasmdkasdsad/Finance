"""Deploying on Render: old and new instances overlap, processes stop mid-cycle or crash right after sending an
order, and the cloud status must tell the truth — all against the fake Alpaca paper API (never a real account).

Render starts the new instance while the old one keeps running, switches traffic once the new one is healthy,
and only then sends the old one SIGTERM (it may finish for up to ``maxShutdownDelaySeconds``). These tests drive
exactly that sequence with two complete API containers on one database.
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.db import migrate
from quantpulse.db.models import BrokerOrderRow
from quantpulse.services.container import Container
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed
from tests.pg import database_url

from .conftest import NOW, make_settings
from .test_brain_cycle import WIDE, brain_client, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, brain_orders

ORDERS = "/v2/orders"
STATUS = f"{API}/cloud-status"


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def order_posts(fake: FakeAlpacaPaper) -> int:
    return sum(1 for m, p in fake.log if m == "POST" and p == ORDERS)


def shared(clock, **kw):
    return FakeAlpacaPaper(clock=clock, **kw), TrendFeed(clock, drifts=WIDE)


class Instance:
    """One running instance — an API container with its lifespan — in a task of its own, as on a server:
    ``stop()`` runs the real shutdown (drain, lease release), which is what SIGTERM does on Render."""

    def __init__(self, tmp_path, clock, fake, feed, **overrides):
        self._args = (tmp_path, clock)
        self._kwargs = {"fake": fake, "feed": feed, **{**OWNS, **ENABLED, **overrides}}
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self):
        ready: asyncio.Future = asyncio.get_running_loop().create_future()

        async def run():
            async for api in brain_client(*self._args, **self._kwargs):
                ready.set_result(api)
                await self._stop.wait()

        self._task = asyncio.create_task(run())
        done, _ = await asyncio.wait({self._task, ready}, return_when=asyncio.FIRST_COMPLETED)
        if ready not in done:
            self._task.result()  # the start-up failed: raise its error
        return ready.result()

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            await self._task


async def start(tmp_path, clock, fake, feed, **overrides):
    instance = Instance(tmp_path, clock, fake, feed, **overrides)
    return instance, await instance.start()


class Crash(BaseException):
    """The process dies (a kill, an out-of-memory): nothing after this point runs."""


# --------------------------------------------------------------------------- a deploy
async def test_a_deploy_hands_the_brain_over_without_a_duplicate_or_a_gap(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "accept"  # working orders the new instance must take over
    old_gen, old = await start(tmp_path, clock, fake, feed)
    assert "cycle" in await old.container.brain.supervisor.tick()
    working = [o["client_order_id"] for o in brain_orders(fake)]
    assert working
    # the new version starts while the old one still runs (Render's zero-downtime deploy)
    new_gen, new = await start(tmp_path, clock, fake, feed)
    try:
        assert (await new.container.brain.supervisor.tick()).startswith("standby")
        status = (await new.get(STATUS)).json()  # served by the new instance, true about the old leader
        assert status["supervisor"]["leader"]["live"] and not status["supervisor"]["this_process_is_leader"]
        assert status["supervisor"]["role"] == "standby"
        assert status["supervisor"]["leader"]["holder"] == old.container.lease.holder
        assert (
            status["supervisor"]["last_result"]
            and new.container.lease.holder in status["supervisor"]["standby_processes"]
        )
        # Render: traffic moves to the new instance, then SIGTERM to the old one (a clean stop releases the lease)
        await old_gen.stop()
        clock.advance(60)
        done = await new.container.brain.supervisor.tick()
        assert done.split(", ")[0] == "startup_recovery"  # recovery first, before anything else
        cids = [o["client_order_id"] for o in brain_orders(fake)]
        assert len(cids) == len(set(cids)) and set(working) <= set(cids)  # nothing sent twice
        status = (await new.get(STATUS)).json()
        assert status["supervisor"]["this_process_is_leader"] and status["supervisor"]["recovered"]
        assert status["reconciliation"]["last_success_at"]
    finally:
        await new_gen.stop()


async def test_a_cycle_running_when_the_deploy_stops_the_old_instance_is_allowed_to_finish(
    tmp_path, monkeypatch
):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    sup = api.container.brain.supervisor
    real = sup._tick

    async def slow_tick():
        await asyncio.sleep(0.3)  # a tick still running when the supervisor is drained
        return await real()

    monkeypatch.setattr(sup, "_tick", slow_tick)
    tick = asyncio.create_task(sup.tick())
    await asyncio.sleep(0.05)
    assert await sup.drain(10)  # waits for the tick instead of cutting it off
    assert tick.done() and "cycle" in tick.result() and order_posts(fake) > 0
    assert (await sup.tick()).startswith("stopping")  # nothing new once shutting down
    lease = api.container.lease
    await gen.stop()
    assert not await lease.held()  # handed over at once


async def test_a_cycle_that_outlasts_the_drain_is_cancelled_safely(tmp_path, monkeypatch):
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed, shutdown_drain_seconds=0.2)
    sup = api.container.brain.supervisor

    async def stuck():
        await asyncio.sleep(30)
        return "never"

    monkeypatch.setattr(sup, "_tick", stuck)
    tick = asyncio.create_task(sup.tick())
    await asyncio.sleep(0.05)
    assert not await sup.drain(0.2)  # does not wait forever
    tick.cancel()
    await asyncio.gather(tick, return_exceptions=True)
    await gen.stop()  # the shutdown completes


# --------------------------------------------------------------------------- crashes
async def test_a_crash_right_after_an_order_reached_alpaca_never_sends_it_twice(tmp_path, monkeypatch):
    """The worst moment: Alpaca accepted the order, and the process died before recording the answer. The order
    was recorded before it was sent (write-ahead), so the next supervisor finds it at Alpaca by its client id."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    a_gen, a = await start(tmp_path, clock, fake, feed)
    real_submit = a.container.broker.submit

    async def send_then_die(spec):
        await real_submit(spec)
        raise Crash()

    monkeypatch.setattr(a.container.broker, "submit", send_then_die)
    with pytest.raises(Crash):
        await a.container.brain.supervisor.tick()
    sent = brain_orders(fake)
    assert len(sent) == 1
    cid = sent[0]["client_order_id"]
    async with a.container.db.session() as s:
        row = (
            await s.execute(BrokerOrderRow.__table__.select().where(BrokerOrderRow.client_order_id == cid))
        ).first()
    assert row is not None and row.status == "pending_submit"  # recorded, answer never seen
    a.container.lease.stop_heartbeat()  # the process is gone: it renews nothing, releases nothing
    b_gen, b = await start(tmp_path, clock, fake, feed)
    try:
        clock.advance(a.container.lease.ttl.total_seconds() + 1)  # its lease lapses
        done = await b.container.brain.supervisor.tick()
        assert done.split(", ")[0] == "startup_recovery"
        async with b.container.db.session() as s:
            row = (
                await s.execute(
                    BrokerOrderRow.__table__.select().where(BrokerOrderRow.client_order_id == cid)
                )
            ).first()
        assert row.status == "filled" and row.alpaca_order_id  # resolved from Alpaca, not resent
        cids = [o["client_order_id"] for o in brain_orders(fake)]
        assert cids.count(cid) == 1 and len(cids) == len(set(cids))
    finally:
        await b_gen.stop()
        monkeypatch.setattr(a.container.broker, "submit", real_submit)
        await a_gen.stop()


async def test_a_restart_after_a_partial_fill_keeps_the_order_and_the_position(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "partial"
    gen, api = await start(tmp_path, clock, fake, feed)
    await api.container.brain.supervisor.tick()
    partial = [o for o in brain_orders(fake) if o["status"] == "partially_filled"]
    assert partial
    held = {s: p["qty"] for s, p in fake.positions.items()}
    await gen.stop()  # a restart
    clock.advance(10 * 60)
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        await api.container.brain.supervisor.tick()  # recovery: reconciles the working order and the position
        orders = {o["client_order_id"]: o for o in (await api.get("/api/v1/trading/orders")).json()}
        for o in partial:
            assert orders[o["client_order_id"]]["status"] == "partially_filled"
            assert orders[o["client_order_id"]]["filled_qty"] == pytest.approx(float(o["filled_qty"]))
        pos = (await api.get("/api/v1/trading/positions")).json()  # Alpaca's positions, as reconciled
        assert {p["symbol"]: p["qty"] for p in pos} == pytest.approx(held)
        cids = [o["client_order_id"] for o in brain_orders(fake)]
        assert len(cids) == len(set(cids))
    finally:
        await gen.stop()


# --------------------------------------------------------------------------- the cloud status
@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"alpaca_trading_enabled": False}, "QP_ALPACA_TRADING_ENABLED=false"),
        ({"trading_dry_run": True}, "QP_TRADING_DRY_RUN=true"),
        ({"brain_mode": "research_only"}, "does not manage the Alpaca paper account"),
        ({"brain_kill_switch": True}, "Brain kill switch ON"),
        ({"trading_kill_switch": True}, "kill switch ON"),
        ({"brain_supervisor_enabled": False}, "supervisor is disabled"),
    ],
)
async def test_the_cloud_status_says_exactly_why_the_brain_may_not_trade(
    tmp_path, monkeypatch, overrides, reason
):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed, **overrides)
    try:
        await api.container.brain.supervisor.tick()
        if (
            overrides.get("brain_supervisor_enabled") is not False
        ):  # a cycle started by hand sends nothing either
            forced = await api.post(f"{API}/run")
            assert order_posts(fake) == 0, forced.text[:300]
        else:  # no supervisor: nothing runs by itself (a person may still start a cycle)
            assert order_posts(fake) == 0
        ae = (await api.get(STATUS)).json()["autonomous_execution"]
        assert not ae["permitted"] and any(reason in r for r in ae["reasons"]), ae
    finally:
        await gen.stop()


async def test_the_cloud_status_when_everything_allows_trading(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        before = (await api.get(STATUS)).json()
        assert not before["autonomous_execution"]["permitted"]  # no supervisor tick yet: not yet
        assert any("has not ticked" in r for r in before["autonomous_execution"]["reasons"])
        await api.container.brain.supervisor.tick()
        st = (await api.get(STATUS)).json()
        assert st["autonomous_execution"] == {**st["autonomous_execution"], "permitted": True, "reasons": []}
        assert (
            st["alpaca"]["paper_endpoint_verified"]
            and st["alpaca"]["endpoint"] == "https://paper-api.alpaca.markets"
        )
        assert st["database"]["schema"] == st["database"]["schema_head"] == migrate.head_revision()
        assert st["database"]["schema_at_head"] is True
        assert st["alpaca"]["account"] == "PAPER" and st["alpaca"]["live_trading_possible"] is False
        sup = st["supervisor"]
        assert sup["role"] == "leader" and sup["leader"]["holder"] == api.container.lease.holder
        assert sup["leader"]["clock"] == "process"  # tests drive a fake clock; production uses the database's
        assert sup["leader"]["heartbeat_age_seconds"] == 0 and sup["leader"]["expires_in_seconds"] > 0
        assert st["today"]["brain_orders"] >= 1 and st["latest_decision"]["sent"]
        assert st["switches"]["brain_mode"] == "paper_execution" and not st["switches"]["dry_run"]
    finally:
        await gen.stop()


async def test_market_closed_means_no_orders_and_says_so(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    night = datetime(2026, 9, 25, 7, 0, tzinfo=UTC)  # 03:00 in New York
    clock = FakeClock(night)
    fake, feed = shared(clock)
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        await api.container.brain.supervisor.tick()
        await api.post(f"{API}/run")
        assert order_posts(fake) == 0
        ae = (await api.get(STATUS)).json()["autonomous_execution"]
        assert any("market is closed" in r for r in ae["reasons"])
    finally:
        await gen.stop()


async def test_a_skewed_clock_keeps_the_brain_in_recovery_and_sends_nothing(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    alpaca_clock = FakeClock(NOW + timedelta(minutes=10))  # this server's clock is 10 minutes off Alpaca's
    fake, feed = FakeAlpacaPaper(clock=alpaca_clock), TrendFeed(clock, drifts=WIDE)
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        result = await api.container.brain.supervisor.tick()
        assert result.startswith("waiting: startup recovery has not passed") and "clock_skew" in result
        assert order_posts(fake) == 0
        ae = (await api.get(STATUS)).json()["autonomous_execution"]
        assert any("startup recovery has not passed" in r for r in ae["reasons"])
    finally:
        await gen.stop()


async def test_stale_market_data_means_no_new_buys(tmp_path, monkeypatch):
    """The quote-age limit is not loosened: stale quotes stop new positions, and the status says why."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    feed.quote_age = timedelta(minutes=30)  # far past QP_TRADING_MAX_QUOTE_AGE_SECONDS
    gen, api = await start(tmp_path, clock, fake, feed)
    try:
        await api.container.brain.supervisor.tick()
        buys = [o for o in brain_orders(fake) if o["side"] == "buy"]
        assert buys == []
        assert api.container.settings.trading_max_quote_age_seconds == 600  # unchanged
    finally:
        await gen.stop()


# --------------------------------------------------------------------------- start-up
async def test_the_api_refuses_to_start_on_a_schema_behind_the_code(tmp_path):
    url = database_url(tmp_path / "old.db")
    migrate.upgrade(url, "0021")  # a deploy whose pre-deploy migration did not run
    settings = make_settings(tmp_path, database_url=url, auto_migrate=False)
    c = Container(settings, clock=FakeClock(NOW))
    with pytest.raises(RuntimeError, match="schema is at 0021"):
        await c.startup()
    assert not c.poller.running  # nothing started
    await c.db.dispose()
    await c.http.aclose()


async def test_start_up_waits_for_a_database_that_is_restarting(tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from quantpulse.db.session import Database

    settings = make_settings(tmp_path, db_startup_wait_seconds=30)
    c = Container(settings, clock=FakeClock(NOW))
    real, calls = Database.session, {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OperationalError("SELECT 1", {}, Exception("the database system is starting up"))
        return real(self)

    monkeypatch.setattr(Database, "session", flaky)
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def quick(seconds):
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", quick)
    try:
        await c.startup()  # waits it out: two failures, then it answers
        assert calls["n"] >= 3 and sleeps[:2] == [1.0, 2.0]  # back-off
    finally:
        monkeypatch.setattr(asyncio, "sleep", real_sleep)
        monkeypatch.setattr(Database, "session", real)
        await c.shutdown()


async def test_start_up_gives_up_on_a_database_that_never_answers(tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from quantpulse.db.session import Database

    settings = make_settings(tmp_path, db_startup_wait_seconds=0)
    c = Container(settings, clock=FakeClock(NOW))

    def down(self):
        raise OperationalError("SELECT 1", {}, Exception("could not connect"))

    monkeypatch.setattr(Database, "session", down)
    with pytest.raises(RuntimeError, match="did not answer"):
        await c.startup()
    assert not c.poller.running  # nothing started
    await c.http.aclose()


async def test_structured_log_events_for_the_cloud(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    logs = tmp_path / "logs"
    gen, api = await start(tmp_path, clock, fake, feed, log_json=True, log_dir=str(logs), log_level="INFO")
    await api.container.brain.supervisor.tick()
    await gen.stop()
    lines = [
        json.loads(line) for p in logs.glob("*.log*") for line in p.read_text().splitlines() if line.strip()
    ]
    events = {line.get("event") for line in lines}
    assert {"app.startup", "supervisor.elected", "supervisor.tick", "brain.cycle", "brain.decision",
            "trading.order_submitted", "trading.reconciliation_completed", "app.shutdown"} <= events, events  # fmt: skip
    elected = next(line for line in lines if line.get("event") == "supervisor.elected")
    assert elected["data"]["holder"]
    text = "\n".join(p.read_text() for p in logs.glob("*.log*"))
    for secret in ("test-secret-value-0001", "PKTESTKEYID0001"):
        assert secret not in text
