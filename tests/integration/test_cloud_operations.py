"""Running 24/7 in the cloud, on the fake Alpaca paper API: one supervisor at a time, fail-closed restarts.

Two "processes" here are two complete API containers on the same database (as two cloud instances would be,
or one instance overlapping its own restart). The single-supervisor lease must let only one of them
supervise and send orders — on PostgreSQL as well, when the suite runs with ``QP_TEST_POSTGRES_URL``.
After a restart nothing is sent until recovery has passed; a kill switch survives restarts; an Alpaca or
database outage stops orders instead of producing them.
"""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy.exc import OperationalError

from quantpulse.core.clock import FakeClock
from quantpulse.db.session import Database
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.fakes.market import TrendFeed

from .conftest import NOW
from .test_brain_cycle import WIDE, brain_client, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS, TRADING, brain_orders

ORDERS = "/v2/orders"


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


def order_posts(fake: FakeAlpacaPaper) -> int:
    return sum(1 for m, p in fake.log if m == "POST" and p == ORDERS)


def shared(clock):
    fake = FakeAlpacaPaper(clock=clock)
    return fake, TrendFeed(clock, drifts=WIDE)


# --------------------------------------------------------------------------- one supervisor
async def test_a_second_process_on_the_same_database_stands_by_and_sends_nothing(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for a in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        async for b in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
            assert "cycle" in await a.container.brain.supervisor.tick()
            sent = order_posts(fake)
            assert sent > 0
            standby = await b.container.brain.supervisor.tick()
            assert standby.startswith("standby: another process supervises the Brain")
            clock.advance(60)  # A's lease is live (in production its heartbeat renews it every minute)
            forced = (await b.post(f"{API}/run")).json()  # even a manual cycle on the standby sends nothing
            assert forced["summary"]["orders_sent"] == 0 and order_posts(fake) == sent
            blocked = [
                d for d in forced["decisions"] if d["quantity"] and (d.get("execution") or {}).get("reason")
            ]
            assert blocked and all("order lease" in d["execution"]["reason"] for d in blocked)
            status = (await b.get(f"{API}/supervisor")).json()
            assert status["standby"] and not status["lease"]["mine"] and status["lease"]["live"]
            assert (await a.get(f"{API}/supervisor")).json()["lease"]["mine"]


async def test_concurrent_startup_elects_exactly_one_supervisor(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for a in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        async for b in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
            results = await asyncio.gather(
                a.container.brain.supervisor.tick(), b.container.brain.supervisor.tick()
            )
            active = [r for r in results if not r.startswith("standby")]
            assert len(active) == 1 and "cycle" in active[0], results
            assert sum(r.startswith("standby") for r in results) == 1
            cids = [o["client_order_id"] for o in brain_orders(fake)]
            assert cids and len(cids) == len(set(cids))  # one set of orders, never two
            assert len((await a.get(f"{API}/cycles")).json()) == 1


async def test_after_a_crash_another_process_takes_over_only_when_the_lease_lapses(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "accept"  # orders rest at Alpaca: working orders the survivor must recover
    async for a in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        async for b in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
            await a.container.brain.supervisor.tick()
            working = [o["client_order_id"] for o in brain_orders(fake)]
            assert working
            a.container.lease.stop_heartbeat()  # A "crashes": it stops renewing, without releasing
            clock.advance(60)
            assert (await b.container.brain.supervisor.tick()).startswith("standby")  # A's lease still live
            clock.advance(a.container.lease.ttl.total_seconds())
            done = await b.container.brain.supervisor.tick()  # lapsed: B takes over, recovering first
            assert done.split(", ")[0] == "startup_recovery"
            events = [e["kind"] for e in (await b.get(f"{TRADING}/events")).json()]
            assert "reconciliation_completed" in events
            # the working orders were recovered, not resent: one order per symbol and side in the slot
            cids = [o["client_order_id"] for o in brain_orders(fake)]
            assert len(cids) == len(set(cids)) and set(working) <= set(cids)
            assert (await b.get(f"{API}/supervisor")).json()["lease"]["mine"]
            # A comes back (a paused process): it lost the lease and must not send anything
            assert (await a.container.brain.supervisor.tick()).startswith("standby")
            assert "order lease" in " ".join(await a.container.trading.pre_submit_blockers("brain"))


async def test_a_process_that_lost_its_lease_is_stopped_at_the_last_gate(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for a in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        async for b in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
            assert await a.container.lease.acquire()
            a.container.lease.stop_heartbeat()
            clock.advance(a.container.lease.ttl.total_seconds() + 1)
            assert await b.container.lease.acquire()  # B took it over while A was paused
            blockers = await a.container.trading.pre_submit_blockers("brain")
            assert any("another QuantPulse process holds the order lease" in x for x in blockers)
            assert not await a.container.lease.held() and await b.container.lease.held()


# --------------------------------------------------------------------------- restarts
async def test_the_brain_kill_switch_survives_a_restart_and_cancels_working_orders(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "accept"
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        await api.container.brain.supervisor.tick()
        assert any(o["status"] == "accepted" for o in fake.orders.values())
        r = await api.post(
            f"{API}/kill-switch", json={"active": True, "reason": "from my phone", "cancel_open_orders": True}
        )
        assert r.status_code == 200 and r.json()["active"]
        assert all(o["status"] == "canceled" for o in fake.orders.values())  # working Brain orders canceled
    sent = order_posts(fake)
    clock.advance(31 * 60)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):  # a restart
        ks = (await api.get(f"{API}/kill-switch")).json()
        assert ks["active"] and ks["reason"] == "from my phone" and ks["source"] == "runtime"
        await api.container.brain.supervisor.tick()
        assert order_posts(fake) == sent
        ex = (await api.get(f"{API}/execution")).json()
        assert any(
            "Brain kill switch ON (dashboard/API: from my phone)" in b for b in ex["blockers_scheduled"]
        ), ex


async def test_an_alpaca_outage_at_startup_keeps_the_brain_waiting_and_sends_nothing(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.outage = True
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        sup = api.container.brain.supervisor
        waiting = await sup.tick()
        assert waiting.startswith("waiting: startup recovery has not passed")
        assert (await api.get(f"{API}/cycles")).json() == [] and order_posts(fake) == 0
        status = (await api.get(f"{API}/supervisor")).json()
        assert status["waiting"] and not status["recovered"]
        fake.outage = False  # Alpaca is back: the next tick recovers first, then works
        clock.advance(60)
        done = await sup.tick()
        assert done.split(", ")[0] == "startup_recovery" and "cycle" in done
        # an interruption mid-session: the cycle's execution fails closed, nothing is sent
        sent = order_posts(fake)
        fake.outage = True
        clock.advance(31 * 60)
        await sup.tick()
        assert order_posts(fake) == sent


async def test_a_database_failure_stops_the_tick_not_the_safeguards(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        sup = api.container.brain.supervisor
        real = Database.session

        def down(self):
            raise OperationalError("SELECT 1", {}, Exception("database unreachable"))

        monkeypatch.setattr(Database, "session", down)
        with pytest.raises(OperationalError):
            await sup.tick()  # the lease cannot even be read: nothing runs
        assert order_posts(fake) == 0
        monkeypatch.setattr(Database, "session", real)  # the database is back
        assert "cycle" in await sup.tick() and order_posts(fake) > 0


async def test_the_schedule_and_positions_are_recovered_after_a_restart(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        assert "cycle" in await api.container.brain.supervisor.tick()
        cycles = len((await api.get(f"{API}/cycles")).json())
        held = set(fake.positions)
        assert held
    fake.hold("MIDC", 7, feed.live_price("MIDC"))  # bought by hand while QuantPulse was down
    clock.advance(10 * 60)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):  # a restart
        status = (await api.get(f"{API}/supervisor")).json()
        assert status["last"]["cycle"]  # the schedule survived in the database
        assert status["next_cycle_at"] == (NOW + timedelta(minutes=30)).isoformat()
        done = await api.container.brain.supervisor.tick()
        assert done == "startup_recovery, reconcile, monitor"  # the next full cycle is not due yet
        assert len((await api.get(f"{API}/cycles")).json()) == cycles  # history intact, nothing re-run
        clock.advance(21 * 60)
        await api.container.brain.supervisor.tick()
        pos = (await api.get(f"{API}/positions")).json()
        assert pos["unexpected"] == ["MIDC"] and held <= {p["symbol"] for p in pos["open"]}


# --------------------------------------------------------------------------- another installation
async def test_another_installation_trading_the_same_account_stops_the_brain(tmp_path, monkeypatch):
    """The PC left running next to the cloud: two databases, so no shared lease — the cloud notices QuantPulse
    orders it never placed and turns its Brain kill switch on before sending anything."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "accept"  # the PC's orders are still working
    async for pc in brain_client(tmp_path / "pc", clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        assert "cycle" in await pc.container.brain.supervisor.tick()
        assert brain_orders(fake)
        async for cloud in brain_client(tmp_path / "cloud", clock, fake=fake, feed=feed, **OWNS, **ENABLED):
            sent = order_posts(fake)
            await cloud.container.brain.supervisor.tick()  # startup recovery reconciles first
            kill = (await cloud.get(f"{API}/kill-switch")).json()
            assert kill["active"] and "placed by another installation" in kill["reason"], kill
            events = [e["kind"] for e in (await cloud.get(f"{TRADING}/events")).json()]
            assert "foreign_orders_detected" in events
            clock.advance(31 * 60)
            await cloud.container.brain.supervisor.tick()
            assert order_posts(fake) == sent  # nothing from the cloud while the PC trades


async def test_old_finished_orders_of_a_retired_installation_are_adopted_quietly(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for pc in brain_client(tmp_path / "pc", clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        assert "cycle" in await pc.container.brain.supervisor.tick()
    assert brain_orders(fake) and all(o["status"] == "filled" for o in fake.orders.values())
    clock.advance(3 * 3600)  # the PC was stopped hours ago; its history was not imported
    async for cloud in brain_client(tmp_path / "cloud", clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        await cloud.container.brain.supervisor.tick()
        assert not (await cloud.get(f"{API}/kill-switch")).json()["active"]


async def test_the_kill_switch_works_during_an_alpaca_outage_and_cancels_once_it_answers(
    tmp_path, monkeypatch
):
    """Stopping from the phone must never fail because Alpaca is unreachable: the switch holds at once, and the
    working Brain orders are canceled as soon as Alpaca answers again (the health job retries every minute)."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    fake.default_mode = "accept"
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **OWNS, **ENABLED):
        await api.container.brain.supervisor.tick()
        assert any(o["status"] == "accepted" for o in fake.orders.values())
        fake.outage = True
        body = {"active": True, "reason": "from my phone", "cancel_open_orders": True}
        r = await api.post(f"{API}/kill-switch", json=body)
        assert r.status_code == 200 and r.json()["active"]
        assert any(o["status"] == "accepted" for o in fake.orders.values())  # not reachable yet
        await api.container.health.run()
        assert any(o["status"] == "accepted" for o in fake.orders.values())
        fake.outage = False
        clock.advance(61)
        await api.container.health.run()
        assert all(o["status"] == "canceled" for o in fake.orders.values())
        assert (await api.get(f"{API}/kill-switch")).json()["active"]  # still on: a person releases it
        assert await api.container.trading.retry_pending_cancels() is None  # nothing owed any more
        events = (await api.get(f"{TRADING}/events")).json()
        assert any("once Alpaca answered again" in e["message"] for e in events)
