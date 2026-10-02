"""The server's supervisor watchdog, on the fake Alpaca paper API.

A stalled supervisor (a tick that never returns, while the process still answers HTTP) is reported as
``stalled``; the watchdog's cure is a restart — the same graceful stop a deploy does, then a fresh process whose
first tick is the startup recovery. Asking the watchdog, however often and in whatever state, never ticks the
supervisor and never sends an order.
"""

import asyncio

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.workers.poller import Poller

from .conftest import NOW
from .test_brain_cycle import brain_client, with_stock_model
from .test_brain_execution import ENABLED, OWNS, TRADING
from .test_cloud_operations import order_posts, shared

WATCHDOG = "/api/v1/system/watchdog"
FAST_STOP = {"shutdown_drain_seconds": 0.2, "polling_enabled": True}
# the restart test's clock moves 22 minutes: with 30-minute cycles the next one is not due yet, so any order sent
# after the restart could only be a re-send (the default 5-minute cadence would legitimately trade again)
SLOW_CYCLES = {"brain_cycle_minutes": 30}


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


@pytest.fixture(autouse=True)
def _scheduler_driven_by_the_test(monkeypatch):
    """The background scheduler counts as running (the watchdog checks it) but runs no job: each test asks for
    every tick itself, as the poller would once a minute."""

    def start(self):
        self._tasks = [asyncio.create_task(asyncio.Event().wait(), name="scheduler")]

    monkeypatch.setattr(Poller, "start", start)


async def verdict(api) -> dict:
    r = await api.get(WATCHDOG)
    assert r.status_code == 200, r.text
    return r.json()


async def test_a_stalled_supervisor_is_detected_and_a_restart_recovers_it(tmp_path, monkeypatch):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for a in brain_client(
        tmp_path, clock, fake=fake, feed=feed, **FAST_STOP, **SLOW_CYCLES, **OWNS, **ENABLED
    ):
        sup = a.container.brain.supervisor
        assert (await verdict(a))["verdict"] == "starting"
        assert "cycle" in await sup.tick()  # the leader has traded normally
        sent = order_posts(fake)
        assert sent > 0 and (await verdict(a))["verdict"] == "ok"

        inside, never = asyncio.Event(), asyncio.Event()

        async def stuck(
            inside=inside, never=never
        ) -> str:  # an await that never completes (a hung connection)
            inside.set()
            await never.wait()
            return "unreachable"

        monkeypatch.setattr(sup, "_tick", stuck)
        clock.advance(60)
        hung = asyncio.create_task(sup.tick())
        a.container.poller._tasks.append(hung)  # the poller's Brain job, stuck inside the tick
        await inside.wait()
        assert sup.tick_started_at is not None
        running = await verdict(a)
        assert (running["verdict"], running["restart"]) == ("ok", False)  # a tick in progress is work

        clock.advance(21 * 60)  # the market is open: 20 minutes is the limit for one tick
        stalled = await verdict(a)
        assert (stalled["verdict"], stalled["restart"]) == ("stalled", True), stalled
        assert "hung: the current tick has run for 21 min" in stalled["reason"]
        assert stalled["paper"] is True
        for _ in range(5):  # the watchdog polling it changes nothing
            assert (await verdict(a))["verdict"] == "stalled"
        assert order_posts(fake) == sent and not hung.done()
    # leaving the block is the watchdog's restart (SIGTERM): the drain times out on the hung tick, which is
    # cancelled; the leader reconciles once more and hands the lease over
    assert hung.cancelled()

    clock.advance(60)
    async for b in brain_client(
        tmp_path, clock, fake=fake, feed=feed, **FAST_STOP, **SLOW_CYCLES, **OWNS, **ENABLED
    ):
        sup = b.container.brain.supervisor
        assert (await verdict(b))["verdict"] == "starting"
        done = await sup.tick()
        assert done.split(", ")[0] == "startup_recovery", done  # recovery before any other work
        assert order_posts(fake) == sent  # the next full cycle is not due yet: nothing re-sent
        events = [e["kind"] for e in (await b.get(f"{TRADING}/events")).json()]
        assert "reconciliation_completed" in events
        assert sup.takeover is None  # the lease was handed over, not taken from a crashed holder
        healthy = await verdict(b)
        assert (healthy["verdict"], healthy["restart"]) == ("ok", False) and healthy["leader"]
        assert healthy["last_cycle"]["status"] == "completed"


async def test_the_watchdog_never_ticks_the_supervisor_or_sends_an_order(tmp_path, monkeypatch):
    """The account is ready to trade (paper keys, execution enabled, the market open, a cycle due): a real tick
    would send orders. Asking the watchdog — healthy, stalled or starting — sends none and runs nothing."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    fake, feed = shared(clock)
    async for api in brain_client(tmp_path, clock, fake=fake, feed=feed, **FAST_STOP, **OWNS, **ENABLED):
        sup = api.container.brain.supervisor
        seen: list[str] = []
        for minutes in (0, 5, 6, 20):  # nothing asks for a tick: starting, then (after 10 minutes) stalled
            clock.advance(minutes * 60)
            seen.append((await verdict(api))["verdict"])
            await api.get("/api/v1/system/health", params={"fresh": "true"})  # what `qp status` reads too
        assert seen == ["starting", "starting", "stalled", "stalled"]
        assert order_posts(fake) == 0 and fake.orders == {}
        assert sup.last_attempt_at is None and sup.last_tick_at is None  # never ticked
        assert (await api.get("/api/v1/brain/cycles")).json() == []
        # the state was ready: the first real tick recovers, then trades
        assert "cycle" in await sup.tick() and order_posts(fake) > 0
