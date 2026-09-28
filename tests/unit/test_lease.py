"""The single-process lease on a real database (SQLite, or PostgreSQL with QP_TEST_POSTGRES_URL)."""

import asyncio
from datetime import UTC, datetime, timedelta

from quantpulse.core.clock import FakeClock
from quantpulse.services.lease import Lease

NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
TTL = timedelta(seconds=180)


async def test_many_processes_racing_for_the_lease_elect_exactly_one(database):
    clock = FakeClock(NOW)
    leases = [Lease(database, clock, ttl=TTL, holder=f"proc-{i}") for i in range(12)]
    won = await asyncio.gather(*(le.acquire() for le in leases))
    assert sum(won) == 1
    winner = leases[won.index(True)]
    for _ in range(3):  # again and again: the holder renews, nobody else gets it
        results = await asyncio.gather(*(le.acquire() for le in leases))
        assert [le for le, ok in zip(leases, results, strict=True) if ok] == [winner]
    assert await winner.held() and not await leases[(won.index(True) + 1) % 12].held()


async def test_expiry_takeover_and_release(database):
    clock = FakeClock(NOW)
    a, b = Lease(database, clock, ttl=TTL, holder="a"), Lease(database, clock, ttl=TTL, holder="b")
    assert await a.acquire() and not await b.acquire()
    info = await b.info()
    assert info["holder"] == "a" and info["live"] and not info["mine"]
    clock.advance(TTL.total_seconds() - 1)
    assert not await b.acquire()  # still live: never taken early
    assert await a.acquire()  # renewed by its holder
    clock.advance(TTL.total_seconds() + 1)
    assert await b.acquire() and not await a.held()  # lapsed: taken over
    first = (await b.info())["acquired_at"]
    clock.advance(30)
    assert await b.acquire() and (await b.info())["acquired_at"] == first  # a renewal keeps the tenure
    await b.release()
    assert await a.acquire()  # released: taken at once, no waiting for expiry
    assert (await a.info())["acquired_at"] != first
