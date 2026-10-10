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


# --------------------------------------------------------------------------- the database's clock (production)
async def test_with_database_time_skewed_instance_clocks_cannot_split_the_brain(database):
    """Instance B's own clock runs an hour ahead of A's. With process clocks B would think A's lease expired
    long ago; with the database's clock (production) both agree: A holds it."""
    a = Lease(database, FakeClock(NOW), ttl=timedelta(seconds=30), holder="a", db_time=True)
    b = Lease(
        database, FakeClock(NOW + timedelta(hours=1)), ttl=timedelta(seconds=30), holder="b", db_time=True
    )
    assert await a.acquire()
    assert not await b.acquire() and not await b.held()
    b._clock.advance(-7200)  # B's clock jumps backwards two hours (an NTP correction): still nothing changes
    assert not await b.acquire() and await a.held()
    info = await b.info()
    assert info["holder"] == "a" and info["live"] and info["clock"] == "database"
    assert 0 < info["expires_in_seconds"] <= 30


async def test_with_database_time_a_lapsed_lease_is_taken_over_for_real(database):
    a = Lease(database, FakeClock(NOW), ttl=timedelta(seconds=1), holder="a", db_time=True)
    b = Lease(database, FakeClock(NOW), ttl=timedelta(seconds=1), holder="b", db_time=True)
    assert await a.acquire() and not await b.acquire()
    await asyncio.sleep(1.3)  # A stopped renewing (a crash): its lease lapses on the database's clock
    assert await b.acquire() and not await a.held() and not await a.acquire()


async def test_racing_instances_with_database_time_elect_exactly_one(database):
    leases = [Lease(database, FakeClock(NOW + timedelta(minutes=i)), ttl=TTL, holder=f"p{i}", db_time=True)
              for i in range(10)]  # fmt: skip
    won = await asyncio.gather(*(le.acquire() for le in leases))
    assert sum(won) == 1


# --------------------------------------------------------------------------- the database failing
async def test_a_database_failure_during_the_election_elects_nobody(database, monkeypatch):
    from sqlalchemy.exc import OperationalError

    from quantpulse.db.session import Database

    clock = FakeClock(NOW)
    a, b = Lease(database, clock, ttl=TTL, holder="a"), Lease(database, clock, ttl=TTL, holder="b")
    real = Database.session

    def down(self):
        raise OperationalError("UPDATE service_leases", {}, Exception("database unreachable"))

    monkeypatch.setattr(Database, "session", down)
    for lease in (a, b):
        try:
            await lease.acquire()
            raise AssertionError("acquire must not succeed without the database")
        except OperationalError:
            pass
    monkeypatch.setattr(Database, "session", real)
    assert (await b.info())["holder"] is None  # nobody was elected by the failure
    assert await a.acquire() and not await b.acquire()


async def test_a_database_failure_during_renewal_lets_the_lease_lapse_and_the_old_leader_stops(
    database, monkeypatch
):
    """The leader cannot renew (its database connection is gone) for longer than the lease: another instance
    takes over; when the old leader's connection comes back it finds the lease taken and stands by."""
    from sqlalchemy.exc import OperationalError

    clock = FakeClock(NOW)
    a, b = Lease(database, clock, ttl=TTL, holder="a"), Lease(database, clock, ttl=TTL, holder="b")
    assert await a.acquire()
    real_session = database.session
    failing = {"on": True}

    def flaky():
        if failing["on"]:
            raise OperationalError("UPDATE service_leases", {}, Exception("connection reset"))
        return real_session()

    monkeypatch.setattr(a._db, "session", flaky)  # only A's view of the database fails: a partition
    a.start_heartbeat(interval=0.05)
    await asyncio.sleep(0.2)  # renewals fail and are retried; the heartbeat survives them
    assert a._heartbeat is not None and not a._heartbeat.done()
    monkeypatch.setattr(a._db, "session", real_session)
    monkeypatch.setattr(b._db, "session", real_session)
    failing["on"] = False
    clock.advance(TTL.total_seconds() + 1)  # A's last renewal is now older than the lease
    a.stop_heartbeat()
    assert await b.acquire()  # B takes over
    assert not await a.acquire() and not await a.held()  # A, back, finds it taken: it stands by
