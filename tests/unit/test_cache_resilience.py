import asyncio

import pytest

from quantpulse.core.cache import SingleFlight, TTLCache
from quantpulse.core.circuit_breaker import BreakerState, CircuitBreaker
from quantpulse.core.clock import FakeClock
from quantpulse.core.errors import ProviderRateLimited
from quantpulse.core.http import parse_retry_after
from quantpulse.core.rate_limit import TokenBucket


def test_ttl_expiry_and_stale_window():
    clock = FakeClock()
    cache = TTLCache(clock=clock)
    cache.set("k", 1, ttl=10, stale_ttl=100)
    assert cache.get("k").value == 1
    clock.advance(11)
    assert cache.get("k") is None
    assert cache.get_stale("k").value == 1
    clock.advance(100)
    assert cache.get_stale("k") is None
    assert len(cache) == 0


def test_lru_eviction_and_stats():
    cache = TTLCache(max_entries=2, clock=FakeClock())
    cache.set("a", 1, 60)
    cache.set("b", 2, 60)
    cache.get("a")  # a is now most recently used
    cache.set("c", 3, 60)
    assert "a" in cache and "c" in cache and "b" not in cache
    assert cache.stats.evictions == 1
    assert cache.invalidate_prefix("a") == 1
    snap = cache.snapshot()
    assert snap["entries"] == 1 and snap["hits"] == 1


async def test_single_flight_coalesces_and_survives_leader_cancellation():
    flight = SingleFlight()
    calls = 0
    release = asyncio.Event()

    async def work():
        nonlocal calls
        calls += 1
        await release.wait()
        return 42

    leader = asyncio.create_task(flight.run("x", work))
    await asyncio.sleep(0)
    followers = [asyncio.create_task(flight.run("x", work)) for _ in range(5)]
    await asyncio.sleep(0)
    leader.cancel()
    release.set()
    assert await asyncio.gather(*followers) == [42] * 5
    assert calls == 1
    assert flight.inflight == 0


async def test_single_flight_propagates_errors():
    flight = SingleFlight()

    async def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await flight.run("y", boom)
    assert flight.inflight == 0


async def test_token_bucket_waits_then_fails_fast():
    clock = FakeClock()
    slept = []

    async def fake_sleep(s):
        slept.append(s)
        clock.advance(s)

    bucket = TokenBucket("p", rate=2.0, capacity=2, max_wait=1.0, clock=clock, sleep=fake_sleep)
    await bucket.acquire()
    await bucket.acquire()
    await bucket.acquire()  # needs 0.5s
    assert slept == [pytest.approx(0.5)]
    bucket.penalize(30)
    with pytest.raises(ProviderRateLimited):
        await bucket.acquire()
    assert not bucket.try_acquire()
    clock.advance(31)
    assert bucket.try_acquire()


def test_circuit_breaker_transitions():
    clock = FakeClock()
    br = CircuitBreaker("p", failure_threshold=2, cooldown_seconds=30, clock=clock)
    br.record_failure("e1")
    assert br.state is BreakerState.CLOSED
    br.record_failure("e2")
    assert br.state is BreakerState.OPEN and not br.allow()
    clock.advance(30)
    assert br.state is BreakerState.HALF_OPEN and br.allow()
    br.record_failure("e3")  # a failed probe re-opens immediately
    assert br.state is BreakerState.OPEN
    clock.advance(30)
    br.record_success()
    assert br.state is BreakerState.CLOSED
    br.record_failure("429", cooldown_hint=120)  # server back-off opens at once, for longer
    assert br.state is BreakerState.OPEN
    clock.advance(60)
    assert br.state is BreakerState.OPEN


def test_parse_retry_after():
    assert parse_retry_after("7") == 7
    assert parse_retry_after(None) is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0
    assert parse_retry_after("garbage") is None
