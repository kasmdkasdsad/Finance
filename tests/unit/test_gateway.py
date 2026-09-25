import asyncio
from datetime import UTC, datetime

from quantpulse.core.cache import TTLCache
from quantpulse.core.clock import FakeClock
from quantpulse.core.errors import ProviderHTTPError, ProviderRateLimited
from quantpulse.core.gateway import DataGateway, Source
from quantpulse.schemas.common import DataStatus


def make_gateway(**kw):
    clock = FakeClock()
    gw = DataGateway(TTLCache(clock=clock), clock=clock, stale_grace_seconds=3600, **kw)
    return gw, clock


def ok(value, counter=None):
    async def fetch():
        if counter is not None:
            counter.append(1)
        return value

    return fetch


def fail(exc):
    async def fetch():
        raise exc

    return fetch


async def test_live_then_cached():
    gw, clock = make_gateway()
    calls = []
    sources = [Source("p1", ok("v", calls))]
    first = await gw.resolve("k", sources, lambda: "synthetic", ttl=10)
    assert first.status is DataStatus.LIVE and first.value == "v" and first.provenance.provider == "p1"
    second = await gw.resolve("k", sources, lambda: "synthetic", ttl=10)
    assert second.status is DataStatus.CACHED and len(calls) == 1
    clock.advance(11)
    third = await gw.resolve("k", sources, lambda: "synthetic", ttl=10)
    assert third.status is DataStatus.LIVE and len(calls) == 2


async def test_failover_order_and_attempt_log():
    gw, _ = make_gateway()
    sources = [
        Source("unconfigured", ok("x"), configured=False),
        Source("broken", fail(ProviderHTTPError("broken", 500, "boom"))),
        Source("good", ok("v")),
    ]
    res = await gw.resolve("k", sources, lambda: "synthetic", ttl=10)
    assert res.value == "v" and res.provenance.provider == "good"
    attempts = [(a.provider, a.ok) for a in res.provenance.attempts]
    assert attempts == [("unconfigured", False), ("broken", False), ("good", True)]


async def test_stale_then_archive_then_synthetic():
    gw, clock = make_gateway()
    good = [Source("p", ok("v"))]
    bad = [Source("p", fail(ProviderHTTPError("p", 503, "down")))]
    await gw.resolve("k", good, lambda: "s", ttl=10)
    clock.advance(20)
    stale = await gw.resolve("k", bad, lambda: "s", ttl=10)
    assert stale.status is DataStatus.STALE and stale.value == "v"
    assert "down" in stale.provenance.message

    archived_at = datetime(2026, 1, 1, tzinfo=UTC)

    async def archive():
        return "from-db", archived_at, "treasury"

    res = await gw.resolve("other", bad, lambda: "s", ttl=10, archive=archive)
    assert res.status is DataStatus.STALE and res.value == "from-db"
    assert res.provenance.provider == "warehouse:treasury" and res.provenance.as_of == archived_at

    syn = await gw.resolve("third", bad, lambda: "s", ttl=10)
    assert syn.status is DataStatus.SYNTHETIC and syn.value == "s"
    assert "p: HTTP 503" in syn.provenance.message


async def test_circuit_breaker_skips_failing_provider():
    gw, clock = make_gateway(failure_threshold=2, cooldown_seconds=60)
    calls = []

    async def flaky():
        calls.append(1)
        raise ProviderHTTPError("p", 500, "err")

    for i in range(3):
        await gw.resolve(f"k{i}", [Source("p", flaky)], lambda: "s", ttl=1)
    assert len(calls) == 2  # third request short-circuited
    res = await gw.resolve("k9", [Source("p", flaky)], lambda: "s", ttl=1)
    assert "circuit open" in res.provenance.attempts[0].error
    clock.advance(61)
    await gw.resolve("k10", [Source("p", flaky)], lambda: "s", ttl=1)
    assert len(calls) == 3  # half-open probe


async def test_rate_limit_opens_breaker_for_retry_after():
    gw, clock = make_gateway(cooldown_seconds=10)
    await gw.resolve("a", [Source("p", fail(ProviderRateLimited("p", retry_after=300)))], lambda: "s", ttl=1)
    assert gw.breaker("p").state.value == "open"
    clock.advance(100)
    assert gw.breaker("p").state.value == "open"


async def test_live_disabled_goes_straight_to_fallback():
    gw, _ = make_gateway(live_enabled=False)
    calls = []
    res = await gw.resolve("k", [Source("p", ok("v", calls))], lambda: "s", ttl=10)
    assert res.status is DataStatus.SYNTHETIC and not calls


async def test_unexpected_exceptions_are_contained():
    gw, _ = make_gateway()
    res = await gw.resolve("k", [Source("p", fail(KeyError("missing")))], lambda: "s", ttl=10)
    assert res.status is DataStatus.SYNTHETIC
    assert "KeyError" in res.provenance.attempts[0].error


async def test_on_live_hook_failure_does_not_break_response():
    gw, _ = make_gateway()

    async def hook(value, provider):
        raise RuntimeError("db down")

    res = await gw.resolve("k", [Source("p", ok("v"))], lambda: "s", ttl=10, on_live=hook)
    assert res.status is DataStatus.LIVE


async def test_concurrent_requests_share_one_fetch():
    gw, _ = make_gateway()
    calls = []

    async def slow():
        calls.append(1)
        await asyncio.sleep(0.01)
        return "v"

    results = await asyncio.gather(
        *[gw.resolve("k", [Source("p", slow)], lambda: "s", ttl=10) for _ in range(20)]
    )
    assert len(calls) == 1
    assert {r.value for r in results} == {"v"}


async def test_force_refresh_bypasses_cache():
    gw, _ = make_gateway()
    calls = []
    src = [Source("p", ok("v", calls))]
    await gw.resolve("k", src, lambda: "s", ttl=100)
    res = await gw.resolve("k", src, lambda: "s", ttl=100, force_refresh=True)
    assert res.status is DataStatus.LIVE and len(calls) == 2
    snap = gw.snapshot()
    assert snap["providers"]["p"]["successes"] == 2


async def test_fallback_is_memoised_briefly_but_never_blocks_recovery():
    gw, clock = make_gateway()
    generated = []

    def synth():
        generated.append(1)
        return f"s{len(generated)}"

    bad = [Source("p", fail(ProviderHTTPError("p", 503, "down")))]
    first = await gw.resolve("k", bad, synth, ttl=300)
    second = await gw.resolve("k", bad, synth, ttl=300)
    assert first.value == second.value == "s1" and len(generated) == 1
    assert second.status is DataStatus.SYNTHETIC
    # A recovered provider is used immediately — the memo lives under a separate key.
    live = await gw.resolve("k", [Source("p", ok("live"))], synth, ttl=300)
    assert live.status is DataStatus.LIVE and live.value == "live"
    clock.advance(31)
    await gw.resolve("other", bad, synth, ttl=300)
    await gw.resolve("other", bad, synth, ttl=300)
    assert len(generated) == 2
