import httpx
import pytest
import respx

from quantpulse.core.clock import FakeClock
from quantpulse.core.errors import ProviderError, ProviderHTTPError, ProviderParseError, ProviderRateLimited
from quantpulse.core.http import HttpClient
from quantpulse.core.rate_limit import TokenBucket

URL = "https://api.example.test/data"


@respx.mock
async def test_retries_transient_5xx_then_succeeds(http):
    route = respx.get(URL).mock(side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": 1})])
    assert await http.get_json("p", URL) == {"ok": 1}
    assert route.call_count == 2


@respx.mock
async def test_4xx_is_not_retried(http):
    route = respx.get(URL).mock(return_value=httpx.Response(403, json={"status": "NOT_AUTHORIZED"}))
    with pytest.raises(ProviderHTTPError) as err:
        await http.get_json("p", URL)
    assert err.value.status_code == 403 and "NOT_AUTHORIZED" in err.value.message
    assert route.call_count == 1


@respx.mock
async def test_429_penalises_bucket_and_raises():
    clock = FakeClock()
    bucket = TokenBucket("p", rate=10, capacity=10, max_wait=1.0, clock=clock)
    client = HttpClient(limiters={"p": bucket}, backoff_base=0.0)
    respx.get(URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "120"}))
    with pytest.raises(ProviderRateLimited) as err:
        await client.get_json("p", URL)
    assert err.value.retry_after == 120
    assert bucket.blocked_for == pytest.approx(120)
    with pytest.raises(ProviderRateLimited):  # subsequent calls fail fast without touching the network
        await client.get_json("p", URL)
    await client.aclose()


@respx.mock
async def test_network_errors_and_bad_json(http):
    respx.get(URL).mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(ProviderError, match="network error"):
        await http.get_json("p", URL)
    respx.get(URL).mock(return_value=httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(ProviderParseError):
        await http.get_json("p", URL)


@respx.mock
async def test_concurrency_limit_serialises_requests():
    import asyncio

    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={})

    respx.get(URL).mock(side_effect=handler)
    client = HttpClient(concurrency={"p": 2})
    await asyncio.gather(*(client.get_json("p", URL) for _ in range(8)))
    assert peak == 2
    await client.aclose()
