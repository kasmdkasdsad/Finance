"""The reader gateway forwards the read-only key's GETs to the monitoring pages, and nothing else: no other
method, page, key or header, never faster than its rate limit, and never a WebSocket."""

import httpx
import pytest

from quantpulse.reader import MAX_BYTES, Reader

KEY = "k" * 24 + "0123456789abcdef"  # 40 characters
STATUS = "/api/v1/brain/status"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def gateway(seen: list[httpx.Request], key: str | None = KEY, handler=None, **kw) -> httpx.AsyncClient:
    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"path": request.url.path, "query": request.url.query.decode()})

    reader = Reader(key, "http://api:8000", transport=httpx.MockTransport(handler or answer), **kw)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=reader), base_url="http://reader")


async def test_a_get_to_a_monitoring_page_is_forwarded_with_only_the_read_only_key():
    seen: list[httpx.Request] = []
    async with gateway(seen) as c:
        r = await c.get(
            f"{STATUS}?limit=5&x=a%20b",
            headers={"X-API-Key": KEY, "Cookie": "qp_session=abc", "Authorization": "Bearer other",
                     "X-Forwarded-For": "1.2.3.4"},
        )  # fmt: skip
    assert r.status_code == 200 and r.json() == {"path": STATUS, "query": "limit=5&x=a%20b"}
    assert r.headers["cache-control"] == "no-store"
    (sent,) = seen
    assert str(sent.url) == f"http://api:8000{STATUS}?limit=5&x=a%20b"
    assert sent.headers["x-api-key"] == KEY
    for header in ("cookie", "authorization", "x-forwarded-for"):
        assert header not in sent.headers


@pytest.mark.parametrize(
    ("method", "path", "key", "status"),
    [
        ("GET", STATUS, None, 401),
        ("GET", STATUS, "wrong" + KEY, 401),
        ("GET", STATUS, KEY[:-1], 401),
        ("POST", "/api/v1/brain/kill-switch", KEY, 405),
        ("DELETE", "/api/v1/trading/orders/1", KEY, 405),
        ("PUT", STATUS, KEY, 405),
        ("GET", "/api/v1/forecast/AAPL", KEY, 403),
        ("GET", "/api/v1/market/stream", KEY, 403),
        ("GET", "/docs", KEY, 403),
    ],
)
async def test_everything_else_is_refused_before_it_reaches_the_api(method, path, key, status):
    seen: list[httpx.Request] = []
    async with gateway(seen) as c:
        r = await c.request(method, path, headers={"X-API-Key": key} if key else {})
    assert r.status_code == status, r.text
    assert seen == []


@pytest.mark.parametrize(
    "path", ["/api/v1/brain/../trading/test-order", "/api/v1/brain/./status", "//api/v1/brain"]
)
async def test_a_path_that_climbs_out_of_a_section_is_refused(path):
    """As sent by a raw client (httpx would tidy these paths before sending them)."""
    seen: list[httpx.Request] = []
    reader = Reader(KEY, "http://api:8000", transport=httpx.MockTransport(lambda r: seen.append(r)))
    scope = {"type": "http", "method": "GET", "path": path, "query_string": b"",
             "headers": [(b"x-api-key", KEY.encode())]}  # fmt: skip
    status, _, _ = await reader.handle(scope)
    assert status == 403 and seen == []


@pytest.mark.parametrize("key", [None, "", "short-key"])
async def test_without_a_proper_key_the_reader_is_off(key):
    seen: list[httpx.Request] = []
    async with gateway(seen, key=key) as c:
        r = await c.get(STATUS, headers={"X-API-Key": key or "x"})
    assert r.status_code == 503 and "off" in r.json()["detail"] and seen == []


async def test_at_most_sixty_requests_a_minute():
    seen: list[httpx.Request] = []
    clock = Clock()
    async with gateway(seen, clock=clock) as c:
        codes = [(await c.get(STATUS, headers={"X-API-Key": KEY})).status_code for _ in range(61)]
        assert codes[:60] == [200] * 60 and codes[60] == 429
        clock.t += 1.0  # one request a second refills
        assert (await c.get(STATUS, headers={"X-API-Key": KEY})).status_code == 200
        assert (await c.get(STATUS, headers={"X-API-Key": KEY})).status_code == 429
    assert len(seen) == 61


async def test_an_api_that_does_not_answer_or_a_huge_page_is_a_502():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def huge(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (MAX_BYTES + 1))

    for handler, text in ((down, "did not answer"), (huge, "too large")):
        async with gateway([], handler=handler) as c:
            r = await c.get(STATUS, headers={"X-API-Key": KEY})
        assert r.status_code == 502 and text in r.json()["detail"]


async def test_websockets_are_refused():
    sent: list[dict] = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    reader = Reader(KEY, "http://api:8000", transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    scope = {"type": "websocket", "path": "/api/v1/market/ws", "headers": [(b"x-api-key", KEY.encode())]}
    await reader(scope, receive, send)
    assert sent == [{"type": "websocket.close", "code": 4403}]
