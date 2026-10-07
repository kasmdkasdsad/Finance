"""The reader gateway: the read-only door to QuantPulse on your private Tailscale network.

``./qp reader on`` runs it beside the API (the compose service ``reader``) and publishes it on the tailnet only
(``tailscale serve --https=8443``); the API itself stays bound to the server. A request is forwarded only when
all of these hold, and refused otherwise:

* it carries the read-only key (``X-API-Key``), compared in constant time;
* it is a GET to one of the monitoring pages (``quantpulse.readonly``, the same rule the API applies to the key);
* it is within the rate limit (60 a minute), so reading cannot load the server the Brain runs on.

Only the path, the query and the read-only key are forwarded: no cookie, no other header. The gateway holds no
Alpaca key, no database password and not the API's full token, so even a compromised gateway can only read.
The API checks the key and the page again on its side. WebSockets are refused, and the streams are not
monitoring pages.
"""

from __future__ import annotations

import hmac
import json
import os
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import httpx

from quantpulse.readonly import readable

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

RATE = 60  # requests a minute
TIMEOUT = 60.0
MAX_BYTES = 32 * 1024 * 1024
MIN_KEY = 32  # as the preflight requires of the read-only key


def _error(status: int, detail: str) -> tuple[int, bytes, bytes]:
    return status, json.dumps({"detail": detail}).encode(), b"application/json"


def _header(scope: Scope, name: bytes) -> bytes | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return bytes(value)
    return None


class Bucket:
    """A token bucket: ``rate`` requests a minute, in bursts of at most ``rate``."""

    def __init__(self, rate: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate, self.clock = rate, clock
        self.tokens, self.at = float(rate), clock()

    def take(self) -> bool:
        now = self.clock()
        self.tokens = min(float(self.rate), self.tokens + (now - self.at) * self.rate / 60.0)
        self.at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class Reader:
    """The gateway as a plain ASGI application (run by uvicorn as ``quantpulse.reader:app``)."""

    def __init__(
        self,
        key: str | None,
        upstream: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        rate: int = RATE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.key = key.encode() if key and len(key.strip()) >= MIN_KEY else None
        self.upstream = upstream.rstrip("/")
        self.transport = transport
        self.bucket = Bucket(rate, clock)
        self.client: httpx.AsyncClient | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":  # a WebSocket: refused before it is accepted
            await send({"type": "websocket.close", "code": 4403})
            return
        status, body, ctype = await self.handle(scope)
        headers = [(b"content-type", ctype), (b"cache-control", b"no-store")]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def handle(self, scope: Scope) -> tuple[int, bytes, bytes]:
        method, path = str(scope.get("method", "")), str(scope.get("path", ""))
        if self.key is None:
            return _error(503, "the reader is off: no read-only key is configured (./qp reader on)")
        if not self.bucket.take():
            return _error(429, f"more than {self.bucket.rate} requests a minute: wait a little")
        supplied = _header(scope, b"x-api-key")
        if supplied is None or not hmac.compare_digest(supplied, self.key):
            return _error(401, "missing or invalid X-API-Key")
        if method != "GET":
            return _error(405, "the reader only reads (GET)")
        if not readable(method, path):
            return _error(403, "not a monitoring page: the read-only key may only read those")
        url = httpx.URL(self.upstream + path, query=bytes(scope.get("query_string") or b""))
        headers = {"X-API-Key": self.key.decode(), "Accept": "application/json"}
        try:
            async with self._client().stream("GET", url, headers=headers) as r:
                body = bytearray()
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        return _error(502, "the page is too large to forward")
                ctype = r.headers.get("content-type", "application/json").encode("latin-1")
                return r.status_code, bytes(body), ctype
        except httpx.HTTPError as exc:
            return _error(502, f"the API did not answer ({type(exc).__name__})")

    def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(transport=self.transport, timeout=TIMEOUT, follow_redirects=False)
        return self.client

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                if self.client is not None:
                    await self.client.aclose()
                    self.client = None
                await send({"type": "lifespan.shutdown.complete"})
                return


app = Reader(os.environ.get("QP_API_READ_TOKEN"), os.environ.get("QP_READER_UPSTREAM", "http://api:8000"))
