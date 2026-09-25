"""Shared async HTTP client with rate limiting, bounded retries and typed provider errors."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import httpx

from quantpulse.core.errors import (
    ProviderError,
    ProviderHTTPError,
    ProviderParseError,
    ProviderRateLimited,
)
from quantpulse.core.rate_limit import TokenBucket

logger = logging.getLogger(__name__)

TRANSIENT_STATUS = frozenset({500, 502, 503, 504})
DEFAULT_USER_AGENT = "QuantPulseTerminal/1.0 (+https://github.com/kasmdkasdsad/Finance)"


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


class HttpClient:
    """Thin wrapper around :class:`httpx.AsyncClient`.

    * acquires a token from the provider's :class:`TokenBucket` before every request;
    * converts HTTP 429 into :class:`ProviderRateLimited` and blocks the bucket for ``Retry-After``;
    * retries transient failures (connect/read errors, 5xx) with jittered exponential back-off;
    * converts everything else into :class:`ProviderHTTPError` / :class:`ProviderParseError`.
    """

    def __init__(
        self,
        timeout: float = 12.0,
        max_retries: int = 1,
        limiters: Mapping[str, TokenBucket] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff_base: float = 0.25,
        concurrency: Mapping[str, int] | None = None,
    ) -> None:
        self._limiters: dict[str, TokenBucket] = dict(limiters or {})
        # Bounding in-flight requests per provider keeps token-bucket queues short, so a burst (e.g. a
        # 30-symbol screen) is paced instead of tripping the limiter's fail-fast ``max_wait``.
        self._semaphores: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(max(1, n)) for name, n in (concurrency or {}).items()
        }
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json, text/plain, */*"},
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            transport=transport,
        )

    @property
    def raw(self) -> httpx.AsyncClient:
        return self._client

    def limiter(self, provider: str) -> TokenBucket | None:
        return self._limiters.get(provider)

    def register_limiter(self, bucket: TokenBucket) -> None:
        self._limiters[bucket.name] = bucket

    @property
    def limiters(self) -> dict[str, TokenBucket]:
        return dict(self._limiters)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        provider: str,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
        timeout: float | None = None,
    ) -> httpx.Response:
        semaphore = self._semaphores.get(provider)
        if semaphore is None:
            return await self._request(
                provider, method, url, params=params, headers=headers, expected=expected, timeout=timeout
            )
        async with semaphore:
            return await self._request(
                provider, method, url, params=params, headers=headers, expected=expected, timeout=timeout
            )

    async def _request(
        self,
        provider: str,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
        timeout: float | None = None,
    ) -> httpx.Response:
        limiter = self._limiters.get(provider)
        attempt = 0
        while True:
            if limiter is not None:
                await limiter.acquire()
            try:
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=httpx.USE_CLIENT_DEFAULT if timeout is None else httpx.Timeout(timeout),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < self._max_retries:
                    attempt += 1
                    await self._backoff(attempt)
                    continue
                raise ProviderError(provider, f"network error: {type(exc).__name__}: {exc}") from exc

            if response.status_code == 429:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                if limiter is not None:
                    limiter.penalize(retry_after if retry_after is not None else 30.0)
                raise ProviderRateLimited(provider, retry_after=retry_after)
            if response.status_code in TRANSIENT_STATUS and attempt < self._max_retries:
                attempt += 1
                await self._backoff(attempt)
                continue
            if response.status_code not in expected:
                raise ProviderHTTPError(provider, response.status_code, _snippet(response))
            return response

    async def get_json(
        self,
        provider: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        response = await self.request(provider, "GET", url, params=params, headers=headers, timeout=timeout)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderParseError(provider, f"invalid JSON: {_snippet(response)}") from exc

    async def get_text(
        self,
        provider: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> str:
        response = await self.request(provider, "GET", url, params=params, headers=headers, timeout=timeout)
        return response.text

    async def _backoff(self, attempt: int) -> None:
        delay = self._backoff_base * (2 ** (attempt - 1))
        await asyncio.sleep(delay + random.uniform(0, delay / 2))


def _snippet(response: httpx.Response, limit: int = 200) -> str:
    try:
        text = response.text
    except Exception:  # pragma: no cover - undecodable body
        return "<unreadable body>"
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")
