"""Async token-bucket rate limiter with server-directed back-off (HTTP 429 ``Retry-After``)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from quantpulse.core.clock import Clock, SystemClock
from quantpulse.core.errors import ProviderRateLimited


class TokenBucket:
    """Classic token bucket.

    ``capacity`` tokens are available in a burst and the bucket refills at ``rate`` tokens per second.
    :meth:`acquire` waits for a token but never longer than ``max_wait`` seconds: if the projected wait
    exceeds it, :class:`ProviderRateLimited` is raised immediately so the gateway can fail over to the next
    source instead of stalling the request.
    """

    def __init__(
        self,
        name: str,
        rate: float,
        capacity: float,
        max_wait: float = 2.0,
        clock: Clock | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.name = name
        self.rate = rate
        self.capacity = capacity
        self.max_wait = max_wait
        self._clock = clock or SystemClock()
        self._sleep = sleep or asyncio.sleep
        self._tokens = capacity
        self._updated = self._clock.monotonic()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock.monotonic()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = now

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    @property
    def blocked_for(self) -> float:
        return max(0.0, self._blocked_until - self._clock.monotonic())

    def penalize(self, retry_after: float) -> None:
        """Block the bucket until ``retry_after`` seconds from now (server told us to back off)."""
        self._blocked_until = max(self._blocked_until, self._clock.monotonic() + max(0.0, retry_after))
        self._tokens = 0.0

    def try_acquire(self) -> bool:
        if self.blocked_for > 0:
            return False
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    async def acquire(self) -> None:
        async with self._lock:
            blocked = self.blocked_for
            if blocked > self.max_wait:
                raise ProviderRateLimited(self.name, retry_after=blocked)
            if blocked > 0:
                await self._sleep(blocked)
            self._refill()
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                if wait > self.max_wait:
                    raise ProviderRateLimited(self.name, retry_after=wait)
                await self._sleep(wait)
                self._refill()
                # Guard against clocks that did not advance during the sleep (e.g. fake clocks).
                self._tokens = max(self._tokens, 1.0)
            self._tokens -= 1.0

    def snapshot(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "rate_per_sec": self.rate,
            "capacity": self.capacity,
            "available": round(self.available, 3),
            "blocked_for_sec": round(self.blocked_for, 3),
        }
