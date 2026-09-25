"""In-memory TTL cache with LRU eviction, stale-while-error retention and single-flight loading."""

from __future__ import annotations

import asyncio
import functools
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from quantpulse.core.clock import Clock, SystemClock

T = TypeVar("T")


@dataclass(slots=True)
class CacheEntry(Generic[T]):
    value: T
    stored_at: float
    expires_at: float
    stale_until: float
    meta: dict[str, Any] = field(default_factory=dict)

    def is_fresh(self, now: float) -> bool:
        return now < self.expires_at

    def is_servable_stale(self, now: float) -> bool:
        return now < self.stale_until


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    stale_hits: int = 0
    sets: int = 0
    evictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_hits": self.stale_hits,
            "sets": self.sets,
            "evictions": self.evictions,
        }


class TTLCache:
    """A bounded LRU cache where every entry has a fresh TTL and a longer stale grace window.

    * Fresh entries are served directly.
    * Expired-but-within-grace entries are *not* returned by :meth:`get` but are available through
      :meth:`get_stale` so callers can degrade gracefully when every live source fails.
    * Least-recently-used entries are evicted once ``max_entries`` is exceeded.

    The cache is designed for a single asyncio event loop; mutations never ``await``, so no lock is
    needed for consistency.
    """

    def __init__(self, max_entries: int = 4096, clock: Clock | None = None) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._clock = clock or SystemClock()
        self._data: OrderedDict[str, CacheEntry[Any]] = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        entry = self._data.get(key)
        return entry is not None and entry.is_fresh(self._clock.monotonic())

    def get(self, key: str) -> CacheEntry[Any] | None:
        """Return a *fresh* entry or ``None``."""
        entry = self._data.get(key)
        now = self._clock.monotonic()
        if entry is None or not entry.is_fresh(now):
            self.stats.misses += 1
            if entry is not None and not entry.is_servable_stale(now):
                del self._data[key]
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return entry

    def get_stale(self, key: str) -> CacheEntry[Any] | None:
        """Return an entry that is expired but still inside its stale grace window."""
        entry = self._data.get(key)
        if entry is None:
            return None
        now = self._clock.monotonic()
        if entry.is_servable_stale(now):
            self.stats.stale_hits += 1
            return entry
        del self._data[key]
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: float,
        stale_ttl: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> CacheEntry[Any]:
        if ttl < 0 or stale_ttl < 0:
            raise ValueError("ttl and stale_ttl must be non-negative")
        now = self._clock.monotonic()
        entry: CacheEntry[Any] = CacheEntry(
            value=value,
            stored_at=now,
            expires_at=now + ttl,
            stale_until=now + ttl + stale_ttl,
            meta=dict(meta or {}),
        )
        self._data[key] = entry
        self._data.move_to_end(key)
        self.stats.sets += 1
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)
            self.stats.evictions += 1
        return entry

    def invalidate(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def invalidate_prefix(self, prefix: str) -> int:
        keys = [k for k in self._data if k.startswith(prefix)]
        for k in keys:
            del self._data[k]
        return len(keys)

    def clear(self) -> None:
        self._data.clear()

    def purge_expired(self) -> int:
        """Drop entries past their stale window. Returns the number removed."""
        now = self._clock.monotonic()
        dead = [k for k, e in self._data.items() if not e.is_servable_stale(now)]
        for k in dead:
            del self._data[k]
        return len(dead)

    def snapshot(self) -> dict[str, Any]:
        return {"entries": len(self._data), "max_entries": self._max_entries, **self.stats.as_dict()}


class SingleFlight:
    """Coalesce concurrent calls for the same key into one in-flight task.

    Prevents a thundering herd of identical upstream requests (and rate-limit exhaustion) when many
    clients ask for the same symbol at the same moment. The shared work runs in its own task, so a
    caller that disconnects (is cancelled) does not cancel the work other callers are waiting on.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(fn())
            self._inflight[key] = task
            task.add_done_callback(functools.partial(self._on_done, key))
        return await asyncio.shield(task)

    def _on_done(self, key: str, task: asyncio.Task[Any]) -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()  # mark retrieved; callers re-raise it themselves
