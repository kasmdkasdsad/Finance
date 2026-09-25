"""The data gateway: one place that decides where every piece of data comes from.

Resolution order for a request::

    fresh in-memory cache  ->  live providers (in priority order, guarded by circuit breakers)
        ->  stale in-memory cache  ->  warehouse (last-known-good rows in SQLite)  ->  synthetic

Every result carries a :class:`~quantpulse.schemas.common.Provenance` so the UI can display an honest
LIVE / CACHED / STALE / SYNTHETIC badge together with the reasons each live source was skipped.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from quantpulse.core.cache import SingleFlight, TTLCache
from quantpulse.core.circuit_breaker import CircuitBreaker
from quantpulse.core.clock import Clock, SystemClock
from quantpulse.core.errors import (
    ProviderError,
    ProviderNotConfigured,
    ProviderRateLimited,
)
from quantpulse.schemas.common import DataStatus, Provenance, ProviderAttempt

logger = logging.getLogger(__name__)

T = TypeVar("T")
FALLBACK_MEMO_SECONDS = 30.0


@dataclass(slots=True)
class Source(Generic[T]):
    """A candidate live source for one request."""

    name: str
    fetch: Callable[[], Awaitable[T]]
    configured: bool = True
    not_configured_reason: str = "credentials not configured"


@dataclass(slots=True)
class Resolved(Generic[T]):
    value: T
    provenance: Provenance

    @property
    def status(self) -> DataStatus:
        return self.provenance.status


@dataclass(slots=True)
class ProviderHealth:
    name: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    avg_latency_ms: float | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error: str | None = None
    _latency_samples: int = field(default=0, repr=False)

    def record(self, ok: bool, latency_ms: float, at: datetime, error: str | None = None) -> None:
        self.calls += 1
        if ok:
            self.successes += 1
            self.last_success_at = at
            self._latency_samples += 1
            if self.avg_latency_ms is None:
                self.avg_latency_ms = latency_ms
            else:  # exponential moving average
                self.avg_latency_ms = 0.8 * self.avg_latency_ms + 0.2 * latency_ms
        else:
            self.failures += 1
            self.last_failure_at = at
            self.last_error = error

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "avg_latency_ms": None if self.avg_latency_ms is None else round(self.avg_latency_ms, 1),
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "last_error": self.last_error,
        }


ArchiveLoader = Callable[[], Awaitable[tuple[T, datetime, str] | None]]
LiveHook = Callable[[T, str], Awaitable[None]]
TTL = float | Callable[[T], float]


class DataGateway:
    def __init__(
        self,
        cache: TTLCache,
        *,
        live_enabled: bool = True,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        stale_grace_seconds: float = 86400.0,
        clock: Clock | None = None,
    ) -> None:
        self.cache = cache
        self.live_enabled = live_enabled
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._stale_grace = stale_grace_seconds
        self._clock = clock or SystemClock()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._health: dict[str, ProviderHealth] = {}
        self._flights = SingleFlight()

    # ------------------------------------------------------------------ health / introspection
    def breaker(self, provider: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(provider, self._failure_threshold, self._cooldown, self._clock)
            self._breakers[provider] = breaker
        return breaker

    def health(self, provider: str) -> ProviderHealth:
        health = self._health.get(provider)
        if health is None:
            health = ProviderHealth(provider)
            self._health[provider] = health
        return health

    def snapshot(self) -> dict[str, Any]:
        names = sorted(set(self._breakers) | set(self._health))
        return {
            "live_enabled": self.live_enabled,
            "inflight": self._flights.inflight,
            "providers": {
                name: {**self.health(name).snapshot(), "breaker": self.breaker(name).snapshot()}
                for name in names
            },
        }

    # ------------------------------------------------------------------ resolution
    async def resolve(
        self,
        key: str,
        sources: Sequence[Source[T]],
        synthetic: Callable[[], T],
        ttl: TTL[T],
        *,
        as_of: Callable[[T], datetime | None] | None = None,
        archive: ArchiveLoader[T] | None = None,
        on_live: LiveHook[T] | None = None,
        force_refresh: bool = False,
        fallback_provider: str = "synthetic",
        fallback_status: DataStatus = DataStatus.SYNTHETIC,
    ) -> Resolved[T]:
        """Resolve ``key`` through cache → live sources → stale cache → archive → fallback.

        ``ttl`` may be a number or a function of the fetched value (e.g. shorter while games are live).
        ``fallback_provider`` / ``fallback_status`` label the last-resort value; the default is SYNTHETIC,
        while packaged reference data (e.g. verified EPA ratings) can be labelled STALE instead.
        """
        if not force_refresh:
            cached = self._from_cache(key)
            if cached is not None:
                return cached
        flight_key = f"{key}|force" if force_refresh else key
        return await self._flights.run(
            flight_key,
            lambda: self._resolve(
                key,
                sources,
                synthetic,
                ttl,
                as_of,
                archive,
                on_live,
                force_refresh,
                fallback_provider,
                fallback_status,
            ),
        )

    def _from_cache(self, key: str) -> Resolved[Any] | None:
        entry = self.cache.get(key)
        if entry is None:
            return None
        prov: Provenance = entry.meta["provenance"]
        age = self._clock.monotonic() - entry.stored_at
        return Resolved(
            entry.value,
            prov.model_copy(
                update={"status": DataStatus.CACHED, "message": f"served from cache (age {age:.0f}s)"}
            ),
        )

    async def _resolve(
        self,
        key: str,
        sources: Sequence[Source[T]],
        synthetic: Callable[[], T],
        ttl: TTL[T],
        as_of: Callable[[T], datetime | None] | None,
        archive: ArchiveLoader[T] | None,
        on_live: LiveHook[T] | None,
        force_refresh: bool,
        fallback_provider: str,
        fallback_status: DataStatus,
    ) -> Resolved[T]:
        if not force_refresh:
            cached = self._from_cache(key)  # another flight may have filled it while we queued
            if cached is not None:
                return cached

        attempts: list[ProviderAttempt] = []
        if not self.live_enabled:
            attempts.append(
                ProviderAttempt(
                    provider="*", ok=False, error="live data disabled (QP_ENABLE_LIVE_DATA=false)"
                )
            )
        else:
            for source in sources:
                result = await self._try_source(source, attempts)
                if result is None:
                    continue
                value, latency_ms = result
                now = self._clock.now()
                observed = (as_of(value) if as_of else None) or now
                prov = Provenance(
                    status=DataStatus.LIVE,
                    provider=source.name,
                    as_of=observed,
                    fetched_at=now,
                    latency_ms=round(latency_ms, 1),
                    attempts=attempts,
                )
                ttl_seconds = ttl(value) if callable(ttl) else ttl
                self.cache.set(key, value, ttl_seconds, self._stale_grace, meta={"provenance": prov})
                if on_live is not None:
                    try:
                        await on_live(value, source.name)
                    except Exception:  # persistence must never break a live response
                        logger.exception("on_live hook failed for %s", key)
                return Resolved(value, prov)

        reasons = "; ".join(f"{a.provider}: {a.error}" for a in attempts if a.error)

        stale = self.cache.get_stale(key)
        if stale is not None:
            prov = stale.meta["provenance"]
            return Resolved(
                stale.value,
                prov.model_copy(
                    update={
                        "status": DataStatus.STALE,
                        "message": f"live refresh failed, serving last good value ({reasons})",
                        "attempts": attempts,
                    }
                ),
            )

        if archive is not None:
            try:
                archived = await archive()
            except Exception:
                logger.exception("archive loader failed for %s", key)
                archived = None
            if archived is not None:
                value, observed, provider = archived
                now = self._clock.now()
                return Resolved(
                    value,
                    Provenance(
                        status=DataStatus.STALE,
                        provider=f"warehouse:{provider}",
                        as_of=observed,
                        fetched_at=now,
                        message=f"live sources unavailable, serving warehouse snapshot ({reasons})",
                        attempts=attempts,
                    ),
                )

        # The fallback is memoised briefly under its own key: it never shadows the live key (so a recovered
        # provider is used on the very next request) but avoids regenerating identical synthetic data.
        fallback_key = f"{key}#fallback"
        memo = self.cache.get(fallback_key)
        if memo is not None:
            prov = memo.meta["provenance"]
            return Resolved(memo.value, prov.model_copy(update={"attempts": attempts}))
        now = self._clock.now()
        value = synthetic()
        label = (
            "synthetic fallback"
            if fallback_status is DataStatus.SYNTHETIC
            else f"{fallback_provider} fallback"
        )
        prov = Provenance(
            status=fallback_status,
            provider=fallback_provider,
            as_of=now,
            fetched_at=now,
            message=f"{label} — {reasons}" if reasons else label,
            attempts=attempts,
        )
        ttl_seconds = ttl(value) if callable(ttl) else ttl
        self.cache.set(
            fallback_key, value, min(ttl_seconds, FALLBACK_MEMO_SECONDS), meta={"provenance": prov}
        )
        return Resolved(value, prov)

    async def _try_source(self, source: Source[T], attempts: list[ProviderAttempt]) -> tuple[T, float] | None:
        if not source.configured:
            attempts.append(
                ProviderAttempt(provider=source.name, ok=False, error=source.not_configured_reason)
            )
            return None
        breaker = self.breaker(source.name)
        if not breaker.allow():
            attempts.append(
                ProviderAttempt(
                    provider=source.name, ok=False, error=f"circuit open (retry in {breaker.retry_in:.0f}s)"
                )
            )
            return None
        health = self.health(source.name)
        started = time.perf_counter()
        try:
            value = await source.fetch()
        except ProviderNotConfigured as exc:
            attempts.append(ProviderAttempt(provider=source.name, ok=False, error=exc.message))
            return None
        except ProviderRateLimited as exc:
            latency = (time.perf_counter() - started) * 1000
            breaker.record_failure(exc.message, cooldown_hint=exc.retry_after or self._cooldown)
            health.record(False, latency, self._clock.now(), exc.message)
            attempts.append(
                ProviderAttempt(
                    provider=source.name, ok=False, error=exc.message, latency_ms=round(latency, 1)
                )
            )
            return None
        except ProviderError as exc:
            latency = (time.perf_counter() - started) * 1000
            breaker.record_failure(exc.message)
            health.record(False, latency, self._clock.now(), exc.message)
            attempts.append(
                ProviderAttempt(
                    provider=source.name, ok=False, error=exc.message, latency_ms=round(latency, 1)
                )
            )
            return None
        except Exception as exc:  # defensive: an unexpected payload shape must not 500 the API
            latency = (time.perf_counter() - started) * 1000
            message = f"unexpected {type(exc).__name__}: {exc}"
            logger.exception("provider %s raised unexpectedly", source.name)
            breaker.record_failure(message)
            health.record(False, latency, self._clock.now(), message)
            attempts.append(
                ProviderAttempt(provider=source.name, ok=False, error=message, latency_ms=round(latency, 1))
            )
            return None
        latency = (time.perf_counter() - started) * 1000
        breaker.record_success()
        health.record(True, latency, self._clock.now())
        attempts.append(ProviderAttempt(provider=source.name, ok=True, latency_ms=round(latency, 1)))
        return value, latency
