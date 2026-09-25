"""Per-provider circuit breaker: stop calling a failing upstream until a cooldown elapses."""

from __future__ import annotations

from enum import StrEnum

from quantpulse.core.clock import Clock, SystemClock


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Closed -> (N consecutive failures) -> Open -> (cooldown) -> Half-open -> success closes / failure reopens."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        clock: Clock | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock or SystemClock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._cooldown = cooldown_seconds
        self.total_failures = 0
        self.total_successes = 0
        self.last_error: str | None = None

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock.monotonic() - self._opened_at >= self._cooldown:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    @property
    def retry_in(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self._cooldown - (self._clock.monotonic() - self._opened_at))

    def allow(self) -> bool:
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
        self._cooldown = self.cooldown_seconds
        self.total_successes += 1

    def record_failure(self, error: str | None = None, cooldown_hint: float | None = None) -> None:
        """Register a failure. ``cooldown_hint`` (e.g. a 429 Retry-After) extends the open window."""
        self.total_failures += 1
        self.last_error = error
        was_half_open = self.state is BreakerState.HALF_OPEN
        self._consecutive_failures += 1
        if was_half_open or self._consecutive_failures >= self.failure_threshold or cooldown_hint:
            self._opened_at = self._clock.monotonic()
            self._cooldown = max(self.cooldown_seconds, cooldown_hint or 0.0)

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "consecutive_failures": self._consecutive_failures,
            "retry_in_sec": round(self.retry_in, 1),
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_error": self.last_error,
        }
