"""Injectable clocks so time-dependent logic (TTL, rate limits, breakers) is deterministic in tests."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def now(self) -> datetime: ...


class SystemClock:
    """Real wall-clock / monotonic time."""

    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    """Manually advanced clock for tests."""

    def __init__(self, start: datetime | None = None) -> None:
        self._mono = 1_000.0
        self._now = start or datetime(2026, 1, 5, 15, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self._mono

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        from datetime import timedelta

        self._mono += seconds
        self._now = self._now + timedelta(seconds=seconds)


def utcnow() -> datetime:
    return datetime.now(UTC)
