"""Shared helpers for live data providers: strict wire-schema parsing and time conversion."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from quantpulse.core.errors import ProviderNoData, ProviderParseError

M = TypeVar("M", bound=BaseModel)


class WireModel(BaseModel):
    """Base for vendor payload schemas.

    Vendors add fields without notice, so unknown keys are ignored; every field we *consume* is typed
    and validated, so a breaking upstream change surfaces as a :class:`ProviderParseError` (and a clean
    fallback) instead of corrupt numbers flowing into the analytics.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


def parse_wire(provider: str, model: type[M], payload: Any) -> M:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise ProviderParseError(
            provider, f"payload failed {model.__name__} validation at '{where}': {first.get('msg')}"
        ) from exc


def epoch_to_datetime(value: float | int, unit: str = "s") -> datetime:
    divisor = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]
    return datetime.fromtimestamp(float(value) / divisor, tz=UTC)


def epoch_to_date(value: float | int) -> date:
    return epoch_to_datetime(value).date()


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def positive_or_none(value: float | None) -> float | None:
    f = finite_or_none(value)
    return f if f is not None and f > 0 else None


def require(condition: bool, provider: str, message: str) -> None:
    if not condition:
        raise ProviderNoData(provider, message)


def occ_parse(symbol: str) -> tuple[str, date, str, float]:
    """Parse an OCC option symbol, e.g. ``AAPL261016C00210000`` or ``O:AAPL261016C00210000``.

    Returns ``(root, expiration, kind, strike)``.
    """
    raw = symbol.split(":", 1)[-1].strip()
    if len(raw) < 16:
        raise ValueError(f"not an OCC symbol: {symbol!r}")
    tail = raw[-15:]
    root = raw[:-15].strip()
    yymmdd, cp, strike_digits = tail[:6], tail[6], tail[7:]
    if cp not in ("C", "P") or not (yymmdd.isdigit() and strike_digits.isdigit()) or not root:
        raise ValueError(f"not an OCC symbol: {symbol!r}")
    expiration = date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    return root, expiration, "call" if cp == "C" else "put", int(strike_digits) / 1000.0
