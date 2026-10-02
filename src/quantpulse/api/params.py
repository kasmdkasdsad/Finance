"""Shared parameter parsing/validation helpers for routers."""

from __future__ import annotations

from datetime import date

from fastapi import Path
from pydantic import TypeAdapter, ValidationError

from quantpulse.core.errors import DomainError
from quantpulse.schemas.common import Symbol

_SYMBOL = TypeAdapter(Symbol)
MAX_SYMBOLS = 50


def normalise_symbol(raw: str) -> str:
    try:
        return _SYMBOL.validate_python(raw)
    except ValidationError as exc:
        raise DomainError(f"invalid symbol {raw!r}: {exc.errors()[0]['msg']}") from exc


def symbol_path(symbol: str = Path(min_length=1, max_length=15, description="Ticker, e.g. AAPL")) -> str:
    return normalise_symbol(symbol)


def parse_symbols(csv: str, limit: int = MAX_SYMBOLS) -> list[str]:
    symbols: list[str] = []
    for part in csv.split(","):
        if part.strip():
            s = normalise_symbol(part)
            if s not in symbols:
                symbols.append(s)
    if not symbols:
        raise DomainError("at least one symbol is required")
    if len(symbols) > limit:
        raise DomainError(f"at most {limit} symbols per request")
    return symbols


def parse_dates(csv: str | None) -> list[date] | None:
    if not csv:
        return None
    try:
        return sorted({date.fromisoformat(p.strip()) for p in csv.split(",") if p.strip()})
    except ValueError as exc:
        raise DomainError(f"expirations must be ISO dates (YYYY-MM-DD): {exc}") from exc
