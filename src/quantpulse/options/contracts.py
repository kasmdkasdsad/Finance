"""Option contracts: OCC symbols, expiration times, days to expiration.

An OCC symbol is ``ROOT`` + ``YYMMDD`` + ``C``/``P`` + the strike × 1000 in eight digits, e.g.
``AAPL261016C00210000`` (AAPL, 16 Oct 2026, call, $210). US equity options are American-style, settle
into 100 shares per contract, and stop trading at the close of their expiration day (13:00 New York on an
early-close day).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, regular_close
from quantpulse.options import MULTIPLIER

Kind = Literal["call", "put"]
Style = Literal["american", "european"]
OCC = re.compile(r"^(?P<root>[A-Z0-9.]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
SECONDS_PER_YEAR = 365.0 * 24 * 3600


class ContractError(ValueError):
    """A contract that cannot exist (bad symbol, strike, expiration)."""


@dataclass(frozen=True, slots=True)
class OptionContract:
    """One listed option. Immutable: its identity never changes."""

    underlying: str
    expiration: date
    kind: Kind
    strike: float
    multiplier: int = MULTIPLIER
    style: Style = "american"

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ContractError(f"kind must be call or put, not {self.kind!r}")
        if not (math.isfinite(self.strike) and self.strike > 0):
            raise ContractError(f"strike must be a positive number, not {self.strike!r}")
        if round(self.strike * 1000) >= 10**8:
            raise ContractError(f"strike {self.strike} does not fit an OCC symbol")
        if self.multiplier <= 0:
            raise ContractError("multiplier must be positive")
        if not self.underlying or not re.fullmatch(r"[A-Z0-9.]{1,6}", self.underlying):
            raise ContractError(f"bad underlying {self.underlying!r}")

    @property
    def symbol(self) -> str:
        return occ_symbol(self.underlying, self.expiration, self.kind, self.strike)

    @property
    def is_call(self) -> bool:
        return self.kind == "call"

    def intrinsic(self, spot: float) -> float:
        """Per share (multiply by :attr:`multiplier` for a contract)."""
        return max(spot - self.strike, 0.0) if self.is_call else max(self.strike - spot, 0.0)

    def moneyness(self, spot: float) -> float:
        """log(K / S): negative for in-the-money calls, positive for in-the-money puts."""
        return math.log(self.strike / spot)

    def expires_at(self) -> datetime:
        return expiration_time(self.expiration)

    def dte(self, now: datetime) -> int:
        return days_to_expiration(self.expiration, now)

    def years(self, now: datetime) -> float:
        return years_to_expiration(self.expiration, now)

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at()


def occ_symbol(underlying: str, expiration: date, kind: Kind, strike: float) -> str:
    root = underlying.upper().replace(".", "")[:6]
    return f"{root}{expiration:%y%m%d}{'C' if kind == 'call' else 'P'}{round(strike * 1000):08d}"


def parse_occ(symbol: str) -> OptionContract:
    """``AAPL261016C00210000`` (also ``O:AAPL…`` and space-padded roots) → :class:`OptionContract`."""
    raw = symbol.split(":", 1)[-1].replace(" ", "").upper()
    m = OCC.match(raw)
    if m is None:
        raise ContractError(f"not an OCC option symbol: {symbol!r}")
    ymd = m.group("ymd")
    try:
        expiration = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    except ValueError as exc:
        raise ContractError(f"bad expiration in {symbol!r}") from exc
    strike = int(m.group("strike")) / 1000.0
    return OptionContract(m.group("root"), expiration, "call" if m.group("cp") == "C" else "put", strike)


def is_option_symbol(symbol: str) -> bool:
    return OCC.match(symbol.split(":", 1)[-1].replace(" ", "").upper()) is not None


def expiration_time(expiration: date) -> datetime:
    """When trading in the contract stops: the close of its expiration day (early closes included)."""
    return datetime.combine(expiration, regular_close(expiration), tzinfo=NEW_YORK)


def days_to_expiration(expiration: date, now: datetime) -> int:
    """Calendar days from today (New York) to expiration; 0 on the expiration day, negative after it."""
    return (expiration - now.astimezone(NEW_YORK).date()).days


def years_to_expiration(expiration: date, now: datetime) -> float:
    """Exact time left to the last trading moment, in years (0 once it has passed)."""
    return max((expiration_time(expiration) - now).total_seconds(), 0.0) / SECONDS_PER_YEAR


def trading_days_to_expiration(expiration: date, now: datetime) -> int:
    """Trading sessions left, counting today's if it has not closed yet."""
    local = now.astimezone(NEW_YORK)
    n, d = 0, local.date()
    while d <= expiration:
        if is_trading_day(d) and (d > local.date() or local.time() < regular_close(d)):
            n += 1
        d += timedelta(days=1)
    return n
