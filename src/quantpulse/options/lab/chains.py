"""Where a backtest's option prices come from — and what that makes the result worth.

* :class:`RecordedChains` — chains QuantPulse recorded from the market (``options_quotes``) or a historical
  dataset imported by a person. Real quotes: evidence grade ``recorded``.
* :class:`ModelChains` — chains *priced by a model* from the underlying's own daily history: an implied
  volatility built from realized volatility (with a volatility premium, a term slope and a skew — all
  parameters stated), Black-Scholes prices, a bid/ask spread model, open interest falling away from the
  money. It tests a strategy's logic and the underlying's path; it cannot show whether real option prices
  offered the edge. Evidence grade ``model`` — never mistaken for market evidence.

Both answer only with data known on the day asked (no look-ahead): a model chain on day *t* uses closes up
to *t*; a recorded chain on day *t* is the last snapshot taken on *t*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

import numpy as np

from quantpulse.core.market_calendar import NEW_YORK, is_trading_day
from quantpulse.options.contracts import Kind, OptionContract, expiration_time
from quantpulse.options.data import ChainSnapshot
from quantpulse.options.pricing import greeks
from quantpulse.options.quotes import Greeks, OptionQuote

KINDS: tuple[Kind, Kind] = ("call", "put")


class ChainSource(Protocol):
    grade: str  # "recorded" | "model"

    def chain(
        self, underlying: str, day: date, dte: tuple[int, int] | None = None
    ) -> ChainSnapshot | None: ...

    def quote(self, symbol: str, underlying: str, day: date) -> OptionQuote | None: ...

    def spot(self, underlying: str, day: date) -> float | None: ...

    def atm_iv(self, underlying: str, day: date) -> float | None: ...


def close_time(day: date) -> datetime:
    """A day's end-of-day snapshot time: 15:55 New York (inside the session, before the close)."""
    from quantpulse.core.market_calendar import regular_close

    c = datetime.combine(day, regular_close(day), tzinfo=NEW_YORK)
    return (c - timedelta(minutes=5)).astimezone(UTC)


def standard_expirations(day: date, horizon_days: int = 200) -> list[date]:
    """Weekly Fridays for eight weeks, then the monthly third Fridays (a holiday moves it to Thursday)."""
    out: set[date] = set()
    d = day + timedelta(days=1)
    end = day + timedelta(days=horizon_days)
    while d <= end:
        if d.weekday() == 4:
            third = 15 <= d.day <= 21
            if (d - day).days <= 56 or third:
                e = d
                while not is_trading_day(e):
                    e -= timedelta(days=1)
                if e > day:
                    out.add(e)
        d += timedelta(days=1)
    return sorted(out)


def strike_step(spot: float) -> float:
    return 0.5 if spot < 25 else 1.0 if spot < 100 else 2.5 if spot < 250 else 5.0


@dataclass(frozen=True, slots=True)
class VolModel:
    """How a model chain's implied volatility is made (every number stated, none hidden)."""

    premium: float = 0.10  # implied runs this much above 20-day realized on average (a VRP proxy)
    floor: float = 0.08
    cap: float = 1.50
    term_slope: float = 0.02  # per year of maturity: a mild contango
    skew: float = 0.8  # IV rises this much per unit of log-moneyness below the spot (puts richer)
    smile: float = 0.3  # and curves up on both wings
    half_spread_pct: float = 0.03  # of the mid, per side
    min_half_spread: float = 0.025
    open_interest_atm: float = 5000


class ModelChains:
    grade = "model"

    def __init__(
        self,
        closes: Mapping[str, Mapping[date, float]],
        vol: VolModel | None = None,
        *,
        rate: float = 0.04,
        width: float = 0.30,
    ) -> None:
        self._closes = {u: dict(sorted(v.items())) for u, v in closes.items()}
        self._days = {u: list(v) for u, v in self._closes.items()}
        self._arr = {u: np.array(list(v.values()), dtype=float) for u, v in self._closes.items()}
        self._idx = {u: {d: i for i, d in enumerate(days)} for u, days in self._days.items()}
        self._vol = vol or VolModel()
        self._rate = rate
        self._width = width
        self._iv_cache: dict[tuple[str, date], float | None] = {}

    def days(self, underlying: str) -> list[date]:
        return self._days.get(underlying, [])

    def spot(self, underlying: str, day: date) -> float | None:
        return self._closes.get(underlying, {}).get(day)

    def atm_iv(self, underlying: str, day: date) -> float | None:
        key = (underlying, day)
        if key not in self._iv_cache:
            i = self._idx.get(underlying, {}).get(day)
            if i is None or i < 21:
                self._iv_cache[key] = None
            else:
                window = self._arr[underlying][i - 20 : i + 1]
                r = np.diff(np.log(window))
                rv = float(r.std(ddof=1) * math.sqrt(252))
                v = self._vol
                self._iv_cache[key] = min(v.cap, max(v.floor, rv * (1 + v.premium)))
        return self._iv_cache[key]

    def iv(self, underlying: str, day: date, strike: float, expiration: date) -> float | None:
        atm = self.atm_iv(underlying, day)
        spot = self.spot(underlying, day)
        if atm is None or spot is None:
            return None
        v = self._vol
        years = max((expiration - day).days, 0) / 365
        x = math.log(strike / spot)
        smile = atm * (1 + v.term_slope * years) - v.skew * atm * min(x, 0) * 1.0 + v.smile * atm * x * x
        return min(v.cap, max(v.floor, smile))

    def _quote(self, c: OptionContract, day: date, spot: float) -> OptionQuote | None:
        at = close_time(day)
        vol = self.iv(c.underlying, day, c.strike, c.expiration)
        if vol is None:
            return None
        years = max((expiration_time(c.expiration) - at).total_seconds(), 0) / (365 * 86400)
        g = greeks(c.kind, spot, c.strike, years, vol, self._rate)
        v = self._vol
        half = max(v.min_half_spread, v.half_spread_pct * g.price)
        bid = round(max(g.price - half, 0.0), 2)
        ask = round(g.price + half, 2)
        if ask <= 0.01:
            return None
        moneyness = abs(math.log(c.strike / spot))
        oi = v.open_interest_atm * math.exp(-8 * moneyness)
        return OptionQuote(c, bid if bid > 0 else None, ask, at, "model", "model", last=round(g.price, 2),
                           volume=round(oi / 10), open_interest=round(oi), iv=vol,
                           greeks=Greeks(g.delta, g.gamma, g.theta, g.vega, g.rho, "model"),
                           underlying_price=spot, underlying_at=at)  # fmt: skip

    def chain(self, underlying: str, day: date, dte: tuple[int, int] | None = None) -> ChainSnapshot | None:
        """The day's chain; ``dte`` limits it to expirations in that window (the rest are never priced)."""
        spot = self.spot(underlying, day)
        if spot is None or self.atm_iv(underlying, day) is None:
            return None
        step = strike_step(spot)
        lo = math.floor(spot * (1 - self._width) / step) * step
        hi = math.ceil(spot * (1 + self._width) / step) * step
        strikes = np.arange(lo, hi + step / 2, step)
        quotes: list[OptionQuote] = []
        for e in standard_expirations(day):
            if dte is not None and not dte[0] <= (e - day).days <= dte[1]:
                continue
            for k in strikes:
                if k <= 0:
                    continue
                for kind in KINDS:
                    q = self._quote(OptionContract(underlying, e, kind, round(float(k), 2)), day, spot)
                    if q is not None:
                        quotes.append(q)
        at = close_time(day)
        return ChainSnapshot(underlying, spot, at, at, "model", "model", quotes,
                             ["MODEL-PRICED: prices from a volatility model, not the market"])  # fmt: skip

    def quote(self, symbol: str, underlying: str, day: date) -> OptionQuote | None:
        from quantpulse.options.contracts import parse_occ

        spot = self.spot(underlying, day)
        if spot is None:
            return None
        c = parse_occ(symbol)
        if c.expiration < day:
            return None
        return self._quote(c, day, spot)


class RecordedChains:
    grade = "recorded"

    def __init__(
        self,
        snapshots: Mapping[tuple[str, date], ChainSnapshot],
        closes: Mapping[str, Mapping[date, float]] | None = None,
    ) -> None:
        self._snaps = dict(snapshots)
        self._closes = {u: dict(v) for u, v in (closes or {}).items()}
        self._index: dict[tuple[str, date], dict[str, OptionQuote]] = {
            k: {q.symbol: q for q in snap.quotes} for k, snap in self._snaps.items()
        }

    def days(self, underlying: str) -> list[date]:
        return sorted(d for u, d in self._snaps if u == underlying)

    def chain(self, underlying: str, day: date, dte: tuple[int, int] | None = None) -> ChainSnapshot | None:
        return self._snaps.get((underlying, day))

    def quote(self, symbol: str, underlying: str, day: date) -> OptionQuote | None:
        return self._index.get((underlying, day), {}).get(symbol)

    def spot(self, underlying: str, day: date) -> float | None:
        snap = self._snaps.get((underlying, day))
        if snap is not None:
            return snap.underlying_price
        return self._closes.get(underlying, {}).get(day)

    def atm_iv(self, underlying: str, day: date) -> float | None:
        snap = self._snaps.get((underlying, day))
        if snap is None:
            return None
        from quantpulse.options.analytics import by_expiry, constant_maturity_iv

        return constant_maturity_iv(by_expiry(snap.quotes, snap.underlying_price, close_time(day)), 30)


def closes_by_day(dates: Sequence[date], closes: Sequence[float]) -> dict[date, float]:
    return {d: float(c) for d, c in zip(dates, closes, strict=True) if c and math.isfinite(c)}
