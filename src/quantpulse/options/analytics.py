"""Implied-volatility analytics from chains and history — each number computed, with its sample shown.

* IV rank: where today's implied volatility sits between the lowest and highest of the look-back
  (0 = the low, 100 = the high); IV percentile: the share of days in the look-back with a lower IV. Both need
  enough history, and say how much they had (``None`` otherwise — never a guess).
* Realized volatility over 1–252 days (close to close, annualized); the IV/RV spread and ratio (a proxy for
  the volatility risk premium: implied minus what then happened is the premium itself, measurable only later).
* The term structure (IV by days to expiration; contango when later months are higher, backwardation when
  the front is), the skew (25-delta put IV minus 25-delta call IV), the smile's curvature.
* The implied move (the at-the-money straddle's price over the spot) and the model's expected move.
* Empirical probabilities: how often the underlying actually moved beyond a level over the same horizon in
  its own history — to set beside the model's (they differ, and the difference is information).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

import numpy as np

from quantpulse.options.contracts import OptionContract
from quantpulse.options.pricing import implied_vol
from quantpulse.options.quotes import OptionQuote

RV_WINDOWS = (1, 5, 10, 20, 30, 60, 90, 252)
TRADING_DAYS = 252
MIN_IV_HISTORY = 20  # days of IV history before a rank or percentile means anything


@dataclass(frozen=True, slots=True)
class IVStanding:
    iv: float
    rank: float | None  # 0–100
    percentile: float | None  # 0–100
    low: float | None
    high: float | None
    days: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in
                ((f, getattr(self, f)) for f in self.__dataclass_fields__)}  # fmt: skip


def iv_rank(current: float, history: Sequence[float], min_days: int = MIN_IV_HISTORY) -> IVStanding:
    """Rank and percentile of ``current`` within ``history`` (the look-back, today excluded)."""
    h = np.asarray([x for x in history if x is not None and math.isfinite(x) and x > 0], dtype=float)
    if len(h) < min_days:
        return IVStanding(current, None, None, None, None, len(h))
    lo, hi = float(h.min()), float(h.max())
    rank = 50.0 if hi - lo < 1e-12 else float(np.clip((current - lo) / (hi - lo) * 100, 0, 100))
    pct = float((h < current).mean() * 100)
    return IVStanding(current, rank, pct, lo, hi, len(h))


def realized_vol(closes: Sequence[float], window: int) -> float | None:
    """Annualized close-to-close volatility over the last ``window`` returns. For one day it is the size of
    the last move, annualized (a single observation: noisy by nature)."""
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) < window + 1:
        return None
    r = np.diff(np.log(c[-(window + 1) :]))
    if window == 1:
        return float(abs(r[0]) * math.sqrt(TRADING_DAYS))
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def realized_vols(closes: Sequence[float], windows: Sequence[int] = RV_WINDOWS) -> dict[int, float | None]:
    return {w: realized_vol(closes, w) for w in windows}


def iv_rv(iv: float, rv: float | None) -> dict[str, float | None]:
    if rv is None or rv <= 0:
        return {"spread": None, "ratio": None}
    return {"spread": iv - rv, "ratio": iv / rv}


def quote_iv(q: OptionQuote, now: datetime, rate: float = 0.04, div: float = 0.0) -> float | None:
    """The quote's implied volatility: the vendor's when given, else solved from the mid (``None`` if the
    mid has no volatility that reproduces it)."""
    if q.iv is not None and 0 < q.iv < 5:
        return q.iv
    if q.mid is None or not q.underlying_price:
        return None
    c = q.contract
    return implied_vol(c.kind, q.mid, q.underlying_price, c.strike, c.years(now), rate, div)


def _delta_of(q: OptionQuote, now: datetime, iv: float | None) -> float | None:
    if q.greeks.delta is not None:
        return q.greeks.delta
    if iv is None or not q.underlying_price:
        return None
    from quantpulse.options.pricing import greeks

    c = q.contract
    return greeks(c.kind, q.underlying_price, c.strike, c.years(now), iv).delta


@dataclass(frozen=True, slots=True)
class Expiry:
    expiration: str
    dte: int
    atm_iv: float | None
    call25_iv: float | None
    put25_iv: float | None
    skew: float | None  # put25 − call25 (positive: downside protection costs more)
    straddle: float | None  # ATM straddle mid, per share
    implied_move_pct: float | None
    curvature: float | None  # wings' average IV minus the ATM IV


def by_expiry(quotes: Sequence[OptionQuote], spot: float, now: datetime) -> list[Expiry]:
    """Per expiration: ATM IV (the strike nearest the spot, calls and puts averaged), 25-delta IVs, skew,
    the straddle and the implied move."""
    groups: dict[str, list[OptionQuote]] = {}
    for q in quotes:
        groups.setdefault(q.contract.expiration.isoformat(), []).append(q)
    out: list[Expiry] = []
    for exp, qs in sorted(groups.items()):
        c0 = qs[0].contract
        strikes = sorted({q.contract.strike for q in qs})
        if not strikes:
            continue
        atm_k = min(strikes, key=lambda k: abs(k - spot))
        ivs = {(q.contract.kind, q.contract.strike): quote_iv(q, now) for q in qs}
        atm = [v for k, v in ivs.items() if k[1] == atm_k and v is not None]
        atm_iv = float(np.mean(atm)) if atm else None
        by_delta: dict[str, list[tuple[float, float]]] = {"call": [], "put": []}
        for q in qs:
            iv = ivs[(q.contract.kind, q.contract.strike)]
            d = _delta_of(q, now, iv)
            if iv is not None and d is not None:
                by_delta[q.contract.kind].append((abs(d), iv))
        c25 = _nearest(by_delta["call"], 0.25)
        p25 = _nearest(by_delta["put"], 0.25)
        call = next((q for q in qs if q.contract.kind == "call" and q.contract.strike == atm_k), None)
        put = next((q for q in qs if q.contract.kind == "put" and q.contract.strike == atm_k), None)
        straddle = call.mid + put.mid if call and put and call.mid and put.mid else None
        wings = [v for v in (c25, p25) if v is not None]
        out.append(
            Expiry(
                expiration=exp,
                dte=c0.dte(now),
                atm_iv=atm_iv,
                call25_iv=c25,
                put25_iv=p25,
                skew=(p25 - c25) if p25 is not None and c25 is not None else None,
                straddle=straddle,
                implied_move_pct=straddle / spot if straddle else None,
                curvature=(float(np.mean(wings)) - atm_iv) if wings and atm_iv is not None else None,
            )
        )
    return out


def _nearest(points: list[tuple[float, float]], target: float, tolerance: float = 0.12) -> float | None:
    if not points:
        return None
    d, iv = min(points, key=lambda p: abs(p[0] - target))
    return iv if abs(d - target) <= tolerance else None


def term_structure(expiries: Sequence[Expiry]) -> dict[str, float | str | None]:
    """The slope of ATM IV against time (per 30 days) and its shape."""
    pts = [(e.dte, e.atm_iv) for e in expiries if e.atm_iv is not None and e.dte > 0]
    if len(pts) < 2:
        return {
            "slope_per_30d": None,
            "shape": "unknown",
            "front_iv": pts[0][1] if pts else None,
            "back_iv": None,
        }
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0] * 30)
    front, back = float(y[0]), float(y[-1])
    shape = "backwardation" if front > back * 1.02 else "contango" if back > front * 1.02 else "flat"
    return {"slope_per_30d": slope, "shape": shape, "front_iv": front, "back_iv": back}


def constant_maturity_iv(expiries: Sequence[Expiry], days: int = 30) -> float | None:
    """ATM IV interpolated (in total variance) to a fixed number of days: comparable from day to day."""
    pts = sorted((e.dte, e.atm_iv) for e in expiries if e.atm_iv is not None and e.dte > 0)
    if not pts:
        return None
    if days <= pts[0][0]:
        return pts[0][1]
    if days >= pts[-1][0]:
        return pts[-1][1]
    for (d0, v0), (d1, v1) in pairwise(pts):
        if d0 <= days <= d1:
            w0, w1 = v0 * v0 * d0, v1 * v1 * d1
            w = w0 + (w1 - w0) * (days - d0) / (d1 - d0)
            return math.sqrt(max(w, 0.0) / days)
    return None


def empirical_move_probability(
    closes: Sequence[float], horizon: int, move_pct: float
) -> dict[str, float | int | None]:
    """How often, in this history, the price ended ``horizon`` trading days later more than ``move_pct``
    above (or below, for a negative ``move_pct``) where it started — overlapping windows, so the sample is
    smaller than it looks (``effective_n`` is the non-overlapping count)."""
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) <= horizon + 20:
        return {"probability": None, "n": 0, "effective_n": 0}
    r = c[horizon:] / c[:-horizon] - 1
    hit = r > move_pct if move_pct >= 0 else r < move_pct
    return {"probability": float(hit.mean()), "n": len(r), "effective_n": int(len(r) // horizon)}


def empirical_touch_probability(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
                                horizon: int, move_pct: float) -> dict[str, float | int | None]:  # fmt: skip
    """How often the price *touched* a level ``move_pct`` away within ``horizon`` days (intraday highs/lows)."""
    h, lo, c = (np.asarray(a, dtype=float) for a in (highs, lows, closes))
    n = min(len(h), len(lo), len(c))
    if n <= horizon + 20:
        return {"probability": None, "n": 0}
    hits = 0
    total = 0
    for i in range(n - horizon):
        start = c[i]
        if not start > 0:
            continue
        window = h[i + 1 : i + 1 + horizon] if move_pct >= 0 else lo[i + 1 : i + 1 + horizon]
        level = start * (1 + move_pct)
        hits += bool((window >= level).any() if move_pct >= 0 else (window <= level).any())
        total += 1
    return {"probability": hits / total if total else None, "n": total}


def vol_of_vol(iv_history: Sequence[float], window: int = 20) -> float | None:
    """The volatility of implied volatility (annualized standard deviation of daily log IV changes)."""
    x = np.asarray([v for v in iv_history if v and v > 0], dtype=float)
    if len(x) < window + 1:
        return None
    d = np.diff(np.log(x[-(window + 1) :]))
    return float(d.std(ddof=1) * math.sqrt(TRADING_DAYS))


def contract_features(c: OptionContract, spot: float, now: datetime) -> dict[str, float]:
    return {"dte": float(c.dte(now)), "moneyness": c.moneyness(spot), "years": c.years(now)}
