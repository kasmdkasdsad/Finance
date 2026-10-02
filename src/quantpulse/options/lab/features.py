"""Point-in-time features of an underlying and its options, for signals, regimes and learning.

Every feature on day *t* is computed from data up to and including *t* only. The same functions serve the
backtester and the live Brain, so a signal means the same thing in both.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from quantpulse.options.analytics import iv_rank


@dataclass
class DayFeatures:
    day: date
    spot: float
    sma50: float | None = None
    sma200: float | None = None
    ret5: float | None = None
    ret20: float | None = None
    ret20_pct: float | None = None  # percentile of today's 20-day return in the past year
    z5: float | None = None  # 5-day return in units of its own standard deviation
    high20: float | None = None
    low20: float | None = None
    rv20: float | None = None
    rv60: float | None = None
    atr14_pct: float | None = None
    iv: float | None = None
    iv_rank: float | None = None
    iv_percentile: float | None = None
    iv_change5: float | None = None
    iv_rv: float | None = None
    term_shape: str | None = None
    skew: float | None = None
    event_days: int | None = None  # calendar days to the next known earnings date
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("extra", "day")}
        out["day"] = self.day.isoformat()
        out.update(self.extra)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def history(closes: Mapping[date, float], iv_by_day: Mapping[date, float | None] | None = None,
            events: Sequence[date] = ()) -> dict[date, DayFeatures]:  # fmt: skip
    """Features for every day of ``closes`` (sorted), each from the past only."""
    days = sorted(closes)
    c = np.array([closes[d] for d in days], dtype=float)
    logr = np.diff(np.log(c), prepend=np.nan)
    ivs = [None if iv_by_day is None else iv_by_day.get(d) for d in days]
    out: dict[date, DayFeatures] = {}
    ev = sorted(events)
    for i, d in enumerate(days):
        f = DayFeatures(d, float(c[i]))
        if i >= 49:
            f.sma50 = float(c[i - 49 : i + 1].mean())
        if i >= 199:
            f.sma200 = float(c[i - 199 : i + 1].mean())
        if i >= 5:
            f.ret5 = float(c[i] / c[i - 5] - 1)
        if i >= 20:
            f.ret20 = float(c[i] / c[i - 20] - 1)
            f.high20 = float(c[i - 20 : i].max())
            f.low20 = float(c[i - 20 : i].min())
            r = logr[i - 19 : i + 1]
            f.rv20 = float(np.std(r, ddof=1) * math.sqrt(252))
        if i >= 60:
            f.rv60 = float(np.std(logr[i - 59 : i + 1], ddof=1) * math.sqrt(252))
        if i >= 272:
            past = c[i - 252 : i + 1]
            r20 = past[20:] / past[:-20] - 1
            f.ret20_pct = float((r20[:-1] < r20[-1]).mean() * 100)
        if i >= 65 and f.ret5 is not None:
            r5 = c[i - 60 : i + 1][5:] / c[i - 60 : i + 1][:-5] - 1
            sd = float(np.std(r5[:-1], ddof=1))
            f.z5 = float(f.ret5 / sd) if sd > 0 else None
        if i >= 14:
            moves = np.abs(np.diff(c[i - 14 : i + 1]))
            f.atr14_pct = float(moves.mean() / c[i])
        iv = ivs[i]
        if iv is not None:
            f.iv = iv
            past_iv = [v for v in ivs[max(0, i - 252) : i] if v is not None]
            standing = iv_rank(iv, past_iv)
            f.iv_rank, f.iv_percentile = standing.rank, standing.percentile
            if i >= 5 and ivs[i - 5] is not None:
                f.iv_change5 = iv - ivs[i - 5]  # type: ignore[operator]
            if f.rv20:
                f.iv_rv = iv / f.rv20
        nxt = next((e for e in ev if e >= d), None)
        f.event_days = (nxt - d).days if nxt is not None else None
        out[d] = f
    return out


def trend_regime(f: DayFeatures) -> str:
    """TRENDING_UP / TRENDING_DOWN / MEAN_REVERTING / CALM / PANIC — measured, with thresholds stated."""
    if f.rv20 is not None and f.ret20 is not None and f.rv20 > 0.45 and f.ret20 < -0.08:
        return "PANIC"
    if f.sma50 and f.sma200:
        if f.spot > f.sma50 > f.sma200:
            return "TRENDING_UP"
        if f.spot < f.sma50 < f.sma200:
            return "TRENDING_DOWN"
    if f.rv20 is not None and f.rv20 < 0.12:
        return "CALM"
    return "MEAN_REVERTING"


def iv_regime(f: DayFeatures) -> str:
    if f.iv_rank is None:
        return "UNKNOWN_IV"
    if f.iv_rank >= 70:
        return "HIGH_IV"
    if f.iv_rank <= 30:
        return "LOW_IV"
    return "NORMAL_IV"


def iv_trend(f: DayFeatures) -> str:
    if f.iv_change5 is None:
        return "IV_UNKNOWN"
    return (
        "IV_EXPANSION" if f.iv_change5 > 0.03 else "IV_CONTRACTION" if f.iv_change5 < -0.03 else "IV_STABLE"
    )


def signal(name: str, f: DayFeatures, *, seed_value: float | None = None) -> bool | None:
    """Does ``name`` fire on day ``f``? ``None`` when the data to decide is missing (never a guess)."""
    if name == "always":
        return True
    if name == "random":
        return None if seed_value is None else seed_value < 0.1
    if name in ("trend_up", "trend_down"):
        if f.sma50 is None or f.sma200 is None:
            return None
        up = f.spot > f.sma50 > f.sma200
        down = f.spot < f.sma50 < f.sma200
        return up if name == "trend_up" else down
    if name in ("momentum_up", "momentum_down"):
        if f.ret20_pct is None:
            return None
        return f.ret20_pct >= 80 if name == "momentum_up" else f.ret20_pct <= 20
    if name in ("breakout_up", "breakout_down"):
        if f.high20 is None or f.low20 is None:
            return None
        return f.spot > f.high20 if name == "breakout_up" else f.spot < f.low20
    if name in ("reversion_up", "reversion_down"):
        if f.z5 is None:
            return None
        return f.z5 <= -2 if name == "reversion_up" else f.z5 >= 2
    if name in ("iv_high", "iv_low"):
        if f.iv_rank is None:
            return None
        return f.iv_rank >= 50 if name == "iv_high" else f.iv_rank <= 50
    if name == "pre_event":
        return None if f.event_days is None else f.event_days <= 45
    return None


def state_vector(f: DayFeatures) -> dict[str, float | None]:
    """The analogue engine's view of a day: trend, volatility, IV, its rank, the term structure, momentum,
    event proximity — numbers only (a missing one stays missing)."""
    return {
        "trend": None if not (f.sma50 and f.sma200) else (f.sma50 / f.sma200 - 1),
        "ret20": f.ret20,
        "rv20": f.rv20,
        "iv": f.iv,
        "iv_rank": f.iv_rank,
        "iv_rv": f.iv_rv,
        "term": None
        if f.term_shape is None
        else {"backwardation": -1.0, "flat": 0.0, "contango": 1.0}.get(f.term_shape),
        "skew": f.skew,
        "event": None if f.event_days is None else float(f.event_days <= 30),
        "atr": f.atr14_pct,
    }
