"""Cross-sectional multi-factor stock screen producing a 1–10 rating.

Factors (computed from daily closes, most recent last):

========================  ===========================================================  ======
factor                    definition                                                   weight
========================  ===========================================================  ======
``momentum_12_1``         close[t−21] / close[t−252] − 1  (skips the latest month)      0.20
``momentum_3m``           close[t] / close[t−63] − 1                                   0.10
``trend``                 mean(close/SMA50 − 1, SMA50/SMA200 − 1)                       0.25
``risk_adjusted``         annualised mean / s.d. of the last 126 daily returns          0.25
``low_volatility``        −(annualised s.d. of the last 63 daily returns)               0.10
``reversal``              50 − RSI(14)  (short-term pullbacks score higher)             0.10
========================  ===========================================================  ======

Each factor is z-scored across the universe (winsorised at ±3); a stock's composite is the weighted mean
of its available factor z-scores. Composites are standardised again and mapped to a rating with
``rating = round(1 + 9·Φ(z))`` clipped to 1…10, so ratings are *relative to the screened universe*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

WEIGHTS: dict[str, float] = {
    "momentum_12_1": 0.20,
    "momentum_3m": 0.10,
    "trend": 0.25,
    "risk_adjusted": 0.25,
    "low_volatility": 0.10,
    "reversal": 0.10,
}
LABELS: dict[str, str] = {
    "momentum_12_1": "12-1 month momentum",
    "momentum_3m": "3-month momentum",
    "trend": "trend (50/200-day)",
    "risk_adjusted": "risk-adjusted return",
    "low_volatility": "low volatility",
    "reversal": "short-term pullback",
}
MIN_BARS = 64
WINSOR = 3.0


@dataclass(slots=True)
class RawFactors:
    momentum_12_1: float | None
    momentum_3m: float | None
    trend: float | None
    risk_adjusted: float | None
    volatility_3m: float | None
    rsi_14: float | None

    def scored(self) -> dict[str, float | None]:
        """Factor values oriented so that higher is better."""
        return {
            "momentum_12_1": self.momentum_12_1,
            "momentum_3m": self.momentum_3m,
            "trend": self.trend,
            "risk_adjusted": self.risk_adjusted,
            "low_volatility": None if self.volatility_3m is None else -self.volatility_3m,
            "reversal": None if self.rsi_14 is None else 50.0 - self.rsi_14,
        }


@dataclass(slots=True)
class ScreenResult:
    symbol: str
    factors: RawFactors
    z: dict[str, float] = field(default_factory=dict)
    composite: float = 0.0
    standardized: float = 0.0
    rating: int = 5
    drivers: list[str] = field(default_factory=list)


def rsi(closes: np.ndarray, period: int = 14) -> float | None:
    """Wilder's RSI using the full history for smoothing."""
    if closes.size < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, lo in zip(gains[period:], losses[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + lo) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_factors(closes: Sequence[float]) -> RawFactors:
    c = np.asarray(closes, dtype=float)
    n = c.size
    if n < MIN_BARS or np.any(c <= 0):
        raise ValueError(f"need at least {MIN_BARS} positive closes, got {n}")
    rets = c[1:] / c[:-1] - 1.0
    mom_12_1 = float(c[-22] / c[-253] - 1.0) if n >= 253 else None
    mom_3m = float(c[-1] / c[-64] - 1.0)
    trend_parts: list[float] = []
    if n >= 50:
        sma50 = float(c[-50:].mean())
        trend_parts.append(c[-1] / sma50 - 1.0)
        if n >= 200:
            trend_parts.append(sma50 / float(c[-200:].mean()) - 1.0)
    trend = float(np.mean(trend_parts)) if trend_parts else None
    risk_adj = None
    if n >= 127:
        window = rets[-126:]
        sd = float(window.std(ddof=1))
        risk_adj = float(window.mean() / sd * math.sqrt(252)) if sd > 0 else None
    vol = float(rets[-63:].std(ddof=1) * math.sqrt(252))
    return RawFactors(mom_12_1, mom_3m, trend, risk_adj, vol, rsi(c))


def _zscores(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    sd = float(arr.std(ddof=0))
    if sd == 0 or len(arr) < 2:
        return dict.fromkeys(values, 0.0)
    mean = float(arr.mean())
    return {k: float(np.clip((v - mean) / sd, -WINSOR, WINSOR)) for k, v in values.items()}


def rating_from_z(z: float) -> int:
    return int(min(10, max(1, math.floor(1 + 9 * float(norm.cdf(z)) + 0.5))))


def screen(universe: Mapping[str, RawFactors]) -> list[ScreenResult]:
    """Score and rank a universe; returns results sorted best-first."""
    results = {s: ScreenResult(symbol=s, factors=f) for s, f in universe.items()}
    for factor in WEIGHTS:
        available = {s: v for s, r in results.items() if (v := r.factors.scored()[factor]) is not None}
        for s, z in _zscores(available).items():
            results[s].z[factor] = z
    for r in results.values():
        weight = sum(WEIGHTS[f] for f in r.z)
        r.composite = sum(WEIGHTS[f] * z for f, z in r.z.items()) / weight if weight else 0.0
        contributions = sorted(((WEIGHTS[f] * z, f) for f, z in r.z.items()), reverse=True)
        r.drivers = [LABELS[f] for c, f in contributions if c > 0][:2]
    standardized = _zscores({s: r.composite for s, r in results.items()})
    for s, r in results.items():
        r.standardized = standardized.get(s, 0.0)
        r.rating = rating_from_z(r.standardized)
    return sorted(results.values(), key=lambda r: (-r.composite, r.symbol))
