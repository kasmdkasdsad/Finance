"""Do relationships QuantPulse learned still hold?

A relationship is a slope (or correlation) between two measured quantities — IV minus realized volatility and
the next month's realized volatility, a strategy's result and the IV rank it entered at, slippage and
micro-volatility, the IV/RV premium and micro-volatility, two underlyings' returns. Each is re-estimated on a
recent window and compared with its established estimate:

==============  ============================================================================================
STABLE          the recent estimate is inside the established one's confidence band
STRENGTHENED    same sign, significantly larger
WEAKENED        same sign, significantly smaller (but still distinguishable from zero)
DISAPPEARED     no longer distinguishable from zero, where it used to be
INVERTED        significantly of the opposite sign
EMERGED         was indistinguishable from zero, now is not
INSUFFICIENT    too few observations to say
==============  ============================================================================================

Every estimate is appended to the relationship's history — the old estimate is never overwritten, so the
record shows when a relationship held, when it faded, and whether it came back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Estimate:
    slope: float | None
    se: float | None
    r: float | None  # correlation
    n: int

    @property
    def significant(self) -> bool:
        return (
            self.slope is not None and self.se is not None and self.se > 0 and abs(self.slope / self.se) >= 2
        )

    def as_dict(self) -> dict[str, float | int | None]:
        return {"slope": self.slope, "se": self.se, "r": self.r, "n": self.n}


def estimate(x: Sequence[float | None], y: Sequence[float | None], min_n: int = 15) -> Estimate:
    pairs = [(a, b) for a, b in zip(x, y, strict=False) if a is not None and b is not None
             and math.isfinite(a) and math.isfinite(b)]  # fmt: skip
    n = len(pairs)
    if n < min_n:
        return Estimate(None, None, None, n)
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    if xa.std() == 0 or ya.std() == 0:
        return Estimate(0.0, None, 0.0, n)
    slope, intercept = np.polyfit(xa, ya, 1)
    resid = ya - (slope * xa + intercept)
    se = math.sqrt(float((resid**2).sum()) / max(n - 2, 1)) / (float(xa.std()) * math.sqrt(n))
    return Estimate(round(float(slope), 6), round(se, 6), round(float(np.corrcoef(xa, ya)[0, 1]), 4), n)


def status(established: Estimate, recent: Estimate) -> dict[str, Any]:
    """Compare a recent estimate with the established one (a z-test on the difference of slopes)."""
    if established.slope is None or recent.slope is None or not established.se or not recent.se:
        return {"status": "INSUFFICIENT", "z": None}
    diff = recent.slope - established.slope
    z = diff / math.sqrt(established.se**2 + recent.se**2)
    was, now = established.significant, recent.significant
    if not was:
        state = "EMERGED" if now else "STABLE"
    elif not now and abs(z) >= 2:
        state = "DISAPPEARED"
    elif now and math.copysign(1, recent.slope) != math.copysign(1, established.slope):
        state = "INVERTED"
    elif abs(z) < 2:
        state = "STABLE"
    elif abs(recent.slope) > abs(established.slope):
        state = "STRENGTHENED"
    else:
        state = "WEAKENED" if now else "DISAPPEARED"
    return {
        "status": state,
        "z": round(float(z), 3),
        "established": established.as_dict(),
        "recent": recent.as_dict(),
    }


def rolling(
    x: Sequence[float | None], y: Sequence[float | None], window: int, step: int | None = None
) -> list[Estimate]:
    """The relationship re-estimated on consecutive windows (its history)."""
    step = step or window
    return [
        estimate(x[i : i + window], y[i : i + window]) for i in range(0, max(len(x) - window + 1, 0), step)
    ]


RELATIONSHIPS = {
    "vrp_predicts_rv": "IV minus realized volatility today → the realized volatility that follows (is implied an "
    "unbiased forecast, or does it carry a premium?)",
    "strategy_vs_iv_rank": "a strategy's result per $ at risk against the IV rank it entered at",
    "slippage_vs_micro_vol": "execution slippage against the underlying's 1-minute realized volatility",
    "vrp_vs_micro_vol": "the IV/RV premium against micro-volatility (does intraday noise get priced?)",
    "expectancy_vs_micro_vol": "strategy results against micro-volatility at entry",
    "spread_vs_micro_vol": "option spreads against micro-volatility",
    "pair_correlation": "the correlation of two underlyings' daily returns",
}
