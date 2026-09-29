"""StrategyDecay: has a strategy stopped working — with evidence, not after a few losses?

For each strategy with live evidence (shadow or paper trades, never pooled with backtests) the rolling
expectancy, hit rate, drawdown, slippage and regime mix are compared with what its validation predicted:

* **HEALTHY** — inside its own expected range;
* **WATCH** — the recent window is below expectation but the evidence is thin (few trades) or mixed;
* **DEGRADING** — the recent mean is below the lower bound of its validated range *and* a one-sided test
  says so (p < 0.10), or slippage runs well above what was modelled, or the drawdown exceeds the
  backtest's 95th-percentile drawdown;
* **BROKEN** — a CUSUM of the shortfall against the expected mean crosses its limit and the recent
  window is negative with p < 0.05;
* **RETIRED** — set by the promotion rules (history kept).

Nothing is decided on fewer than ``min_trades`` trades: until then the answer is WATCH at worst.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Expectation:
    """What validation said the strategy should do (per trade, per dollar at risk)."""

    mean: float
    sd: float
    p95_drawdown: float | None = None  # a negative fraction
    modelled_slippage: float | None = None  # dollars per trade


def cusum(x: Sequence[float], target: float, slack: float) -> float:
    """Downward CUSUM: the largest accumulated shortfall below ``target − slack``."""
    s, worst = 0.0, 0.0
    for v in x:
        s = min(0.0, s + (v - (target - slack)))
        worst = min(worst, s)
    return -worst


def assess(
    returns: Sequence[float],
    exp: Expectation,
    *,
    window: int = 20,
    min_trades: int = 15,
    slippage: Sequence[float] | None = None,
    equity_drawdown: float | None = None,
) -> dict[str, Any]:
    """``returns``: live trade results per dollar at risk, oldest first."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out: dict[str, Any] = {"trades": n, "expected_mean": exp.mean}
    if n < min_trades:
        out.update(status="HEALTHY" if n == 0 else "WATCH" if r.mean() < exp.mean else "HEALTHY",
                   reasons=[f"only {n} live trades: too few to judge decay"])  # fmt: skip
        return out
    recent = r[-window:]
    from scipy.stats import ttest_1samp

    sd = max(exp.sd, 1e-6)
    lower = exp.mean - 1.64 * sd / math.sqrt(len(recent))
    p_below = float(ttest_1samp(recent, exp.mean, alternative="less").pvalue)
    p_negative = float(ttest_1samp(recent, 0.0, alternative="less").pvalue)
    c = cusum(r.tolist(), exp.mean, 0.5 * sd)
    limit = 5 * sd
    hit = float((recent > 0).mean())
    reasons: list[str] = []
    status = "HEALTHY"
    if recent.mean() < lower:
        status = "WATCH"
        reasons.append(f"recent mean {recent.mean():.3f} below the expected range (lower bound {lower:.3f})")
    if recent.mean() < lower and p_below < 0.10:
        status = "DEGRADING"
        reasons.append(f"below expectation with p = {p_below:.3f}")
    if slippage is not None and exp.modelled_slippage and len(slippage) >= min_trades:
        ratio = float(np.mean(slippage[-window:])) / exp.modelled_slippage
        out["slippage_ratio"] = round(ratio, 3)
        if ratio > 1.5:
            status = "DEGRADING" if status != "BROKEN" else status
            reasons.append(f"slippage {ratio:.1f}× what was modelled")
    if equity_drawdown is not None and exp.p95_drawdown is not None and equity_drawdown < exp.p95_drawdown:
        status = "DEGRADING" if status != "BROKEN" else status
        reasons.append(
            f"drawdown {equity_drawdown:.1%} beyond the backtest's 95th percentile {exp.p95_drawdown:.1%}"
        )
    if c > limit and recent.mean() < 0 and p_negative < 0.05:
        status = "BROKEN"
        reasons.append(
            f"CUSUM shortfall {c:.2f} beyond {limit:.2f} and the recent window is negative (p = {p_negative:.3f})"
        )
    out.update(status=status, reasons=reasons or ["within its expected range"], recent_mean=round(float(recent.mean()), 4),
               hit_rate=round(hit, 3), cusum=round(c, 4), p_below_expectation=round(p_below, 4))  # fmt: skip
    return out
