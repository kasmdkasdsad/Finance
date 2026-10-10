"""Overfitting defences: when a result is probably luck, a mirage, or a fitted curve.

* :func:`overfit_risk` — the OVERFIT_RISK_SCORE (0 = no warning signs, 1 = all of them) with every warning
  named: too many parameters for the trades, a small sample, train/test divergence, an out-of-sample collapse,
  unstable parameters across windows, dependence on a single symbol or a single regime, a result too good to
  be true, an edge that exists only under flattering fills, and excessive search. A score of 0.5 or more
  means: DO NOT PROMOTE.
* :func:`deflated_sharpe` — the probability that the true Sharpe ratio is positive after accounting for how
  many variants were tried and for non-normal returns (Bailey & López de Prado, 2014).
* :func:`benjamini_hochberg` — controls the false discovery rate when a population of strategies is tested at
  once: with thousands of candidates, some look good by chance, and they must not graduate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scipy.stats import norm

EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, sd_of_sharpes: float = 1.0) -> float:
    """The Sharpe ratio the best of ``n_trials`` pure-luck strategies is expected to reach."""
    if n_trials <= 1:
        return 0.0
    return sd_of_sharpes * (
        (1 - EULER) * norm.ppf(1 - 1 / n_trials) + EULER * norm.ppf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int, skew: float = 0.0, kurtosis: float = 0.0,
                    sd_of_sharpes: float = 1.0) -> float | None:  # fmt: skip
    """P(true Sharpe > the best-of-``n_trials`` luck benchmark). ``sharpe`` per observation (not annualized),
    ``kurtosis`` excess. ``None`` with fewer than 3 observations."""
    if n_obs < 3:
        return None
    bench = expected_max_sharpe(n_trials, sd_of_sharpes / math.sqrt(n_obs))
    denom = math.sqrt(max(1e-12, 1 - skew * sharpe + (kurtosis + 2) / 4 * sharpe * sharpe))
    return float(norm.cdf((sharpe - bench) * math.sqrt(n_obs - 1) / denom))


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.10) -> list[bool]:
    """Which hypotheses are discoveries at false discovery rate ``q``."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= q * rank / m:
            cutoff = rank
    keep = set(order[:cutoff]) if cutoff > 0 else set()
    return [i in keep for i in range(m)]


def p_value_from_t(t: float | None, n: int) -> float:
    """One-sided p-value that the mean is > 0 (Student t)."""
    if t is None or n < 3:
        return 1.0
    from scipy.stats import t as student

    return float(student.sf(t, n - 1))


def share_of_pnl(groups: Mapping[str, float]) -> tuple[str | None, float]:
    """The group that carries the largest share of the total positive P&L."""
    pos = {k: v for k, v in groups.items() if v > 0}
    total = sum(pos.values())
    if total <= 0:
        return None, 0.0
    k = max(pos, key=lambda x: pos[x])
    return k, pos[k] / total


def overfit_risk(
    *,
    parameter_count: int,
    trades: int,
    train_ror: float | None,
    test_ror: float | None,
    parameter_stability: float | None,
    pnl_by_symbol: Mapping[str, float],
    pnl_by_regime: Mapping[str, float],
    sharpe_annual: float | None,
    win_rate: float | None,
    ror_by_model: Mapping[str, float | None],
    variants_tried: int,
    deflated: float | None,
) -> dict[str, Any]:
    warnings: list[tuple[str, float]] = []
    if trades and parameter_count > trades / 10:
        warnings.append(
            (f"{parameter_count} parameters for {trades} trades (more than one per ten trades)", 0.2)
        )
    if trades < 30:
        warnings.append((f"small sample: {trades} trades", 0.2 if trades >= 10 else 0.35))
    if train_ror is not None and test_ror is not None:
        if train_ror > 0 and test_ror <= 0:
            warnings.append((f"sign flips out of sample (train {train_ror:.3f}, test {test_ror:.3f})", 0.35))
        elif train_ror > 0 and test_ror < 0.5 * train_ror:
            warnings.append(
                (f"train/test divergence (test {test_ror:.3f} under half of train {train_ror:.3f})", 0.2)
            )
    if parameter_stability is not None and parameter_stability < 0.5:
        warnings.append((f"unstable parameters across windows (stability {parameter_stability:.2f})", 0.15))
    sym, share = share_of_pnl(pnl_by_symbol)
    if sym is not None and share > 0.6 and len(pnl_by_symbol) > 1:
        warnings.append((f"{share:.0%} of the profit from one symbol ({sym})", 0.2))
    reg, rshare = share_of_pnl(pnl_by_regime)
    if reg is not None and rshare > 0.8 and len(pnl_by_regime) > 1:
        warnings.append((f"{rshare:.0%} of the profit from one regime ({reg})", 0.15))
    if sharpe_annual is not None and sharpe_annual > 3:
        warnings.append((f"suspiciously high Sharpe ({sharpe_annual:.1f})", 0.2))
    if win_rate is not None and win_rate > 0.9 and trades >= 10:
        warnings.append((f"suspiciously high win rate ({win_rate:.0%}): check for hidden tail risk", 0.1))
    flattering = [ror_by_model.get(m) for m in ("OPTIMISTIC", "MIDPOINT")]
    honest = [ror_by_model.get(m) for m in ("REALISTIC", "PESSIMISTIC")]
    if any(x is not None and x > 0 for x in flattering) and all(x is None or x <= 0 for x in honest):
        warnings.append(("the edge exists only under flattering fills (mid or better)", 0.4))
    if variants_tried > 20 and (deflated is None or deflated < 0.9):
        warnings.append((f"{variants_tried} variants tried; the deflated Sharpe does not survive", 0.25))
    elif deflated is not None and deflated < 0.5:
        warnings.append((f"deflated Sharpe probability {deflated:.2f}: indistinguishable from luck", 0.2))
    score = 1 - math.prod(1 - w for _, w in warnings)
    return {"score": round(score, 3), "warnings": [w for w, _ in warnings], "promote_blocked": score >= 0.5}
