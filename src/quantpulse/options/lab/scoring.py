"""The strategy score: many dimensions, each graded on its own — never one number that hides a weakness.

Dimensions: expected return, risk-adjusted return, drawdown, tail risk, robustness (Monte Carlo), out-of-sample
stability, execution quality, liquidity, regime dependence, sample size, data quality and overfit risk. Each
is ``good``, ``ok``, ``poor`` or ``unknown`` with its value; a strategy is *eligible* for preference only when
no critical dimension is poor and none is unknown.
"""

from __future__ import annotations

from typing import Any

CRITICAL = ("expected_return", "tail_risk", "robustness", "out_of_sample", "sample_size", "overfit_risk")


def _grade(value: float | None, good: float, ok: float, *, higher_is_better: bool = True) -> str:
    if value is None:
        return "unknown"
    if higher_is_better:
        return "good" if value >= good else "ok" if value >= ok else "poor"
    return "good" if value <= good else "ok" if value <= ok else "poor"


def score(
    *,
    metrics: dict[str, Any],
    walkforward: dict[str, Any] | None,
    montecarlo: dict[str, Any] | None,
    overfit: dict[str, Any] | None,
    regimes: dict[str, float] | None,
    grade: str,
    slippage_ratio: float | None = None,
    liquidity: float | None = None,
) -> dict[str, Any]:
    d: dict[str, dict[str, Any]] = {}

    def put(name: str, value: Any, g: str) -> None:
        d[name] = {"value": value, "grade": g}

    ror = metrics.get("expectancy_on_risk")
    put("expected_return", ror, _grade(ror, 0.05, 0.0))
    sh = metrics.get("sharpe")
    put("risk_adjusted", sh, _grade(sh, 1.0, 0.3))
    dd = metrics.get("max_drawdown")
    put("drawdown", dd, _grade(-dd if dd is not None else None, 0.10, 0.25, higher_is_better=False))
    ruin = (montecarlo or {}).get("worst_risk_of_ruin")
    put("tail_risk", ruin, _grade(ruin, 0.0, 0.01, higher_is_better=False))
    p5 = ((montecarlo or {}).get("scenarios", {}).get("bootstrap", {}).get("final_pnl") or {}).get("p5")
    put("robustness", p5, _grade(p5, 0.0, -1e12) if p5 is not None else "unknown")
    if p5 is not None and p5 < 0:
        d["robustness"]["grade"] = "ok" if (montecarlo or {}).get("worst_risk_of_ruin", 1) <= 0.01 else "poor"
    oos = (walkforward or {}).get("oos", {}).get("expectancy_on_risk")
    put("out_of_sample", oos, _grade(oos, 0.03, 0.0) if walkforward else "unknown")
    put("execution_quality", slippage_ratio, _grade(slippage_ratio, 1.1, 1.5, higher_is_better=False))
    put("liquidity", liquidity, _grade(liquidity, 0.6, 0.3))
    if regimes:
        positive = sum(1 for v in regimes.values() if v > 0)
        share = positive / len(regimes)
        put("regime_dependence", round(share, 3), _grade(share, 0.6, 0.34))
    else:
        put("regime_dependence", None, "unknown")
    n = metrics.get("trades") or 0
    put("sample_size", n, _grade(n, 100, 30))
    put("data_quality", grade, "good" if grade == "recorded" else "ok" if grade == "model" else "unknown")
    ofs = (overfit or {}).get("score")
    put("overfit_risk", ofs, _grade(ofs, 0.2, 0.49, higher_is_better=False))
    blockers = [k for k in CRITICAL if d[k]["grade"] in ("poor", "unknown")]
    return {"dimensions": d, "eligible": not blockers, "blocking": blockers,
            "note": "model-priced evidence" if grade == "model" else None}  # fmt: skip
