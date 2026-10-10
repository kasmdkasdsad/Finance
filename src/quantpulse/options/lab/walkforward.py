"""Walk-forward validation: choose on the past, judge on what came after, roll forward, repeat.

Each window has three consecutive periods: **train** (the parent and a small, fixed set of its variations
are backtested; the one with the best risk-adjusted result on train is chosen), **validate** (the chosen one
is run; used only to confirm, never to choose again) and **test** (run once; never looked at before). The
window then rolls forward by ``step`` and everything repeats with data available at that time only.

Results: every window's train, validation and test metrics; the out-of-sample trades pooled across the test
periods; whether the chosen parameters stayed stable from window to window; the gap between train and test
(overfitting shows up as a collapse out of sample); and a pass/fail with its reasons. ``variants_tried`` is
reported: the more that were tried, the more a good result may be luck (see :mod:`.overfit`).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from quantpulse.options.fills import ExecutionModel
from quantpulse.options.lab.backtest import BacktestConfig, run
from quantpulse.options.lab.chains import ChainSource
from quantpulse.options.lab.features import DayFeatures
from quantpulse.options.lab.genome import Genome
from quantpulse.options.lab.metrics import trade_stats


@dataclass(frozen=True, slots=True)
class Window:
    train: tuple[date, date]
    validate: tuple[date, date]
    test: tuple[date, date]

    def as_dict(self) -> dict[str, list[str]]:
        return {k: [a.isoformat(), b.isoformat()] for k, (a, b) in
                (("train", self.train), ("validate", self.validate), ("test", self.test))}  # fmt: skip


def windows(start: date, end: date, *, train_days: int = 365 * 2, validate_days: int = 182, test_days: int = 182,
            step_days: int = 182) -> list[Window]:  # fmt: skip
    """Rolling windows over ``[start, end]``; the last one ends on or before ``end``."""
    out: list[Window] = []
    t0 = start
    while True:
        tr = (t0, t0 + timedelta(days=train_days - 1))
        va = (tr[1] + timedelta(days=1), tr[1] + timedelta(days=validate_days))
        te = (va[1] + timedelta(days=1), va[1] + timedelta(days=test_days))
        if te[1] > end:
            break
        out.append(Window(tr, va, te))
        t0 += timedelta(days=step_days)
    return out


def variants(parent: Genome, n: int = 4, seed: int = 3) -> list[Genome]:
    """The parent and ``n`` fixed neighbours (seeded): the whole search space of one walk-forward."""
    rng = random.Random(seed)
    out, seen = [parent], {parent.hash}
    for _ in range(n * 10):
        child = parent.mutate(rng, changes=1)
        if child.hash not in seen:
            seen.add(child.hash)
            out.append(child)
        if len(out) > n:
            break
    return out


def objective(metrics: Mapping[str, Any], min_trades: int) -> float:
    """What training maximizes: expectancy per dollar at risk, shrunk towards zero for few trades."""
    n = metrics.get("trades") or 0
    ror = metrics.get("expectancy_on_risk")
    if n < max(3, min_trades // 3) or ror is None:
        return -math.inf
    return float(ror) * n / (n + min_trades)


def walk_forward(
    parent: Genome,
    source: ChainSource,
    features: Mapping[str, Mapping[date, DayFeatures]],
    underlyings: Sequence[str],
    wins: Sequence[Window],
    *,
    model: ExecutionModel = ExecutionModel.REALISTIC,
    n_variants: int = 4,
    min_trades: int = 20,
    equity: float = 100_000.0,
) -> dict[str, Any]:
    cands = variants(parent, n_variants)
    rows: list[dict[str, Any]] = []
    oos: list[dict[str, Any]] = []
    chosen_hashes: list[str] = []
    train_ror, test_ror = [], []
    for w in wins:

        def bt(g: Genome, period: tuple[date, date]) -> dict[str, Any]:
            res = run(
                g,
                source,
                features,
                BacktestConfig(period[0], period[1], tuple(underlyings), equity=equity, model=model),
            )
            return {"metrics": res.metrics, "trades": res.trades}

        scored = [(objective(bt(g, w.train)["metrics"], min_trades), g) for g in cands]
        best_score, best = max(scored, key=lambda x: x[0])
        tr = bt(best, w.train)
        va = bt(best, w.validate)
        te = bt(best, w.test)
        chosen_hashes.append(best.hash)
        oos.extend(te["trades"])
        if tr["metrics"].get("expectancy_on_risk") is not None:
            train_ror.append(tr["metrics"]["expectancy_on_risk"])
        if te["metrics"].get("expectancy_on_risk") is not None:
            test_ror.append(te["metrics"]["expectancy_on_risk"])
        rows.append({
            "window": w.as_dict(),
            "chosen": best.hash,
            "chosen_is_parent": best.hash == parent.hash,
            "train_objective": None if math.isinf(best_score) else round(best_score, 5),
            "train": _brief(tr["metrics"]),
            "validate": _brief(va["metrics"]),
            "test": _brief(te["metrics"]),
        })  # fmt: skip
    pooled = trade_stats([t["pnl"] for t in oos], [t.get("max_loss") or 0.0 for t in oos])
    stability = 1.0 - (len(set(chosen_hashes)) - 1) / max(len(chosen_hashes), 1) if chosen_hashes else 0.0
    positive_windows = sum(1 for r in rows if (r["test"].get("expectancy_on_risk") or 0) > 0)
    mean_train = sum(train_ror) / len(train_ror) if train_ror else None
    mean_test = sum(test_ror) / len(test_ror) if test_ror else None
    reasons: list[str] = []
    if not wins:
        reasons.append("the data does not cover a single train/validate/test window")
    if (pooled.get("trades") or 0) < min_trades:
        reasons.append(f"{pooled.get('trades', 0)} out-of-sample trades (minimum {min_trades})")
    if (pooled.get("expectancy_on_risk") or 0) <= 0:
        reasons.append("out-of-sample expectancy is not positive")
    if (pooled.get("t_stat") or 0) < 1.0:
        reasons.append(f"out-of-sample t-statistic {pooled.get('t_stat')} below 1")
    if rows and positive_windows * 2 < len(rows):
        reasons.append(f"only {positive_windows} of {len(rows)} test windows positive")
    if mean_train is not None and mean_test is not None and mean_train > 0 and mean_test < 0.25 * mean_train:
        reasons.append(
            f"performance collapses out of sample (train {mean_train:.3f} vs test {mean_test:.3f} per $ at risk)"
        )
    return {
        "windows": rows,
        "variants_tried": len(cands),
        "execution_model": model.value,
        "oos": pooled,
        "oos_trades": oos,
        "parameter_stability": round(stability, 3),
        "mean_train_ror": mean_train,
        "mean_test_ror": mean_test,
        "positive_test_windows": positive_windows,
        "passed": not reasons,
        "reasons": reasons,
    }


def _brief(m: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "trades",
        "expectancy",
        "expectancy_on_risk",
        "win_rate",
        "profit_factor",
        "total_return",
        "max_drawdown",
        "sharpe",
    )
    return {k: m.get(k) for k in keys}
