"""Validation: why one attractive backtest is never enough.

* **Walk-forward** — the history after the features' warm-up is cut into consecutive train/test windows.
  In each, the parameter set with the best *train* active Sharpe (strategy minus the equal-weight universe,
  so market beta is not mistaken for skill) is chosen from the spec's grid and then run, untouched, on the
  *test* window that follows. Only the stitched test windows count as out-of-sample performance; the
  in-sample/out-of-sample ratio shows how much of the backtest was fitting.
* **Deflated Sharpe ratio** (Bailey & López de Prado) — the probability that the out-of-sample active Sharpe is
  above what the best of ``N`` luck-only trials would reach, given the returns' skew and kurtosis. Every grid
  combination of every strategy the lab has ever tried is a trial (:class:`Population`), so the bar rises as
  the search tries more ideas; the spread of the trials is the larger of the variants' spread within each
  training window and the spread of the tried strategies' out-of-sample Sharpe ratios.
* **Random portfolios** — the percentile of the strategy's Sharpe among portfolios of the same number of
  random names on the same schedule.
* **Stress tests** — the benchmark's worst drawdown episodes and worst 20-session windows in the sample
  (strategy vs benchmark), doubled costs, and trading two sessions late.

:mod:`.scrutiny` then looks on purpose for reasons it may still not work — the cost it can bear, its capacity,
the markets its edge comes from, its drawdowns, a few lucky names, and whether nearby parameters work too.

:func:`gates` turns these into pass/fail checks; a strategy is *validated* only if every gate passes, and
every report lists the reasons it may not work (``refutation``), validated or not.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantpulse.quant.risk import max_drawdown, sharpe_ratio

from .backtest import BacktestResult, backtest, equal_weight, metrics, random_sharpes, scores
from .scrutiny import refute, scrutinise
from .spec import StrategySpec

EULER = 0.5772156649
NORMAL = statistics.NormalDist()


@dataclass(frozen=True)
class Population:
    """The strategies the lab has tried so far: each one's grid counts as trials in the deflated Sharpe."""

    tried: int = 1  # strategies tested, this one included
    sharpe_variance: float = 0.0  # variance of their out-of-sample active Sharpe ratios (annualised)


@dataclass(frozen=True)
class Thresholds:
    min_days: int = 500  # out-of-sample sessions needed for a verdict
    min_oos_sharpe: float = 0.0
    min_degradation: float = 0.3  # OOS Sharpe / IS Sharpe
    min_fold_win_rate: float = 0.6  # folds where the strategy beat the equal-weight baseline
    min_dsr: float = 0.9
    min_random_percentile: float = 0.9
    stress_drawdown_multiple: float = 1.5
    min_neighbours_positive: float = 0.6  # nearby parameter sets that still beat equal weight
    min_cost_headroom: float = 2.0  # break-even cost must be this multiple of the assumed cost


def deflated_sharpe(returns: np.ndarray, trials: int, trial_variance: float) -> float | None:
    """Probability that the true Sharpe exceeds the expected maximum of ``trials`` luck-only trials whose
    (annualised) Sharpe ratios vary with ``trial_variance``. One trial reduces it to the probabilistic
    Sharpe ratio (against zero)."""
    r = np.asarray(returns, dtype=float)
    t = r.size
    sd = float(r.std(ddof=1)) if t > 1 else 0.0
    if t < 30 or sd == 0:
        return None
    sr = float(r.mean()) / sd
    n = max(trials, 1)
    var = trial_variance / 252  # annualised → daily Sharpe units
    if n > 1 and var > 0:
        sr0 = math.sqrt(var) * (
            (1 - EULER) * NORMAL.inv_cdf(1 - 1 / n) + EULER * NORMAL.inv_cdf(1 - 1 / (n * math.e))
        )
    else:
        sr0 = 0.0
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return None
    return float(NORMAL.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)))


def warmup(spec: StrategySpec, features: dict[str, pd.DataFrame]) -> int:
    s = scores(spec, features)
    ok = np.flatnonzero((s.notna().sum(axis=1) >= spec.top_n).to_numpy())
    return int(ok[0]) if ok.size else len(s)


def _active(res: BacktestResult, base: BacktestResult) -> pd.Series:
    """Strategy minus the equal-weight baseline, day by day (the selection skill, without the market)."""
    return (res.returns - base.returns.reindex(res.returns.index).fillna(0.0)).dropna()


def _sr(r: pd.Series) -> float | None:
    return sharpe_ratio(r.to_numpy()) if r.size > 20 else None


def walk_forward(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    *,
    train: int = 252,
    test: int = 126,
    population: Population | None = None,
) -> dict[str, Any]:
    """Choose the grid variant with the best *active* Sharpe (vs the equal-weight universe) on each training
    window, run it untouched on the next test window, and stitch the test windows together."""
    population = population or Population()
    variants = spec.variants()
    shared = scores(spec, features)  # the grid changes sizing and timing, not the ranking
    start = warmup(spec, features)
    n = len(close)
    folds: list[dict[str, Any]] = []
    oos: list[pd.Series] = []
    oos_bench: list[pd.Series] = []
    oos_active: list[pd.Series] = []
    fold_variances: list[float] = []  # spread of the variants' active Sharpe within each training window
    s = start
    while s + train + test <= n:
        best, best_sr = variants[0], -math.inf
        tried: list[float] = []
        for v in variants:
            res = backtest(v, features, close, benchmark, s, s + train, score=shared)
            sr = _sr(_active(res, equal_weight(close, benchmark, v, s, s + train)))
            if sr is not None:
                tried.append(sr)
                if sr > best_sr:
                    best, best_sr = v, sr
        if len(tried) > 1:
            fold_variances.append(statistics.pvariance(tried))
        out = backtest(best, features, close, benchmark, s + train, s + train + test, score=shared)
        base = equal_weight(close, benchmark, best, s + train, s + train + test)
        active = _active(out, base)
        oos_sr = _sr(active)
        folds.append(
            {
                "train": [str(close.index[s].date()), str(close.index[s + train - 1].date())],
                "test": [str(close.index[s + train].date()), str(close.index[s + train + test - 1].date())],
                "params": best.params(),
                "is_active_sharpe": round(best_sr, 4) if best_sr > -math.inf else None,
                "oos_active_sharpe": round(oos_sr, 4) if oos_sr is not None else None,
                "oos_return": round(float(np.prod(1 + out.returns.to_numpy()) - 1), 5),
                "baseline_return": round(float(np.prod(1 + base.returns.to_numpy()) - 1), 5),
                "beat_baseline": bool(
                    np.prod(1 + out.returns.to_numpy()) > np.prod(1 + base.returns.to_numpy())
                ),
            }
        )
        oos.append(out.returns)
        oos_bench.append(out.benchmark)
        oos_active.append(active)
        s += test
    if not folds:
        return {
            "folds": [],
            "insufficient": True,
            "reason": f"needs {train + test} sessions after a {start}-session warm-up, has {n}",
        }
    stitched = BacktestResult(pd.concat(oos), pd.concat(oos_bench))
    active_all = pd.concat(oos_active)
    is_sr = [f["is_active_sharpe"] for f in folds if f["is_active_sharpe"] is not None]
    mean_is = sum(is_sr) / len(is_sr) if is_sr else None
    oos_active_sr = _sr(active_all)
    return {
        "folds": folds,
        "oos": metrics(stitched),
        "oos_active_sharpe": round(oos_active_sr, 4) if oos_active_sr is not None else None,
        "is_active_sharpe": round(mean_is, 4) if mean_is is not None else None,
        "degradation": (
            round(oos_active_sr / mean_is, 4)
            if oos_active_sr is not None and mean_is and mean_is > 0
            else None
        ),
        "fold_win_rate": round(sum(f["beat_baseline"] for f in folds) / len(folds), 4),
        "trials": len(variants) * max(1, population.tried),
        "strategies_tried": max(1, population.tried),
        "dsr": deflated_sharpe(
            active_all.to_numpy(),
            len(variants) * max(1, population.tried),
            max(
                sum(fold_variances) / len(fold_variances) if fold_variances else 0.0,
                population.sharpe_variance,
            ),
        ),
        "returns": stitched.returns,
    }


def _episodes(bench: pd.Series, count: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """The benchmark's worst peak-to-trough drawdowns (non-overlapping)."""
    wealth = (1 + bench.fillna(0.0)).cumprod()
    out: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    used = pd.Series(False, index=wealth.index)
    for _ in range(count):
        w = wealth.where(~used)
        if w.dropna().size < 5:
            break
        peak = w.cummax()
        dd = w / peak - 1
        trough = dd.idxmin()
        depth = float(dd.min())
        if not depth < -0.02:
            break
        start = w.loc[:trough].idxmax()
        out.append((start, trough, depth))
        used.loc[start:trough] = True
    return out


def stress(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    returns: pd.Series,
    bench_returns: pd.Series,
    start: int,
) -> dict[str, Any]:
    windows = []
    for a, b, depth in _episodes(bench_returns):
        strat = returns.loc[a:b]
        windows.append(
            {
                "kind": "benchmark drawdown",
                "from": str(a.date()),
                "to": str(b.date()),
                "benchmark": round(depth, 4),
                "strategy": round(float(np.prod(1 + strat.to_numpy()) - 1), 4),
                "strategy_max_drawdown": round(max_drawdown(strat.to_numpy()), 4) if strat.size else 0.0,
            }
        )
    rolling = bench_returns.rolling(20).apply(lambda x: float(np.prod(1 + x) - 1), raw=True).dropna()
    taken: list[pd.Timestamp] = []
    for end, value in rolling.sort_values().items():
        if len(taken) >= 3 or value > -0.03:
            break
        if any(abs((end - t).days) < 30 for t in taken):
            continue
        taken.append(end)
        idx = bench_returns.index.get_loc(end)
        begin = bench_returns.index[max(0, idx - 19)]
        strat = returns.loc[begin:end]
        windows.append(
            {
                "kind": "worst 20 sessions",
                "from": str(begin.date()),
                "to": str(end.date()),
                "benchmark": round(float(value), 4),
                "strategy": round(float(np.prod(1 + strat.to_numpy()) - 1), 4),
                "strategy_max_drawdown": round(max_drawdown(strat.to_numpy()), 4) if strat.size else 0.0,
            }
        )
    costly = backtest(spec, features, close, benchmark, start, cost_multiplier=2.0)
    late = backtest(spec, features, close, benchmark, start, extra_lag=2)
    return {
        "windows": windows,
        "double_costs_sharpe": metrics(costly).get("sharpe"),
        "two_sessions_late_sharpe": metrics(late).get("sharpe"),
    }


def gates(report: dict[str, Any], th: Thresholds) -> list[dict[str, Any]]:
    wf = report["walk_forward"]
    oos = wf.get("oos") or {}
    st = report["stress"]
    days = int(oos.get("days") or 0)
    checks: list[tuple[str, bool, str]] = [
        ("enough out-of-sample history", days >= th.min_days, f"{days} sessions (needs {th.min_days})"),
        ("positive out-of-sample Sharpe", (oos.get("sharpe") or -1) > th.min_oos_sharpe, f"{oos.get('sharpe')}"),
        ("adds value out of sample (active Sharpe vs equal weight > 0)", (wf.get("oos_active_sharpe") or -1) > 0,
         f"{wf.get('oos_active_sharpe')}"),
        ("survives out of sample", (wf.get("degradation") or 0) >= th.min_degradation,
         f"OOS/IS active Sharpe {wf.get('degradation')} (needs ≥ {th.min_degradation})"),
        ("beats the equal-weight baseline in most folds", (wf.get("fold_win_rate") or 0) >= th.min_fold_win_rate,
         f"{wf.get('fold_win_rate')} of folds (needs ≥ {th.min_fold_win_rate})"),
        ("not explained by trying many variants (deflated Sharpe)", (wf.get("dsr") or 0) >= th.min_dsr,
         f"DSR {wf.get('dsr')} over {wf.get('trials')} trials, {wf.get('strategies_tried', 1)} strategies tried "
         f"(needs ≥ {th.min_dsr})"),
        ("better than random portfolios", (report.get("random_percentile") or 0) >= th.min_random_percentile,
         f"percentile {report.get('random_percentile')} (needs ≥ {th.min_random_percentile})"),
        ("survives doubled costs", (st.get("double_costs_sharpe") or -1) > 0, f"Sharpe {st.get('double_costs_sharpe')}"),
        ("survives trading two sessions late", (st.get("two_sessions_late_sharpe") or -1) > 0,
         f"Sharpe {st.get('two_sessions_late_sharpe')}"),
    ]  # fmt: skip
    bad = [
        w for w in st.get("windows", [])
        if w["strategy_max_drawdown"] < th.stress_drawdown_multiple * min(w["benchmark"], 0) - 0.05
    ]  # fmt: skip
    checks.append(("no disproportionate losses in stress windows", not bad,
                   f"{len(bad)} of {len(st.get('windows', []))} windows worse than {th.stress_drawdown_multiple}× the benchmark"))  # fmt: skip
    sc = report.get("scrutiny")
    if sc is not None:
        sens = sc.get("sensitivity") or {}
        share = sens.get("positive_share")
        checks.append(("robust to nearby parameters", share is not None and share >= th.min_neighbours_positive,
                       f"{share} of nearby parameter sets beat equal weight (needs ≥ {th.min_neighbours_positive})"))  # fmt: skip
        c = sc.get("costs") or {}
        need = th.min_cost_headroom * max(float(c.get("assumed_cost_bps") or 0), 1.0)
        checks.append(("edge survives realistic costs", float(c.get("break_even_bps") or 0) >= need,
                       f"break-even {c.get('break_even_bps')}bp per unit traded (needs ≥ {need:g}bp)"))  # fmt: skip
        cap = (sc.get("capacity") or {}).get("capacity_usd")
        capital = float(report.get("capital") or 0)
        checks.append(("capacity covers the paper book", cap is not None and cap >= capital,
                       f"${cap:,.0f} at {100 * (sc.get('capacity') or {}).get('participation', 0):.0f}% of daily volume vs ${capital:,.0f}"
                       if cap is not None else "unknown (no volume data)"))  # fmt: skip
    return [{"gate": g, "passed": bool(p), "detail": d} for g, p, d in checks]


def validate(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    th: Thresholds | None = None,
    *,
    volume: pd.DataFrame | None = None,
    capital: float = 100_000.0,
    population: Population | None = None,
) -> dict[str, Any]:
    """The full report: backtest, baselines, walk-forward, overfitting checks, stress tests, scrutiny (the
    reasons it may not work), gates."""
    th = th or Thresholds()
    start = warmup(spec, features)
    full = backtest(spec, features, close, benchmark, start)
    base = equal_weight(close, benchmark, spec, start, len(close))
    wf = walk_forward(spec, features, close, benchmark, population=population)
    sharpe = metrics(full).get("sharpe")
    randoms = random_sharpes(close, spec, start, len(close))
    percentile = (
        round(sum(1 for r in randoms if r < sharpe) / len(randoms), 4)
        if randoms and sharpe is not None
        else None
    )
    report: dict[str, Any] = {
        "spec": spec.to_dict(),
        "backtest": metrics(full),
        "baselines": {"equal_weight": metrics(base), "random_median_sharpe": round(float(np.median(randoms)), 4) if randoms else None},
        "walk_forward": {k: v for k, v in wf.items() if k != "returns"},
        "random_percentile": percentile,
        "stress": stress(spec, features, close, benchmark, full.returns, full.benchmark, start) if full.days > 40 else {"windows": []},
        "last_holdings": full.holdings[-1][1] if full.holdings else [],
        "capital": capital,
    }  # fmt: skip
    if full.days > 40:
        report["scrutiny"] = scrutinise(spec, features, close, benchmark, start, full, base, volume)
    report["gates"] = gates(report, th)
    report["verdict"] = "validated" if all(g["passed"] for g in report["gates"]) else "rejected"
    report["refutation"] = refute(report, capital)
    return report
