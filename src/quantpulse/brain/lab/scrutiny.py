"""Scrutiny: looking on purpose for the reasons a strategy might not work.

Validation (:mod:`.validation`) asks whether the out-of-sample record is better than luck. Scrutiny asks
how it could still be an illusion or unusable:

* **costs** — the break-even cost: the cost per unit traded at which the strategy stops beating the
  equal-weight universe (its gross active return divided by its turnover), and its Sharpe at 1×, 2× and 4×
  the assumed costs. A strategy whose edge disappears at a realistic spread is not an edge;
* **capacity** — the portfolio size at which a rebalance would trade more than
  ``PARTICIPATION`` of a holding's average daily dollar volume (the conservative 10th percentile across
  rebalances). Beyond it, fills move prices and the backtest is fiction;
* **regimes** — active return in rising/falling and calm/volatile markets (the benchmark's trailing
  63-session return and 21-session volatility, known at the time): an edge from one kind of market only;
* **drawdowns** — the deepest fall, how long it took, whether it recovered, and the time spent under water;
* **concentration** — how much of the gain came from the five best names (a few lucky holdings);
* **sensitivity** — the same rule with nearby parameters (half and 1.5× the names, half and double the
  rebalance interval, a day's more lag): a real effect should not vanish when a parameter moves a little.

:func:`refute` turns all of it (and every failed gate) into plain reasons it may not work. Two findings
are gates as well (``robust to nearby parameters``, ``capacity covers the paper book``), plus the
break-even cost against the assumed one.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from quantpulse.quant.risk import max_drawdown

from .backtest import BacktestResult, backtest, equal_weight, metrics
from .spec import StrategySpec

ANNUAL = 252
PARTICIPATION = 0.05  # at most this share of a name's average daily dollar volume per rebalance


def _active(a: pd.Series, b: pd.Series) -> pd.Series:
    joined = pd.concat([a, b], axis=1, join="inner").fillna(0.0)
    return joined.iloc[:, 0] - joined.iloc[:, 1]


def _sharpe(r: pd.Series) -> float | None:
    if r.size < 20 or float(r.std(ddof=1)) == 0:
        return None
    return round(float(r.mean() / r.std(ddof=1) * math.sqrt(ANNUAL)), 4)


def costs(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    start: int,
) -> dict[str, Any]:
    free = replace(spec, cost_bps=0.0)
    gross = backtest(free, features, close, benchmark, start)
    base = equal_weight(close, benchmark, free, start, len(close))
    active = _active(gross.returns, base.returns)
    years = max(gross.days / ANNUAL, 1e-9)
    turnover = float(sum(gross.turnover)) / years
    edge = float(active.mean()) * ANNUAL if active.size else 0.0
    sharpe = {
        f"{m}x": metrics(backtest(spec, features, close, benchmark, start, cost_multiplier=m)).get("sharpe")
        for m in (1, 2, 4)
    }
    return {
        "assumed_cost_bps": spec.cost_bps,
        "gross_active_annual": round(edge, 5),
        "annual_turnover": round(turnover, 3),
        "break_even_bps": round(edge / turnover * 10_000, 2) if turnover > 0 and edge > 0 else 0.0,
        "sharpe_by_cost": sharpe,
    }


def capacity(result: BacktestResult, close: pd.DataFrame, volume: pd.DataFrame | None) -> dict[str, Any]:
    if volume is None or not result.holdings:
        return {"capacity_usd": None, "note": "no volume data"}
    adv = (close * volume.reindex_like(close)).rolling(20, min_periods=10).mean()
    caps: list[float] = []
    for when, names in result.holdings:
        if not names or when not in adv.index:
            continue
        row = adv.loc[when].reindex(names)
        if row.isna().any() or (row <= 0).any():
            continue
        weight = 1.0 / len(names)
        caps.append(float(row.min()) * PARTICIPATION / weight)
    if not caps:
        return {"capacity_usd": None, "note": "no rebalance with volume for every holding"}
    return {
        "capacity_usd": round(float(np.percentile(caps, 10)), 0),
        "median_usd": round(float(np.median(caps)), 0),
        "participation": PARTICIPATION,
        "rebalances": len(caps),
    }


def regimes(returns: pd.Series, active: pd.Series, benchmark: pd.Series) -> dict[str, Any]:
    bench = benchmark.pct_change(fill_method=None)
    trend = benchmark.pct_change(63, fill_method=None).shift(1)  # known the day before
    vol = bench.rolling(21).std().shift(1) * math.sqrt(ANNUAL)
    label = pd.Series("unknown", index=benchmark.index)
    calm = vol <= vol.median()
    label[(trend >= 0) & calm] = "rising, calm"
    label[(trend >= 0) & ~calm] = "rising, volatile"
    label[(trend < 0) & calm] = "falling, calm"
    label[(trend < 0) & ~calm] = "falling, volatile"
    label = label.reindex(active.index)
    total = float(active.sum())
    out: dict[str, Any] = {}
    for name in ("rising, calm", "rising, volatile", "falling, calm", "falling, volatile"):
        a = active[label == name]
        if a.size < 20:
            continue
        out[name] = {
            "days": int(a.size),
            "active_annual": round(float(a.mean()) * ANNUAL, 5),
            "active_sharpe": _sharpe(a),
            "share_of_active_return": round(float(a.sum()) / total, 3) if total else None,
        }
    return out


def drawdowns(returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, Any]:
    if returns.size < 2:
        return {}
    wealth = (1 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    trough = dd.idxmin()
    start = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[start]]
    rec_date = recovered.index[0] if len(recovered) else None
    pos = {d: i for i, d in enumerate(wealth.index)}
    return {
        "max_drawdown": round(float(dd.min()), 4),
        "benchmark_max_drawdown": round(max_drawdown(benchmark_returns.fillna(0.0).to_numpy()), 4),
        "peak": str(start.date()),
        "trough": str(trough.date()),
        "fall_sessions": pos[trough] - pos[start],
        "recovery_sessions": (pos[rec_date] - pos[trough]) if rec_date is not None else None,
        "recovered": rec_date is not None,
        "time_under_water": round(float((dd < -0.001).mean()), 3),
    }


def concentration(result: BacktestResult, close: pd.DataFrame, spec: StrategySpec) -> dict[str, Any]:
    if not result.holdings:
        return {}
    pos = {d: i for i, d in enumerate(close.index)}
    contrib: dict[str, float] = {}
    rows = [(pos[d] + spec.lag_days, names) for d, names in result.holdings if d in pos]
    for (t, names), nxt in zip(rows, [*rows[1:], (len(close) - 1, [])], strict=True):
        end = min(nxt[0], len(close) - 1)
        if end <= t or not names:
            continue
        for n in names:
            a, b = close[n].iloc[t], close[n].iloc[end]
            if a == a and b == b and a > 0:
                contrib[n] = contrib.get(n, 0.0) + (b / a - 1) / len(names)
    positive = sorted((v for v in contrib.values() if v > 0), reverse=True)
    total = sum(positive)
    return {
        "names_held": len(contrib),
        "names_per_rebalance": round(float(np.mean([len(n) for _, n in result.holdings])), 1),
        "top5_share_of_gains": round(float(sum(positive[:5]) / total), 3) if total > 0 else None,
        "best": sorted(contrib, key=lambda n: -contrib[n])[:5],
    }


def sensitivity(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    start: int,
) -> dict[str, Any]:
    def active_sharpe(s: StrategySpec) -> float | None:
        res = backtest(s, features, close, benchmark, start)
        base = equal_weight(close, benchmark, s, start, len(close))
        return _sharpe(_active(res.returns, base.returns))

    neighbours = {
        "half the names": replace(spec, top_n=max(3, spec.top_n // 2)),
        "1.5x the names": replace(spec, top_n=max(spec.top_n + 1, int(spec.top_n * 1.5))),
        "rebalance twice as often": replace(spec, rebalance_days=max(1, spec.rebalance_days // 2)),
        "rebalance half as often": replace(spec, rebalance_days=spec.rebalance_days * 2),
        "one more session of lag": replace(spec, lag_days=spec.lag_days + 1),
    }
    base = active_sharpe(spec)
    results = {name: active_sharpe(s) for name, s in neighbours.items() if s != spec}
    measured = [v for v in results.values() if v is not None]
    return {
        "base_active_sharpe": base,
        "neighbours": results,
        "positive_share": round(sum(1 for v in measured if v > 0) / len(measured), 3) if measured else None,
        "median": round(float(np.median(measured)), 4) if measured else None,
    }


def scrutinise(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    start: int,
    full: BacktestResult,
    base: BacktestResult,
    volume: pd.DataFrame | None,
) -> dict[str, Any]:
    active = _active(full.returns, base.returns)
    return {
        "costs": costs(spec, features, close, benchmark, start),
        "capacity": capacity(full, close, volume),
        "regimes": regimes(full.returns, active, benchmark),
        "drawdowns": drawdowns(full.returns, full.benchmark),
        "concentration": concentration(full, close, spec),
        "sensitivity": sensitivity(spec, features, close, benchmark, start),
    }


def refute(report: dict[str, Any], capital: float) -> list[str]:
    """Every reason found that the strategy may not work (failed gates first)."""
    out = [f"failed: {g['gate']} ({g['detail']})" for g in report.get("gates", []) if not g["passed"]]
    sc = report.get("scrutiny") or {}
    c = sc.get("costs") or {}
    if c and c.get("break_even_bps", 0) < 2 * max(c.get("assumed_cost_bps") or 0, 1):
        out.append(
            f"thin edge: it stops beating equal weight at {c.get('break_even_bps')}bp per unit traded "
            f"(assumed {c.get('assumed_cost_bps')}bp, turnover {c.get('annual_turnover')}× a year)"
        )
    cap = (sc.get("capacity") or {}).get("capacity_usd")
    if cap is not None and cap < capital:
        out.append(f"capacity ${cap:,.0f} is below the paper book's ${capital:,.0f}")
    regimes_ = sc.get("regimes") or {}
    losing = [k for k, v in regimes_.items() if (v.get("active_annual") or 0) < 0]
    if losing:
        out.append("loses to equal weight in " + ", ".join(losing) + " markets")
    carried = [k for k, v in regimes_.items() if (v.get("share_of_active_return") or 0) > 0.8]
    if carried:
        out.append(f"most of its edge comes from {carried[0]} markets only")
    d = sc.get("drawdowns") or {}
    if d and not d.get("recovered"):
        out.append(f"its deepest drawdown ({d.get('max_drawdown'):.0%}) has not recovered")
    if (
        d
        and d.get("max_drawdown") is not None
        and d["max_drawdown"] < 1.5 * (d.get("benchmark_max_drawdown") or 0)
    ):
        out.append("falls much further than the benchmark at its worst")
    conc = sc.get("concentration") or {}
    if (conc.get("top5_share_of_gains") or 0) > 0.6:
        out.append(
            f"{conc['top5_share_of_gains']:.0%} of its gains came from five names ({', '.join(conc['best'])})"
        )
    s = sc.get("sensitivity") or {}
    if s.get("positive_share") is not None and s["positive_share"] < 0.6:
        weak = [k for k, v in (s.get("neighbours") or {}).items() if v is None or v <= 0]
        out.append("fragile: nearby parameters do not work (" + ", ".join(weak) + ")")
    return out
