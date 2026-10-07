"""Point-in-time backtests and the baselines every strategy is compared with.

On each rebalance date *t* the strategy ranks the names with a valid price on the cross-sectional
z-scores of its features *as of t* (the feature panels only use data up to each date), keeps those that
pass its filters, and targets the ``top_n`` best — equal-weighted or inverse-volatility weighted. The
trade happens at the close ``lag_days`` sessions later (a signal at the close is traded no earlier than
the next session's close) and the portfolio earns returns from the session after the trade; it drifts
with prices until the next rebalance, and every rebalance pays ``cost_bps`` on the turnover. A held name with a missing return (a gap or a delisting) earns nothing that
day. Returns are daily and net of costs.

Baselines: the benchmark, an equal-weighted portfolio of the whole universe on the same schedule and
costs, and portfolios of ``top_n`` random names (the distribution a strategy's Sharpe must beat to show
it is more than luck).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantpulse.domain.features import cross_sectional_z
from quantpulse.quant.risk import max_drawdown, sharpe_ratio, sortino_ratio

from .spec import StrategySpec

ANNUAL = 252


@dataclass
class BacktestResult:
    returns: pd.Series  # daily, net of costs
    benchmark: pd.Series  # the benchmark's daily returns over the same days
    holdings: list[tuple[pd.Timestamp, list[str]]] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)

    @property
    def days(self) -> int:
        return int(self.returns.size)


def scores(spec: StrategySpec, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The strategy's ranking score per date and symbol (NaN where it cannot rank or a filter fails)."""
    total: pd.DataFrame | None = None
    weight = pd.DataFrame(
        0.0, index=next(iter(features.values())).index, columns=next(iter(features.values())).columns
    )
    for feat, w in spec.signal.items():
        z = cross_sectional_z(features[feat])
        part = z * w
        total = part.fillna(0.0) if total is None else total + part.fillna(0.0)
        weight = weight + z.notna() * abs(w)
    assert total is not None
    out = total.where(weight > 0)
    for feat, minimum in spec.filters.items():
        out = out.where(features[feat] > minimum)
    return out


def portfolio_weights(names: list[str], vol: pd.Series | None) -> dict[str, float]:
    """Equal weights, or inverse-volatility weights when ``vol`` (one row of volatilities) is given."""
    if not names:
        return {}
    if vol is None:
        return dict.fromkeys(names, 1.0 / len(names))
    inv = {n: 1.0 / v for n, v in vol.reindex(names).items() if v == v and v > 0}
    if not inv:
        return dict.fromkeys(names, 1.0 / len(names))
    total = sum(inv.values())
    return {n: w / total for n, w in inv.items()}


def simulate(
    targets: list[tuple[int, dict[str, float]]],
    rets: pd.DataFrame,
    cost_bps: float,
    lag: int,
    start: int,
    end: int,
) -> tuple[pd.Series, list[float]]:
    """Daily returns of a portfolio that trades to each target at the close ``lag`` sessions after its
    signal row — so the first return it earns is the session after that trade — and drifts in between;
    ``targets`` are (row index of the signal, weights). Row ``d`` of ``rets`` is close ``d-1`` → close ``d``."""
    values = rets.to_numpy(dtype=float)
    cols = {c: i for i, c in enumerate(rets.columns)}
    out = np.zeros(end - start)
    turnover: list[float] = []
    current: dict[str, float] = {}
    schedule = {t + lag + 1: w for t, w in targets if start <= t + lag + 1 < end}  # first return row earned
    first = min(schedule) if schedule else end
    for day in range(max(start, first), end):
        if day in schedule:
            target = schedule[day]
            names = set(current) | set(target)
            traded = sum(abs(target.get(n, 0.0) - current.get(n, 0.0)) for n in names)
            turnover.append(traded)
            current = dict(target)
            out[day - start] -= traded * cost_bps / 1e4
        if not current:
            continue
        r = np.array([values[day, cols[n]] if n in cols else np.nan for n in current])
        r = np.where(np.isfinite(r), r, 0.0)
        w = np.array(list(current.values()))
        gross = float(np.dot(w, r))
        out[day - start] += gross
        grown = w * (1 + r)
        total = grown.sum()
        if total > 0:
            current = dict(zip(current, grown / total, strict=True))
    idx = rets.index[start:end]
    series = pd.Series(out, index=idx)
    return series.iloc[max(0, first - start) :], turnover


def backtest(
    spec: StrategySpec,
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    benchmark: pd.Series,
    start: int | None = None,
    end: int | None = None,
    *,
    cost_multiplier: float = 1.0,
    extra_lag: int = 0,
    score: pd.DataFrame | None = None,
) -> BacktestResult:
    rets = close.pct_change(fill_method=None)
    bench = benchmark.pct_change(fill_method=None)
    s = scores(spec, features) if score is None else score
    n = len(close)
    start = 0 if start is None else start
    end = n if end is None else end
    vol = features.get("vol_63") if spec.weighting == "inverse_vol" else None
    targets: list[tuple[int, dict[str, float]]] = []
    holdings: list[tuple[pd.Timestamp, list[str]]] = []
    valid = s.notna().sum(axis=1)
    rows = [i for i in range(start, end) if valid.iloc[i] >= spec.top_n]
    t = rows[0] if rows else end
    while t < end - spec.lag_days - extra_lag - 1:
        row = s.iloc[t].dropna()
        row = row[close.iloc[t].reindex(row.index).notna()]
        if len(row) >= spec.top_n:
            names = [str(x) for x in row.sort_values(ascending=False).index[: spec.top_n]]
            targets.append((t, portfolio_weights(names, vol.iloc[t] if vol is not None else None)))
            holdings.append((close.index[t], names))
        t += spec.rebalance_days
    daily, turnover = simulate(
        targets, rets, spec.cost_bps * cost_multiplier, spec.lag_days + extra_lag, start, end
    )
    return BacktestResult(daily, bench.reindex(daily.index).fillna(0.0), holdings, turnover)


def equal_weight(
    close: pd.DataFrame, benchmark: pd.Series, spec: StrategySpec, start: int, end: int
) -> BacktestResult:
    """The whole universe, equally weighted, on the strategy's schedule and costs."""
    rets = close.pct_change(fill_method=None)
    targets = []
    for t in range(start, end, spec.rebalance_days):
        names = [str(c) for c in close.columns[close.iloc[t].notna()]]
        if names:
            targets.append((t, dict.fromkeys(names, 1.0 / len(names))))
    daily, turnover = simulate(targets, rets, spec.cost_bps, spec.lag_days, start, end)
    bench = benchmark.pct_change(fill_method=None)
    return BacktestResult(daily, bench.reindex(daily.index).fillna(0.0), [], turnover)


def random_sharpes(
    close: pd.DataFrame, spec: StrategySpec, start: int, end: int, draws: int = 100, seed: int = 7
) -> list[float]:
    """Sharpe ratios of ``draws`` portfolios of ``top_n`` random names on the strategy's schedule."""
    rng = np.random.default_rng(seed)
    rets = close.pct_change(fill_method=None)
    out: list[float] = []
    for _ in range(draws):
        targets = []
        for t in range(start, end, spec.rebalance_days):
            names = [str(c) for c in close.columns[close.iloc[t].notna()]]
            if len(names) >= spec.top_n:
                pick = list(rng.choice(names, spec.top_n, replace=False))
                targets.append((t, dict.fromkeys(pick, 1.0 / spec.top_n)))
        daily, _ = simulate(targets, rets, spec.cost_bps, spec.lag_days, start, end)
        sr = sharpe_ratio(daily.to_numpy()) if daily.size > 20 else None
        if sr is not None:
            out.append(sr)
    return out


def metrics(result: BacktestResult) -> dict[str, Any]:
    r = result.returns.to_numpy(dtype=float)
    b = result.benchmark.to_numpy(dtype=float)
    if r.size < 20:
        return {"days": int(r.size), "insufficient": True}
    years = r.size / ANNUAL
    total = float(np.prod(1 + r) - 1)
    bench_total = float(np.prod(1 + b) - 1)
    excess = r - b
    te = float(excess.std(ddof=1)) * math.sqrt(ANNUAL)
    mdd = max_drawdown(r)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    var_b = float(np.var(b, ddof=1))
    return {
        "days": int(r.size),
        "start": str(result.returns.index[0].date()),
        "end": str(result.returns.index[-1].date()),
        "total_return": round(total, 5),
        "cagr": round(cagr, 5),
        "volatility": round(float(r.std(ddof=1)) * math.sqrt(ANNUAL), 5),
        "sharpe": _r(sharpe_ratio(r)),
        "sortino": _r(sortino_ratio(r)),
        "max_drawdown": round(mdd, 5),
        "calmar": round(cagr / abs(mdd), 4) if mdd < 0 else None,
        "benchmark_total_return": round(bench_total, 5),
        "benchmark_sharpe": _r(sharpe_ratio(b)),
        "excess_annual": round(float(excess.mean()) * ANNUAL, 5),
        "information_ratio": round(float(excess.mean()) * ANNUAL / te, 4) if te > 0 else None,
        "beta": round(float(np.cov(r, b, ddof=1)[0, 1]) / var_b, 4) if var_b > 0 else None,
        "hit_rate_daily": round(float((r > 0).mean()), 4),
        "rebalances": len(result.turnover),
        "avg_turnover": round(float(np.mean(result.turnover)), 4) if result.turnover else 0.0,
        "skew": round(float(pd.Series(r).skew()), 4),
        "kurtosis": round(float(pd.Series(r).kurt()) + 3.0, 4),  # not excess
    }


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None
