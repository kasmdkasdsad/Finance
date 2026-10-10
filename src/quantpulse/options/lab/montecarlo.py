"""Monte Carlo on a strategy's trades: how bad could it have been, and how often?

From the backtest's trades (P&L and maximum loss each), thousands of alternative histories are drawn:

* **bootstrap** — trades resampled with replacement (the same number as observed);
* **order** — the observed trades in random order (the same total, different drawdowns);
* **volatility shock** — every loss 50% larger;
* **spread widening** — every trade pays its spread cost again;
* **slippage shock** — every trade loses a further 1% of its maximum loss;
* **gap shock** — 5% of trades hit their maximum loss outright;
* **delayed exits** — every trade gives back a further fifth of its time decay (or loses a fifth more);
* **missing fills** — 10% of trades never happen (winners as likely as losers).

For each: the distribution of the final P&L, of the maximum drawdown, the risk of ruin (equity below half its
start), Sharpe-, Sortino- and Calmar-like ratios, the profit factor, the win rate, average win and loss, and
the skew and kurtosis of trade results. No single ratio decides anything: the whole distribution is kept.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

SCENARIOS = ("bootstrap", "order", "volatility_shock", "spread_widening", "slippage_shock", "gap_shock",
             "delayed_exits", "missing_fills")  # fmt: skip


def _paths(pnl: np.ndarray, equity0: float) -> dict[str, np.ndarray]:
    """``pnl``: (paths, trades). Equity paths, their max drawdowns and final values."""
    eq = equity0 + np.cumsum(pnl, axis=1)
    eq = np.concatenate([np.full((pnl.shape[0], 1), equity0), eq], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq / peak - 1).min(axis=1)
    return {"final": eq[:, -1] - equity0, "max_dd": dd, "min_equity": eq.min(axis=1)}


def _stats(pnl: np.ndarray, equity0: float, trades_per_year: float) -> dict[str, Any]:
    p = _paths(pnl, equity0)
    means, sds = pnl.mean(axis=1), pnl.std(axis=1, ddof=1) if pnl.shape[1] > 1 else np.zeros(pnl.shape[0])
    down = np.sqrt(np.mean(np.minimum(pnl, 0) ** 2, axis=1))
    scale = math.sqrt(max(trades_per_year, 1e-9))
    sharpe = np.where(sds > 0, means / np.where(sds > 0, sds, 1) * scale, np.nan)
    sortino = np.where(down > 0, means / np.where(down > 0, down, 1) * scale, np.nan)
    years = pnl.shape[1] / max(trades_per_year, 1e-9)
    ann = p["final"] / equity0 / max(years, 1e-9)
    calmar = np.where(p["max_dd"] < 0, ann / np.abs(np.where(p["max_dd"] < 0, p["max_dd"], -1)), np.nan)
    wins = pnl > 0
    gw = np.where(wins, pnl, 0).sum(axis=1)
    gl = -np.where(~wins, pnl, 0).sum(axis=1)
    pf = np.where(gl > 0, gw / np.where(gl > 0, gl, 1), np.nan)

    def q(x: np.ndarray) -> dict[str, float | None]:
        x = x[np.isfinite(x)]
        if not len(x):
            return {"p5": None, "p50": None, "p95": None}
        return {k: round(float(np.percentile(x, v)), 4) for k, v in (("p5", 5), ("p50", 50), ("p95", 95))}

    flat = pnl.ravel()
    sd = flat.std()
    return {
        "final_pnl": q(p["final"]),
        "return": q(p["final"] / equity0),
        "max_drawdown": q(p["max_dd"]),
        "risk_of_ruin": round(float((p["min_equity"] < equity0 * 0.5).mean()), 4),
        "prob_loss": round(float((p["final"] < 0).mean()), 4),
        "sharpe": q(sharpe),
        "sortino": q(sortino),
        "calmar": q(calmar),
        "profit_factor": q(pf),
        "win_rate": round(float(wins.mean()), 4),
        "avg_win": round(float(flat[flat > 0].mean()), 2) if (flat > 0).any() else None,
        "avg_loss": round(float(flat[flat <= 0].mean()), 2) if (flat <= 0).any() else None,
        "skew": round(float(((flat - flat.mean()) ** 3).mean() / sd**3), 4) if sd > 0 else None,
        "kurtosis": round(float(((flat - flat.mean()) ** 4).mean() / sd**4 - 3), 4) if sd > 0 else None,
    }


def simulate(trades: Sequence[dict[str, Any]], *, equity: float = 100_000.0, paths: int = 2000, seed: int = 5,
             trades_per_year: float | None = None) -> dict[str, Any]:  # fmt: skip
    """Every scenario's distribution. Needs at least five trades (fewer is not a distribution)."""
    if len(trades) < 5:
        return {"trades": len(trades), "scenarios": {}, "note": "too few trades for a Monte Carlo"}
    rng = np.random.default_rng(seed)
    pnl = np.array([float(t["pnl"]) for t in trades])
    risk = np.array([float(t.get("max_loss") or abs(t["pnl"]) or 1.0) for t in trades])
    spread = np.array([float(t.get("spread_cost") or 0.0) for t in trades])
    theta = np.array([float((t.get("attribution") or {}).get("theta") or 0.0) for t in trades])
    n = len(pnl)
    if trades_per_year is None:
        days = [t.get("days_held") or 20 for t in trades]
        trades_per_year = max(1.0, 252 / max(float(np.mean(days)), 1.0))
    idx = rng.integers(0, n, size=(paths, n))
    out: dict[str, Any] = {}
    base = pnl[idx]
    out["bootstrap"] = _stats(base, equity, trades_per_year)
    order = np.array([rng.permutation(pnl) for _ in range(min(paths, 1000))])
    out["order"] = _stats(order, equity, trades_per_year)
    out["volatility_shock"] = _stats(np.where(base < 0, base * 1.5, base), equity, trades_per_year)
    out["spread_widening"] = _stats(base - spread[idx], equity, trades_per_year)
    out["slippage_shock"] = _stats(base - 0.01 * risk[idx], equity, trades_per_year)
    gap = rng.random(size=base.shape) < 0.05
    out["gap_shock"] = _stats(np.where(gap, -risk[idx], base), equity, trades_per_year)
    out["delayed_exits"] = _stats(base - 0.2 * np.abs(theta[idx]), equity, trades_per_year)
    keep = rng.random(size=base.shape) >= 0.10
    out["missing_fills"] = _stats(np.where(keep, base, 0.0), equity, trades_per_year)
    worst_ruin = max(s["risk_of_ruin"] for s in out.values())
    worst_dd = min((s["max_drawdown"]["p5"] or 0) for s in out.values())
    return {
        "trades": n,
        "paths": paths,
        "trades_per_year": round(trades_per_year, 2),
        "scenarios": out,
        "worst_risk_of_ruin": worst_ruin,
        "worst_p5_drawdown": worst_dd,
        "bootstrap_p5_positive": (out["bootstrap"]["final_pnl"]["p5"] or 0) > 0,
    }
