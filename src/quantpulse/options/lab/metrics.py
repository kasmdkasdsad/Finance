"""Performance metrics — many, because one number always hides something.

From the trades: count, win and loss rates, average win and loss, payoff ratio, expectancy (per trade and per
dollar at risk), profit factor, the largest loss, skew and kurtosis of trade returns. From the daily equity
curve: total and annualized return, volatility, Sharpe, Sortino, Calmar, Omega, value at risk and expected
shortfall (95%), maximum drawdown and its length, the Ulcer index. From the attribution: the P&L from delta,
gamma, theta and vega, the cost of spreads and fees, turnover.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

TRADING_DAYS = 252


def _r(x: float | None, n: int = 4) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), n)


def trade_stats(pnls: Sequence[float], risks: Sequence[float] | None = None) -> dict[str, Any]:
    p = np.asarray(pnls, dtype=float)
    n = len(p)
    if n == 0:
        return {"trades": 0}
    wins, losses = p[p > 0], p[p <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    out: dict[str, Any] = {
        "trades": n,
        "win_rate": _r(len(wins) / n),
        "loss_rate": _r(len(losses) / n),
        "avg_win": _r(float(wins.mean()) if len(wins) else 0.0, 2),
        "avg_loss": _r(float(losses.mean()) if len(losses) else 0.0, 2),
        "payoff_ratio": _r(
            float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else None
        ),
        "expectancy": _r(float(p.mean()), 2),
        "profit_factor": _r(
            gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else None)
        ),
        "total_pnl": _r(float(p.sum()), 2),
        "largest_loss": _r(float(p.min()), 2),
        "largest_win": _r(float(p.max()), 2),
        "pnl_sd": _r(float(p.std(ddof=1)) if n > 1 else None, 2),
    }
    if risks is not None and len(risks) == n:
        r = np.asarray(risks, dtype=float)
        ror = np.where(r > 0, p / np.where(r > 0, r, 1), np.nan)
        ror = ror[np.isfinite(ror)]
        if len(ror):
            out["expectancy_on_risk"] = _r(float(ror.mean()))
            out["return_on_risk_sd"] = _r(float(ror.std(ddof=1)) if len(ror) > 1 else None)
            out["trade_skew"] = _r(_skew(ror))
            out["trade_kurtosis"] = _r(_kurt(ror))
            out["t_stat"] = _r(
                float(ror.mean() / (ror.std(ddof=1) / math.sqrt(len(ror))))
                if len(ror) > 2 and ror.std(ddof=1) > 0
                else None
            )
    return out


def _skew(x: np.ndarray) -> float | None:
    if len(x) < 3 or x.std() == 0:
        return None
    return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)


def _kurt(x: np.ndarray) -> float | None:
    if len(x) < 4 or x.std() == 0:
        return None
    return float(((x - x.mean()) ** 4).mean() / x.std() ** 4 - 3)


def drawdowns(equity: Sequence[float]) -> dict[str, Any]:
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return {"max_drawdown": None, "max_drawdown_days": None, "ulcer_index": None}
    peak = np.maximum.accumulate(e)
    dd = np.where(peak > 0, e / peak - 1, 0.0)
    longest, cur = 0, 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        longest = max(longest, cur)
    return {
        "max_drawdown": _r(float(dd.min())),
        "max_drawdown_days": longest,
        "ulcer_index": _r(float(math.sqrt((dd**2 * 10000).mean()))),
    }


def curve_stats(equity: Sequence[float]) -> dict[str, Any]:
    e = np.asarray(equity, dtype=float)
    if len(e) < 3 or e[0] <= 0:
        return {}
    r = np.diff(e) / e[:-1]
    years = len(r) / TRADING_DAYS
    total = e[-1] / e[0] - 1
    cagr = (e[-1] / e[0]) ** (1 / years) - 1 if years > 0 and e[-1] > 0 else None
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(r) > 1 else None
    downside = r[r < 0]
    dvol = float(math.sqrt((downside**2).mean()) * math.sqrt(TRADING_DAYS)) if len(downside) else None
    dd = drawdowns(e.tolist())
    mean_ann = float(r.mean() * TRADING_DAYS)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    q = np.sort(r)
    k = max(1, int(len(q) * 0.05))
    return {
        "total_return": _r(total),
        "cagr": _r(cagr),
        "volatility": _r(vol),
        "sharpe": _r(mean_ann / vol if vol else None),
        "sortino": _r(mean_ann / dvol if dvol else None),
        "calmar": _r(cagr / abs(dd["max_drawdown"]) if cagr is not None and dd["max_drawdown"] else None),
        "omega": _r(float(gains / losses) if losses > 0 else None),
        "var_95": _r(float(q[k - 1])),
        "cvar_95": _r(float(q[:k].mean())),
        "daily_skew": _r(_skew(r)),
        "daily_kurtosis": _r(_kurt(r)),
        **dd,
    }


def attribution_totals(trades: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = ("delta", "gamma", "theta", "vega", "residual", "execution", "fees")
    out = dict.fromkeys(keys, 0.0)
    for t in trades:
        a = t.get("attribution") or {}
        for k in keys:
            out[k] += float(a.get(k) or 0.0)
    return {k: round(v, 2) for k, v in out.items()}


def summarize(trades: Sequence[dict[str, Any]], equity: Sequence[float]) -> dict[str, Any]:
    pnls = [t["pnl"] for t in trades]
    risks = [t.get("max_loss") or 0.0 for t in trades]
    out = {**trade_stats(pnls, risks), **curve_stats(equity), "pnl_attribution": attribution_totals(trades)}
    held = [float(t["days_held"]) for t in trades if t.get("days_held") is not None]
    out["avg_days_held"] = _r(float(np.mean(held)), 2) if held else None
    out["turnover"] = len(trades)
    out["avg_dte_entry"] = _r(float(np.mean([t["dte_entry"] for t in trades])), 1) if trades else None
    out["avg_dte_exit"] = _r(float(np.mean([t["dte_exit"] for t in trades])), 1) if trades else None
    out["spread_cost"] = round(sum(float(t.get("spread_cost") or 0) for t in trades), 2)
    out["fees"] = round(sum(float(t.get("fees") or 0) for t in trades), 2)
    return out
