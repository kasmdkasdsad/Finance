"""Deterministic market intelligence: indicators and statistics computed once per cycle, for every symbol at
once, and shared by all agents (no agent recomputes them, and none of this ever calls a language model).

:func:`compute_indicators` extends the strategy's :func:`~quantpulse.domain.trading_signals.raw_signals`
(momentum, trend, ADX, RSI, VWAP, volume, ATR, volatility, beta, …) with MACD, price z-scores, 20-day
ranges (breakouts, support and resistance), correlation to the benchmark, momentum acceleration and
return-distribution statistics. :func:`market_statistics` summarises the cross-section (breadth,
dispersion, average correlation) for the regime and portfolio agents.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from quantpulse.domain import trading_signals as ts

ANNUAL = math.sqrt(252)


def _last(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[-1] if len(frame) else pd.Series(dtype=float)


def _ema(frame: pd.DataFrame, span: int) -> pd.DataFrame:
    return frame.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_indicators(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series,
    *,
    live_row: bool = False,
    vwap: Mapping[str, float] | None = None,
    session_fraction: float | None = None,
    implied_vol: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """One row per symbol, one column per indicator, on the last row of the panel (which is today's
    partial session when ``live_row``)."""
    if close.empty or len(close) < ts.MIN_HISTORY:
        return pd.DataFrame(index=close.columns)
    base = ts.raw_signals(
        close,
        high,
        low,
        volume,
        benchmark,
        live_row=live_row,
        vwap=vwap,
        session_fraction=session_fraction,
        implied_vol=implied_vol,
    )
    c = close.astype(float)
    ret = c.pct_change(fill_method=None)
    bench = benchmark.reindex(c.index).astype(float).ffill()
    bench_ret = bench.pct_change(fill_method=None)
    out = pd.DataFrame(index=c.columns)

    out["ret_1d"] = _last(ret)
    out["ret_5d"] = _last(c / c.shift(5) - 1)
    out["ret_21d"] = _last(c / c.shift(21) - 1)
    daily_vol = ret.iloc[-64:-1].std()
    out["move_z"] = out["ret_1d"] / daily_vol.where(daily_vol > 0)  # today's move in daily sigmas
    out["ret_5d_z"] = out["ret_5d"] / (daily_vol * math.sqrt(5)).where(daily_vol > 0)

    macd = _ema(c, 12) - _ema(c, 26)
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    price = _last(c)
    out["macd"] = _last(macd) / price
    out["macd_hist"] = _last(macd - signal) / price
    prev_hist = (macd - signal).iloc[-2] if len(c) > 1 else pd.Series(np.nan, index=c.columns)
    out["macd_cross"] = np.sign(_last(macd - signal)) - np.sign(prev_hist)  # +2 bullish cross, −2 bearish

    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["z20"] = _last((c - sma20) / sd20.where(sd20 > 0))
    out["px_vs_sma20"] = _last(c / sma20 - 1)
    out["px_vs_sma200"] = _last(c / c.rolling(200).mean() - 1)
    out["bb_width"] = _last(4 * sd20 / sma20)
    out["bb_width_pct"] = (
        (4 * sd20 / sma20).iloc[-252:].rank(pct=True).iloc[-1]
        if len(c) >= 60
        else pd.Series(np.nan, index=c.columns)
    )

    prior = slice(-21, -1)  # the 20 sessions before the last one
    hi20 = high.astype(float).iloc[prior].max()
    lo20 = low.astype(float).iloc[prior].min()
    atr = base["atr_pct"] * price
    out["high_20"], out["low_20"] = hi20, lo20
    out["breakout_20"] = price > hi20
    out["breakdown_20"] = price < lo20
    out["dist_resistance_atr"] = (hi20 - price) / atr.where(atr > 0)
    out["dist_support_atr"] = (price - lo20) / atr.where(atr > 0)

    vol = volume.astype(float).where(volume > 0)
    done = vol.iloc[:-1] if live_row else vol
    avg20 = done.iloc[-20:].mean()
    out["volume_ratio_1d"] = (done.iloc[-1] / avg20.where(avg20 > 0)) if len(done) else np.nan

    tail20, tail120 = ret.iloc[-20:], ret.iloc[-120:]
    b20, b120 = bench_ret.iloc[-20:], bench_ret.iloc[-120:]
    out["corr_spy_20"] = tail20.apply(lambda s: s.corr(b20))
    out["corr_spy_120"] = tail120.apply(lambda s: s.corr(b120))
    out["rs_1m"] = out["ret_21d"] - float(bench.iloc[-1] / bench.iloc[-22] - 1) if len(bench) > 22 else np.nan
    out["mom_accel"] = out["ret_21d"] - base["mom_3m"] / 3.0  # last month vs the 3-month average pace
    out["skew_63"] = ret.iloc[-63:].skew()
    out["kurt_63"] = ret.iloc[-63:].kurt()
    out["drawdown_252"] = _last(c / c.rolling(252, min_periods=60).max() - 1)

    merged = base.join(out, how="left", rsuffix="_x")
    return merged.replace([np.inf, -np.inf], np.nan)


def market_statistics(close: pd.DataFrame, benchmark: pd.Series) -> dict[str, float | None]:
    """Cross-sectional state: breadth, dispersion and average pairwise correlation (20 vs 120 days)."""
    c = close.astype(float)
    if c.shape[1] < 3 or len(c) < 130:
        return {}
    ret = c.pct_change(fill_method=None)
    sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    last = c.iloc[-1]

    def avg_corr(window: pd.DataFrame) -> float | None:
        m = window.dropna(axis=1, thresh=int(len(window) * 0.8)).corr().to_numpy()
        n = m.shape[0]
        if n < 3:
            return None
        return float((m.sum() - np.trace(m)) / (n * (n - 1)))

    r1m = c.iloc[-1] / c.iloc[-22] - 1
    bret = benchmark.astype(float).pct_change(fill_method=None)
    out: dict[str, float | None] = {
        "breadth_50": float((last > sma50).mean()),
        "breadth_200": float((last > sma200).mean()),
        "dispersion_1m": float(r1m.std()),
        "avg_corr_20": avg_corr(ret.iloc[-20:]),
        "avg_corr_120": avg_corr(ret.iloc[-120:]),
        "benchmark_rv21": float(bret.iloc[-21:].std() * ANNUAL),
        "benchmark_rv252": float(bret.iloc[-252:].std() * ANNUAL) if len(bret) > 252 else None,
        "advancers_1d": float((ret.iloc[-1] > 0).mean()),
    }
    return {k: (v if v is None or math.isfinite(v) else None) for k, v in out.items()}
