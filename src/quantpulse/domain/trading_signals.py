"""Opportunity scores for the paper-trading strategy.

Every tradable symbol gets seven *component* scores, each a cross-sectional z-score (winsorised at ±3) so
that no component dominates because of its scale. A component averages several sub-signals, each z-scored
first; a symbol missing a sub-signal is scored on the rest, and a symbol missing a whole component gets 0
(neutral) for it.

=============  ==========================================================================================
component      sub-signals (the sign says which direction scores higher)
=============  ==========================================================================================
momentum       +12-1 month return, +6-1 month return, +3-month return, +10-day return (short-term),
               +3-month return relative to SPY (relative strength), +trend persistence (R² of a 63-day
               log-price fit, signed by its slope)
trend          +price vs 50-day average, +50-day vs 200-day average, +closeness to the 52-week high,
               +signed ADX(14) trend strength, +price vs today's VWAP, +setup (the better of a breakout
               above the prior 20-day high and a pullback in an uptrend, RSI(14) < 55)
volume         +20-day vs 120-day average volume, +today's volume vs normal for this time of day
               (relative volume), +up-day vs down-day volume (confirmation), +½ log dollar volume
volatility     −ATR(14) as % of price, −21-day vs 63-day volatility (contraction scores higher),
               +6-month Sharpe ratio, −implied vs realised volatility (event risk), when available
fundamental    +earnings yield, +free-cash-flow yield, +½ book-to-market (value); +gross profitability,
               +ROE, −accruals (quality and margins); −½ asset growth (conservative investment);
               +latest earnings reaction (post-earnings drift). Point-in-time SEC data from the stock model.
model          the walk-forward stock model's live z-score (ridge + gradient-boosted trees)
regime         market-regime beta tilt × beta: in bullish markets high-beta names score higher, in
               bearish and risk-off markets defensive (low-beta) names do
=============  ==========================================================================================

The *opportunity score* is the weighted average of the components (configurable weights), standardised
again across the universe: +1 means one standard deviation better than the average candidate.

Absolute filters complement the relative score: a new position needs an intact trend (price above its
50-day average and a positive 3-month return); a held one is only considered *reversed* when the price is
below both its 20- and 50-day averages with a negative 10-day return (hysteresis against churn).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPONENTS: tuple[str, ...] = ("momentum", "trend", "volume", "volatility", "fundamental", "model", "regime")
LABELS: dict[str, str] = {
    "momentum": "Momentum",
    "trend": "Trend & price structure",
    "volume": "Volume",
    "volatility": "Volatility",
    "fundamental": "Fundamentals & earnings",
    "model": "Stock model",
    "regime": "Regime fit",
}
ANNUAL = math.sqrt(252)
WINSOR = 3.0
ATR_TO_SIGMA = 1.4  # a daily ATR is roughly 1.2-1.6 daily standard deviations
MIN_HISTORY = 64  # bars needed before a symbol is scored at all

# Fundamental features (from the stock model) with their orientation and weight.
FUNDAMENTAL_TERMS: tuple[tuple[str, float], ...] = (
    ("earnings_yield", 1.0),
    ("fcf_yield", 1.0),
    ("book_to_market", 0.5),
    ("gross_profitability", 1.0),
    ("roe", 1.0),
    ("accruals", -1.0),
    ("asset_growth", -0.5),
    ("earn_reaction", 1.0),
)


@dataclass(frozen=True, slots=True)
class LiveBar:
    """Today's session so far, from a live snapshot."""

    price: float
    volume: float | None = None
    vwap: float | None = None
    high: float | None = None
    low: float | None = None
    open: float | None = None


def zscore(values: pd.Series) -> pd.Series:
    """Cross-sectional z-score winsorised at ±3; NaN stays NaN, a constant column scores 0. Fewer than three
    values are not a cross-section: all NaN (the sub-signal is then simply skipped)."""
    v = values.astype(float).replace([np.inf, -np.inf], np.nan)
    present = v.dropna()
    if len(present) < 3:
        return v * np.nan
    sd = float(present.std(ddof=0))
    if not sd > 0:
        return v * 0.0
    return ((v - float(present.mean())) / sd).clip(-WINSOR, WINSOR)


def combine(parts: Sequence[tuple[pd.Series, float]], index: pd.Index) -> pd.Series:
    """Weighted average of z-scored sub-signals (skipping missing ones), standardised again; 0 if absent."""
    num = pd.Series(0.0, index=index)
    den = pd.Series(0.0, index=index)
    for series, weight in parts:
        z = zscore(series.reindex(index))
        present = z.notna()
        num[present] += weight * z[present]
        den[present] += abs(weight)
    mean = num / den.where(den > 0)
    return zscore(mean).fillna(0.0)


def _wilder(frame: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    return frame.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def append_live_row(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    live: Mapping[str, LiveBar],
    day: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add today's partial session as the last row (prices from the live snapshot)."""
    cols = close.columns
    row_close = pd.Series({s: live[s].price for s in cols if s in live}, dtype=float).reindex(cols)
    row_high = pd.Series(
        {s: max(live[s].high or live[s].price, live[s].price) for s in cols if s in live}, dtype=float
    ).reindex(cols)
    row_low = pd.Series(
        {s: min(live[s].low or live[s].price, live[s].price) for s in cols if s in live}, dtype=float
    ).reindex(cols)
    row_vol = pd.Series({s: live[s].volume or 0.0 for s in cols if s in live}, dtype=float).reindex(cols)

    def add(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
        base = frame[frame.index < day]
        return pd.concat([base, pd.DataFrame([row.to_numpy()], index=[day], columns=cols)])

    return add(close, row_close), add(high, row_high), add(low, row_low), add(volume, row_vol)


def raw_signals(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series,
    *,
    live_row: bool,
    vwap: Mapping[str, float] | None = None,
    session_fraction: float | None = None,
    implied_vol: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Raw metrics per symbol on the last row (index = symbol). ``live_row`` marks a partial session:
    volume statistics then use completed days only."""
    c = close.astype(float)
    n = len(c)
    ret = c.pct_change(fill_method=None)
    logret = np.log(c).diff()
    bench = benchmark.reindex(c.index).astype(float).ffill()
    bench_ret = bench.pct_change(fill_method=None)
    last = c.iloc[-1]

    def back(k: int) -> pd.Series:
        return c.iloc[-1 - k] if n > k else pd.Series(np.nan, index=c.columns)

    out = pd.DataFrame(index=c.columns)
    out["price"] = last
    out["mom_12_1"] = back(21) / back(252) - 1
    out["mom_6_1"] = back(21) / back(126) - 1
    out["mom_3m"] = last / back(63) - 1
    out["ret_10d"] = last / back(10) - 1
    bench_3m = float(bench.iloc[-1] / bench.iloc[-64]) if n > 63 else np.nan
    out["rel_strength"] = (last / back(63)) / bench_3m - 1

    # trend persistence: R² of a least-squares line through the last 63 log prices, signed by the slope
    window = np.log(c.iloc[-63:]) if n >= 63 else None
    if window is not None:
        x = np.arange(len(window), dtype=float)
        xc = x - x.mean()
        yc = window - window.mean()
        slope = (yc.mul(xc, axis=0)).sum() / float((xc**2).sum())
        ss_tot = (yc**2).sum()
        r2 = (slope**2 * float((xc**2).sum())) / ss_tot.where(ss_tot > 0)
        out["persistence"] = np.sign(slope) * r2.where(window.notna().all())
    else:
        out["persistence"] = np.nan

    sma20 = c.iloc[-20:].mean() if n >= 20 else pd.Series(np.nan, index=c.columns)
    sma50 = c.iloc[-50:].mean() if n >= 50 else pd.Series(np.nan, index=c.columns)
    sma200 = c.iloc[-200:].mean() if n >= 200 else pd.Series(np.nan, index=c.columns)
    out["sma20"], out["sma50"], out["sma200"] = sma20, sma50, sma200
    out["px_vs_sma50"] = last / sma50 - 1
    out["sma50_vs_sma200"] = sma50 / sma200 - 1
    out["near_high"] = last / c.iloc[-252:].max() - 1
    prior_high = high.iloc[-21:-1].max() if n > 21 else pd.Series(np.nan, index=c.columns)
    out["breakout"] = last / prior_high - 1

    # Wilder's ADX with the direction of the dominant move
    prev_close = c.shift(1)
    tr = (high - low).combine((high - prev_close).abs(), np.fmax).combine((low - prev_close).abs(), np.fmax)
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _wilder(tr)
    plus_di = 100 * _wilder(plus_dm) / atr
    minus_di = 100 * _wilder(minus_dm) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).where((plus_di + minus_di) > 0)
    adx = _wilder(dx)
    out["adx_signed"] = adx.iloc[-1] * np.sign(plus_di.iloc[-1] - minus_di.iloc[-1])
    out["atr_pct"] = atr.iloc[-1] / last

    delta = c.diff()
    gain = _wilder(delta.clip(lower=0))
    loss = _wilder((-delta).clip(lower=0))
    rsi = 100 - 100 / (1 + gain / loss.where(loss > 0))
    rsi = rsi.where(loss > 0, 100.0).where(gain.notna())
    out["rsi14"] = rsi.iloc[-1]
    uptrend = (sma50 > sma200) & (last > sma200)
    out["pullback"] = (55.0 - out["rsi14"]).clip(lower=0).where(uptrend, 0.0)
    out["setup"] = pd.concat([zscore(out["breakout"]), zscore(out["pullback"])], axis=1).max(axis=1)

    vw = pd.Series(vwap or {}, dtype=float).reindex(c.columns)
    out["px_vs_vwap"] = last / vw.where(vw > 0) - 1

    # volume and liquidity on completed sessions only
    done = slice(None, -1) if live_row else slice(None)
    vol_done, close_done = volume.iloc[done], c.iloc[done]
    ret_done = ret.iloc[done]
    avg20 = vol_done.iloc[-20:].mean()
    avg120 = vol_done.iloc[-120:].mean()
    out["volume_trend"] = avg20 / avg120.where(avg120 > 0) - 1
    if live_row and session_fraction is not None and session_fraction >= 0.05:
        today_vol = volume.iloc[-1]
        out["rel_volume"] = today_vol / (avg20 * session_fraction).where(avg20 > 0) - 1
    else:
        out["rel_volume"] = np.nan
    v20, r20 = vol_done.iloc[-20:], ret_done.iloc[-20:]
    up_vol = v20.where(r20 > 0).sum()
    down_vol = v20.where(r20 < 0).sum()
    out["updown_volume"] = np.log(up_vol.where(up_vol > 0) / down_vol.where(down_vol > 0))
    dollar = (close_done * vol_done).iloc[-20:].mean()
    out["adv_dollar"] = dollar
    out["log_dollar_volume"] = np.log(dollar.where(dollar > 0))

    rv21 = logret.iloc[-21:].std() * ANNUAL
    rv63 = logret.iloc[-63:].std() * ANNUAL
    out["rv21"], out["rv63"] = rv21, rv63
    out["vol_ratio"] = rv21 / rv63.where(rv63 > 0)
    r126 = ret.iloc[-126:]
    out["sharpe_126"] = r126.mean() / r126.std().where(r126.std() > 0) * ANNUAL
    iv = pd.Series(implied_vol or {}, dtype=float).reindex(c.columns)
    out["implied_vol"] = iv.where(iv > 0)
    out["iv_premium"] = out["implied_vol"] / rv63.where(rv63 > 0) - 1
    bench_var = bench_ret.iloc[-252:].var()
    out["beta"] = ret.iloc[-252:].apply(lambda col: col.cov(bench_ret.iloc[-252:])) / bench_var

    atr_vol = out["atr_pct"] * ANNUAL / ATR_TO_SIGMA
    out["risk_vol"] = pd.concat([rv21, rv63, atr_vol, out["implied_vol"]], axis=1).max(axis=1)
    out["trend_ok"] = (last > sma50) & (out["mom_3m"] > 0)
    out["trend_broken"] = (last < sma50) & (last < sma20) & (out["ret_10d"] < 0)
    out["bars"] = c.notna().sum()
    return out.replace([np.inf, -np.inf], np.nan)


def component_scores(
    raw: pd.DataFrame,
    *,
    model_z: Mapping[str, float] | None = None,
    fundamentals: pd.DataFrame | None = None,
    beta_tilt: float = 0.0,
) -> pd.DataFrame:
    idx = raw.index
    comps = pd.DataFrame(index=idx)
    comps["momentum"] = combine(
        [
            (raw["mom_12_1"], 1.0),
            (raw["mom_6_1"], 1.0),
            (raw["mom_3m"], 1.0),
            (raw["ret_10d"], 1.0),
            (raw["rel_strength"], 1.0),
            (raw["persistence"], 1.0),
        ],
        idx,
    )
    comps["trend"] = combine(
        [
            (raw["px_vs_sma50"], 1.0),
            (raw["sma50_vs_sma200"], 1.0),
            (raw["near_high"], 1.0),
            (raw["adx_signed"], 1.0),
            (raw["px_vs_vwap"], 1.0),
            (raw["setup"], 1.0),
        ],
        idx,
    )
    comps["volume"] = combine(
        [
            (raw["volume_trend"], 1.0),
            (raw["rel_volume"], 1.0),
            (raw["updown_volume"], 1.0),
            (raw["log_dollar_volume"], 0.5),
        ],
        idx,
    )
    comps["volatility"] = combine(
        [
            (-raw["atr_pct"], 1.0),
            (-raw["vol_ratio"], 1.0),
            (raw["sharpe_126"], 1.0),
            (-raw["iv_premium"], 1.0),
        ],
        idx,
    )
    if fundamentals is not None and not fundamentals.empty:
        f = fundamentals.reindex(idx)
        parts = [
            (f[name] * (1.0 if weight > 0 else -1.0), abs(weight))
            for name, weight in FUNDAMENTAL_TERMS
            if name in f.columns
        ]
        comps["fundamental"] = combine(parts, idx) if parts else 0.0
    else:
        comps["fundamental"] = 0.0
    model = pd.Series(model_z or {}, dtype=float).reindex(idx)
    comps["model"] = zscore(model).fillna(0.0)
    comps["regime"] = (zscore(raw["beta"]).fillna(0.0) * beta_tilt).clip(-WINSOR, WINSOR)
    return comps


def opportunity_score(components: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Weighted average of the components, standardised across the universe (NaN-free)."""
    total = sum(w for c, w in weights.items() if c in components.columns)
    if total <= 0:
        return pd.Series(0.0, index=components.index)
    raw = sum(components[c] * w for c, w in weights.items() if c in components.columns) / total
    return zscore(pd.Series(raw, index=components.index)).fillna(0.0)
