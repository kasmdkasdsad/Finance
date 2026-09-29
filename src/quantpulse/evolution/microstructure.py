"""Intraday microstructure and micro-volatility, measured at several timescales.

From one session of intraday bars (timestamps, closes, volumes; highs/lows and quotes when available):

* realized volatility sampled at 1, 5, 15, 30 and 60 minutes (annualized) — the *volatility signature*: with
  no microstructure noise it is flat; noise (bid-ask bounce) inflates the finest scales;
* the variance ratio (5-minute variance / five 1-minute variances): below 1 means short-horizon reversal
  (liquidity provision, bounce), above 1 short-horizon momentum;
* bipower variation and the jump share of variance (Barndorff-Nielsen & Shephard);
* lag-1 autocorrelation of 1- and 5-minute returns;
* Roll's effective-spread estimate from the serial covariance of price changes (when that covariance is
  negative), and the quoted spread in basis points when quotes are given;
* Amihud illiquidity (absolute return per dollar traded);
* the intraday volume profile (share of the day's volume in the first and last 30 minutes) and the
  intraday range (Parkinson) volatility.

Each is a number with its sample size; a metric that cannot be computed from the data given is ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

SCALES = (1, 5, 15, 30, 60)  # minutes
MINUTES_PER_YEAR = 252 * 390


def _resample(ts: Sequence[datetime], px: np.ndarray, minutes: int) -> np.ndarray:
    """The last price of each ``minutes``-minute bucket (from 1-minute data)."""
    if minutes <= 1:
        return px
    t0 = ts[0]
    buckets: dict[int, float] = {}
    for t, p in zip(ts, px, strict=True):
        buckets[int((t - t0).total_seconds() // (60 * minutes))] = float(p)
    return np.array([buckets[k] for k in sorted(buckets)], dtype=float)


def realized_vol(px: np.ndarray, minutes: int) -> float | None:
    if len(px) < 3:
        return None
    r = np.diff(np.log(px))
    return float(math.sqrt((r**2).sum() / (len(r) * minutes) * MINUTES_PER_YEAR))


def session_metrics(
    ts: Sequence[datetime],
    close: Sequence[float],
    volume: Sequence[float] | None = None,
    *,
    high: Sequence[float] | None = None,
    low: Sequence[float] | None = None,
    bid: Sequence[float] | None = None,
    ask: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Metrics for one session of 1-minute bars (or quotes sampled each minute)."""
    px = np.asarray(close, dtype=float)
    ok = np.isfinite(px) & (px > 0)
    px, ts = px[ok], [t for t, k in zip(ts, ok, strict=True) if k]
    out: dict[str, Any] = {"bars": len(px)}
    if len(px) < 30:
        out["note"] = "too few bars for microstructure metrics"
        return out
    sig = {}
    for m in SCALES:
        series = _resample(ts, px, m)
        sig[f"rv_{m}m"] = realized_vol(series, m)
    out.update(sig)
    r1 = np.diff(np.log(px))
    p5 = _resample(ts, px, 5)
    r5 = np.diff(np.log(p5))
    v1, v5 = float(r1.var(ddof=1)), float(r5.var(ddof=1)) if len(r5) > 2 else None
    out["variance_ratio_5_1"] = round(v5 / (5 * v1), 4) if v5 is not None and v1 > 0 else None
    fine, coarse = sig.get("rv_1m"), sig.get("rv_30m")
    out["noise_ratio"] = round(fine / coarse, 4) if fine and coarse else None
    bpv = (math.pi / 2) * float(np.sum(np.abs(r1[1:]) * np.abs(r1[:-1])))
    rv = float((r1**2).sum())
    out["jump_share"] = round(max(0.0, 1 - bpv / rv), 4) if rv > 0 else None
    out["autocorr_1m"] = _ac1(r1)
    out["autocorr_5m"] = _ac1(r5)
    dp = np.diff(px)
    cov = float(np.cov(dp[1:], dp[:-1])[0, 1]) if len(dp) > 3 else 0.0
    out["roll_spread_bps"] = round(2 * math.sqrt(-cov) / float(px.mean()) * 1e4, 3) if cov < 0 else None
    out["micro_vol"] = sig.get("rv_1m")
    if bid is not None and ask is not None:
        b, a = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
        m = np.isfinite(b) & np.isfinite(a) & (a > b) & (b > 0)
        if m.any():
            bps = (a[m] - b[m]) / ((a[m] + b[m]) / 2) * 1e4
            out["quoted_spread_bps"] = round(float(np.median(bps)), 3)
            out["quoted_spread_p90_bps"] = round(float(np.percentile(bps, 90)), 3)
    if volume is not None:
        vol = np.asarray(volume, dtype=float)[ok]
        dollars = vol[1:] * px[1:]
        m = dollars > 0
        if m.any():
            out["amihud"] = float(np.mean(np.abs(r1[m]) / dollars[m]) * 1e6)  # per $1M
        total = float(vol.sum())
        if total > 0:
            t0, t_end = ts[0], ts[-1]
            first = sum(v for t, v in zip(ts, vol, strict=True) if t < t0 + timedelta(minutes=30))
            last = sum(v for t, v in zip(ts, vol, strict=True) if t > t_end - timedelta(minutes=30))
            out["volume_open_share"] = round(first / total, 4)
            out["volume_close_share"] = round(last / total, 4)
    if high is not None and low is not None:
        h, lo = np.asarray(high, dtype=float)[ok], np.asarray(low, dtype=float)[ok]
        m = (h > 0) & (lo > 0) & (h >= lo)
        if m.any():
            park = float(np.mean(np.log(h[m] / lo[m]) ** 2) / (4 * math.log(2)))
            out["parkinson_vol"] = round(math.sqrt(park * MINUTES_PER_YEAR), 5)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def _ac1(r: np.ndarray) -> float | None:
    if len(r) < 10 or r.std() == 0:
        return None
    return round(float(np.corrcoef(r[1:], r[:-1])[0, 1]), 4)


def daily_panel(sessions: Sequence[dict[str, Any]]) -> dict[str, list[float | None]]:
    """Session metrics stacked into time series (one value per session per metric)."""
    keys = sorted({k for s in sessions for k, v in s.items() if isinstance(v, int | float) and k != "bars"})
    return {k: [s.get(k) for s in sessions] for k in keys}
