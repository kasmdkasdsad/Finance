"""Point-in-time stock features computed on a price panel (rows = trading days, columns = symbols).

Every value at date *t* uses only data up to and including the close of *t*. Features are raw
(un-oriented): the model learns each sign from data, and the research lab reports it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpulse.core.errors import DomainError

ANNUAL = math.sqrt(252)
WARMUP = 252  # the longest look-back (12-month momentum, 52-week high, 1-year beta)

FEATURES: dict[str, str] = {
    "mom_12_1": "12-month return skipping the latest month",
    "mom_6_1": "6-month return skipping the latest month",
    "mom_3m": "3-month return",
    "ret_1m": "1-month return (short-term reversal when negative)",
    "ret_5d": "5-day return",
    "trend_50_200": "50-day vs 200-day moving average",
    "px_vs_sma50": "price vs 50-day moving average",
    "high_52w": "distance below the 52-week high",
    "vol_63": "3-month realised volatility",
    "vol_ratio": "1-month vs 3-month volatility",
    "sharpe_126": "6-month return / volatility",
    "rsi_14": "14-day RSI (Wilder)",
    "bollinger_b": "position inside the 20-day Bollinger bands",
    "beta_252": "1-year beta to the benchmark",
    "idio_vol_63": "3-month idiosyncratic volatility",
    "max_ret_21": "largest daily gain in the last month",
    "skew_63": "3-month skewness of daily returns",
    "volume_trend": "20-day vs 120-day average volume",
}


@dataclass
class Panel:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series

    def __post_init__(self) -> None:
        if self.close.empty:
            raise DomainError("empty price panel")
        for name in ("high", "low", "volume"):
            frame = getattr(self, name)
            if not frame.index.equals(self.close.index) or list(frame.columns) != list(self.close.columns):
                raise DomainError(f"{name} must share the close panel's index and columns")
        if not self.benchmark.index.equals(self.close.index):
            raise DomainError("benchmark must share the close panel's index")

    @property
    def symbols(self) -> list[str]:
        return [str(c) for c in self.close.columns]


def compute_features(panel: Panel) -> dict[str, pd.DataFrame]:
    """Raw feature panels keyed by :data:`FEATURES` name."""
    c = panel.close
    ret = c.pct_change(fill_method=None)
    logret = np.log(c).diff()
    bench_ret = panel.benchmark.pct_change(fill_method=None)

    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0.0, np.nan))
    flat = (gain == 0) & (loss == 0)
    rsi = rsi.where(loss > 0, 100.0).where(~flat, 50.0).where(gain.notna())

    bench_var = bench_ret.rolling(252).var()
    cov = ret.rolling(252).cov(bench_ret)
    beta = cov.div(bench_var, axis=0)
    cov63 = ret.rolling(63).cov(bench_ret)
    beta63 = cov63.div(bench_ret.rolling(63).var(), axis=0)
    idio_var = ret.rolling(63).var() - beta63.pow(2).mul(bench_ret.rolling(63).var(), axis=0)

    volume = panel.volume.where(panel.volume > 0)
    out = {
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "mom_6_1": c.shift(21) / c.shift(126) - 1,
        "mom_3m": c / c.shift(63) - 1,
        "ret_1m": c / c.shift(21) - 1,
        "ret_5d": c / c.shift(5) - 1,
        "trend_50_200": sma50 / sma200 - 1,
        "px_vs_sma50": c / sma50 - 1,
        "high_52w": c / c.rolling(252).max() - 1,
        "vol_63": logret.rolling(63).std() * ANNUAL,
        "vol_ratio": logret.rolling(21).std() / logret.rolling(63).std(),
        "sharpe_126": ret.rolling(126).mean() / ret.rolling(126).std() * ANNUAL,
        "rsi_14": rsi,
        "bollinger_b": (c - (sma20 - 2 * sd20)) / (4 * sd20),
        "beta_252": beta,
        "idio_vol_63": np.sqrt(idio_var.clip(lower=0)) * ANNUAL,
        "max_ret_21": ret.rolling(21).max(),
        "skew_63": ret.rolling(63).skew(),
        "volume_trend": volume.rolling(20, min_periods=15).mean() / volume.rolling(120, min_periods=90).mean()
        - 1,
    }
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def cross_sectional_z(wide: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Per-date z-scores (winsorised at ±``clip``); dates with fewer than 3 values are left NaN."""
    counts = wide.notna().sum(axis=1)
    mean = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = wide.sub(mean, axis=0).div(sd, axis=0).clip(-clip, clip)
    return z.where(counts >= 3, np.nan, axis=0)


def feature_matrix(
    features: dict[str, pd.DataFrame], names: list[str] | None = None, min_coverage: float = 0.9
) -> pd.DataFrame:
    """Long matrix indexed by (date, symbol): cross-sectional z-scores, gaps filled with 0 (the
    cross-sectional mean); rows with fewer than ``min_coverage`` of the features present are dropped."""
    names = names or list(FEATURES)
    missing = [n for n in names if n not in features]
    if missing:
        raise DomainError(f"unknown features: {missing}")
    stacked = {n: cross_sectional_z(features[n]).stack(future_stack=True) for n in names}
    frame = pd.DataFrame(stacked)
    frame.index.names = ["date", "symbol"]
    coverage = frame.notna().mean(axis=1)
    return frame[coverage >= min_coverage].fillna(0.0)


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple return from the close of *t* to the close of *t + horizon* (NaN where not yet known)."""
    if horizon < 1:
        raise DomainError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1


def rank_gauss(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-date ranks mapped to standard-normal scores (a robust, outlier-free regression target)."""
    from scipy.stats import norm

    ranks = wide.rank(axis=1)
    n = wide.notna().sum(axis=1)
    u = ranks.sub(0.5).div(n, axis=0)
    return pd.DataFrame(norm.ppf(u.to_numpy(dtype=float)), index=wide.index, columns=wide.columns)


def row_spearman(a: pd.DataFrame, b: pd.DataFrame, min_names: int = 5) -> pd.Series:
    """Per-date Spearman correlation between two aligned wide frames."""
    mask = a.notna() & b.notna()
    ra = a.where(mask).rank(axis=1)
    rb = b.where(mask).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    return ic.where(mask.sum(axis=1) >= min_names)
