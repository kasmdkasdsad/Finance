"""Point-in-time stock features computed on a price panel (rows = trading days, columns = symbols).

Every value at date *t* uses only data up to and including the close of *t*. Features are raw
(un-oriented): the model learns each sign from data, and the research lab reports it.

Besides the price features there are
    * **earnings** features (post-earnings drift): the abnormal price reaction to the latest release;
    * **sector** features: the average momentum of a stock's industry (industry momentum);
    * **fundamental** features (value and quality, :mod:`quantpulse.domain.fundamental_factors`).

Sector-relative comparisons
    Stock-level features can be *sector-neutralised*: on each date the average of the stock's industry
    peers is subtracted, so "cheap" means cheap *for a bank* or *for a software company*, and momentum
    means beating the industry. Industry-wide effects are kept through the separate sector features.
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


EARNINGS_FEATURES: dict[str, str] = {
    "earn_reaction": "abnormal 2-day reaction to the latest earnings release, in volatility units (63 days)",
}
SECTOR_FEATURES: dict[str, str] = {
    "sector_mom_6_1": "industry average 6-month momentum (skipping the latest month)",
    "sector_ret_1m": "industry average 1-month return",
}
EARNINGS_HOLD_DAYS = 63  # the post-earnings drift is measured over roughly one quarter
MIN_GROUP = 3  # smallest industry group whose average is used (smaller groups use the whole market)


def all_features() -> dict[str, str]:
    from quantpulse.domain.fundamental_factors import FUNDAMENTAL_FEATURES

    return {**FEATURES, **EARNINGS_FEATURES, **SECTOR_FEATURES, **FUNDAMENTAL_FEATURES}


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


def earnings_reaction(
    close: pd.DataFrame,
    benchmark: pd.Series,
    reaction_days: dict[str, list[pd.Timestamp]],
    hold: int = EARNINGS_HOLD_DAYS,
) -> pd.DataFrame:
    """Abnormal return from the close before the reaction day to the close after it, divided by the
    stock's prior 63-day daily volatility × √2. Known from the close of the day after the reaction day
    (the 2-day window absorbs any uncertainty about release timing) and carried for ``hold`` sessions."""
    dates = close.index
    out = pd.DataFrame(np.nan, index=dates, columns=close.columns)
    logret = np.log(close).diff()
    vol = logret.rolling(63, min_periods=40).std()
    bench = benchmark.to_numpy(dtype=float)
    for symbol, days in reaction_days.items():
        if symbol not in close.columns or not days:
            continue
        c = close[symbol].to_numpy(dtype=float)
        v = vol[symbol].to_numpy(dtype=float)
        col = out.columns.get_loc(symbol)
        for day in sorted(days):
            p = int(dates.searchsorted(day))  # first session on/after the reaction day
            if p < 1 or p + 1 >= len(dates):
                continue
            r = c[p + 1] / c[p - 1] - 1
            rb = bench[p + 1] / bench[p - 1] - 1
            sigma = v[p - 1]
            if not (np.isfinite(r) and np.isfinite(rb) and np.isfinite(sigma) and sigma > 0):
                continue
            out.iloc[p + 1 : p + 1 + hold, col] = (r - rb) / (sigma * math.sqrt(2))
    return out


def _group_means(vals: np.ndarray, labels: np.ndarray, min_group: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-date industry averages broadcast to members, and a mask of where the group is big enough."""
    means = np.full(vals.shape, np.nan)
    enough = np.zeros(vals.shape, dtype=bool)
    present = ~np.isnan(vals)
    for g in {x for x in labels if x is not None and x == x}:
        idx = np.flatnonzero(labels == g)
        block = vals[:, idx]
        cnt = present[:, idx].sum(axis=1)
        mean = np.where(cnt > 0, np.nansum(block, axis=1) / np.maximum(cnt, 1), np.nan)
        means[:, idx] = mean[:, None]
        enough[:, idx] = (cnt >= min_group)[:, None]
    return means, enough


def _labels(sectors: pd.Series, columns: pd.Index) -> np.ndarray:
    return np.array([None if pd.isna(x) else str(x) for x in sectors.reindex(columns)], dtype=object)


def group_mean(wide: pd.DataFrame, sectors: pd.Series, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Each cell replaced by its industry's average on that date; the market average when the
    industry has fewer than ``min_group`` names that day (or the stock's industry is unknown)."""
    vals = wide.to_numpy(dtype=float)
    present = ~np.isnan(vals)
    cnt = present.sum(axis=1)
    market = np.where(cnt > 0, np.nansum(vals, axis=1) / np.maximum(cnt, 1), np.nan)
    means, enough = _group_means(vals, _labels(sectors, wide.columns), min_group)
    out = np.where(enough, means, market[:, None])
    out[~present] = np.nan
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def sector_features(
    raw: dict[str, pd.DataFrame], sectors: pd.Series, min_group: int = MIN_GROUP
) -> dict[str, pd.DataFrame]:
    """Industry averages of 6-1 momentum and 1-month return, assigned to every industry member
    (NaN for stocks whose industry is unknown or too small that day)."""
    out: dict[str, pd.DataFrame] = {}
    for name, source in (("sector_mom_6_1", "mom_6_1"), ("sector_ret_1m", "ret_1m")):
        wide = raw[source]
        vals = wide.to_numpy(dtype=float)
        means, enough = _group_means(vals, _labels(sectors, wide.columns), min_group)
        values = np.where(enough & ~np.isnan(vals), means, np.nan)
        out[name] = pd.DataFrame(values, index=wide.index, columns=wide.columns)
    return out


def neutralise(wide: pd.DataFrame, sectors: pd.Series, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Subtract each stock's industry average on each date (see :func:`group_mean`)."""
    return wide - group_mean(wide, sectors, min_group)


def cross_sectional_z(wide: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Per-date z-scores (winsorised at ±``clip``); dates with fewer than 3 values are left NaN."""
    counts = wide.notna().sum(axis=1)
    mean = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = wide.sub(mean, axis=0).div(sd, axis=0).clip(-clip, clip)
    return z.where(counts >= 3, np.nan, axis=0)


def feature_matrix(
    features: dict[str, pd.DataFrame],
    names: list[str] | None = None,
    min_coverage: float = 0.9,
    coverage_names: list[str] | None = None,
    dtype: type = np.float64,
) -> pd.DataFrame:
    """Long matrix indexed by (date, symbol): cross-sectional z-scores, gaps filled with 0 (the
    cross-sectional mean); rows with fewer than ``min_coverage`` of the ``coverage_names`` features
    (default: all of them) present are dropped. Sparse inputs such as fundamentals or earnings are
    left out of ``coverage_names`` so a missing report never removes a stock from the cross-section."""
    names = names or list(FEATURES)
    missing = [n for n in names if n not in features]
    if missing:
        raise DomainError(f"unknown features: {missing}")
    stacked = {n: cross_sectional_z(features[n]).stack(future_stack=True) for n in names}
    frame = pd.DataFrame(stacked)
    frame.index.names = ["date", "symbol"]
    required = [n for n in (coverage_names or names) if n in names]
    coverage = frame[required].notna().mean(axis=1)
    return frame[coverage >= min_coverage].fillna(0.0).astype(dtype)


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
