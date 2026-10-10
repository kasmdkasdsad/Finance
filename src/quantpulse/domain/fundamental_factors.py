"""Point-in-time value and quality factors from SEC XBRL facts.

Inputs are annual income/cash-flow figures, year-end balance sheets and the public float, keyed by the
period they describe. A figure only becomes usable after a conservative **availability lag** measured
from its period end (10-K deadlines are 60-90 days), so a backtest never trades on numbers that had not
been published yet.

Market value without share counts
    Share counts are split-sensitive and differ by share class, while vendor prices are split-adjusted
    to today's basis. So market value is taken from the **public float** (dollar value of shares held by
    non-insiders, reported on each 10-K cover as of the second fiscal quarter's end ``D``), rolled
    forward with split-adjusted returns: ``MV_t = float_D · close_t / close_D``. This is immune to
    splits and share classes and is float-adjusted, like the S&P 500 itself.

Factors (all oriented as reported; the model learns signs)
    ============================  ===================================================
    ``earnings_yield``            net income / MV
    ``fcf_yield``                 (operating cash flow − capital expenditure) / MV
    ``book_to_market``            book equity / MV (positive equity only)
    ``gross_profitability``       gross profit / total assets (Novy-Marx)
    ``roe``                       net income / book equity (positive equity only)
    ``asset_growth``              one-year growth in total assets (investment)
    ``accruals``                  (net income − operating cash flow) / total assets (Sloan)
    ============================  ===================================================

Figures older than ``MAX_AGE_DAYS`` since they became available are treated as missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

ANNUAL_LAG = timedelta(days=90)
BALANCE_LAG = timedelta(days=90)
FLOAT_LAG = timedelta(days=270)  # float is measured at Q2 end; the 10-K arrives ~6 months + 60-90 days later
MAX_AGE_DAYS = 550

FUNDAMENTAL_FEATURES: dict[str, str] = {
    "earnings_yield": "net income / market value (point-in-time)",
    "fcf_yield": "free cash flow / market value",
    "book_to_market": "book equity / market value",
    "gross_profitability": "gross profit / total assets",
    "roe": "net income / book equity",
    "asset_growth": "1-year growth in total assets",
    "accruals": "(net income − operating cash flow) / total assets",
}


@dataclass(frozen=True, slots=True)
class Fact:
    end: date
    value: float
    start: date | None = None


@dataclass
class CompanyFacts:
    """Annual durations and instants for one company (lists need not be sorted)."""

    net_income: list[Fact] = field(default_factory=list)
    operating_cash_flow: list[Fact] = field(default_factory=list)
    capex: list[Fact] = field(default_factory=list)
    gross_profit: list[Fact] = field(default_factory=list)
    assets: list[Fact] = field(default_factory=list)
    equity: list[Fact] = field(default_factory=list)
    public_float: list[Fact] = field(default_factory=list)


def is_annual(f: Fact) -> bool:
    return f.start is None or 330 <= (f.end - f.start).days <= 400


def _by_end(facts: Sequence[Fact], annual_only: bool = False) -> dict[date, float]:
    out: dict[date, float] = {}
    for f in sorted(facts, key=lambda x: x.end):
        if annual_only and not is_annual(f):
            continue
        out[f.end] = f.value
    return out


def point_in_time(values: Mapping[date, float], lag: timedelta, dates: pd.DatetimeIndex) -> pd.Series:
    """Latest value whose ``end + lag`` is on or before each date (NaN once older than ``MAX_AGE_DAYS``)."""
    if not values:
        return pd.Series(np.nan, index=dates)
    avail = pd.DatetimeIndex([pd.Timestamp(e + lag) for e in values])
    s = pd.Series(list(values.values()), index=avail).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    when = pd.Series(s.index, index=s.index)
    value = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    since = when.reindex(when.index.union(dates)).ffill().reindex(dates)
    age = (pd.Series(dates, index=dates) - since).dt.days
    return value.where(age <= MAX_AGE_DAYS)


def market_value(public_float: Sequence[Fact], close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """``float_D · close_t / close_D`` from the latest float report available at each date."""
    c = close.dropna()
    if c.empty:
        return pd.Series(np.nan, index=dates)
    base: dict[date, float] = {}
    for f in sorted(public_float, key=lambda x: x.end):
        if f.value <= 0:
            continue
        at = c.index.searchsorted(pd.Timestamp(f.end), side="right") - 1
        if at < 0:
            continue  # the float date is before the price history starts
        base[f.end] = f.value / float(c.iloc[at])  # dollars per unit of (split-adjusted) price
    scale = point_in_time(base, FLOAT_LAG, dates)
    return scale * close.reindex(dates)


def _ratio(num: pd.Series, den: pd.Series, positive_den: bool = True) -> pd.Series:
    d = den.where(den > 0) if positive_den else den.replace(0.0, np.nan)
    return num / d


def company_factors(facts: CompanyFacts, close: pd.Series, dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    ni = point_in_time(_by_end(facts.net_income, True), ANNUAL_LAG, dates)
    cfo = point_in_time(_by_end(facts.operating_cash_flow, True), ANNUAL_LAG, dates)
    capex = point_in_time(_by_end(facts.capex, True), ANNUAL_LAG, dates)
    gp = point_in_time(_by_end(facts.gross_profit, True), ANNUAL_LAG, dates)
    assets_by_end = _by_end(facts.assets)
    assets = point_in_time(assets_by_end, BALANCE_LAG, dates)
    equity = point_in_time(_by_end(facts.equity), BALANCE_LAG, dates)
    growth: dict[date, float] = {}
    ends = sorted(assets_by_end)
    for i, e in enumerate(ends):
        prior = [p for p in ends[:i] if 330 <= (e - p).days <= 400]
        if prior and assets_by_end[prior[-1]] > 0:
            growth[e] = assets_by_end[e] / assets_by_end[prior[-1]] - 1.0
    mv = market_value(facts.public_float, close, dates)
    return {
        "earnings_yield": _ratio(ni, mv),
        "fcf_yield": _ratio(cfo - capex, mv),
        "book_to_market": _ratio(equity.where(equity > 0), mv),
        "gross_profitability": _ratio(gp, assets),
        "roe": _ratio(ni, equity),
        "asset_growth": point_in_time(growth, BALANCE_LAG, dates),
        "accruals": _ratio(ni - cfo, assets),
    }


def fundamental_features(facts: Mapping[str, CompanyFacts], close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Factor panels (dates x symbols); symbols without facts are all-NaN."""
    dates = pd.DatetimeIndex(close.index)
    per_symbol = {s: company_factors(facts[s], close[s], dates) for s in close.columns if s in facts}
    out: dict[str, pd.DataFrame] = {}
    for name in FUNDAMENTAL_FEATURES:
        frame = pd.DataFrame(
            {s: per_symbol[s][name] if s in per_symbol else np.nan for s in close.columns}, index=dates
        )
        out[name] = frame.replace([np.inf, -np.inf], np.nan)
    return out
