"""Market regime: trend, volatility, breadth and the yield curve, with what historically followed.

Regime label (on the benchmark)
    * **Uptrend**: above its 200-day average with volatility below the 70th percentile of the past 3 years;
    * **Volatile uptrend**: above the 200-day average but volatility in the top 30%;
    * **Downtrend**: below the 200-day average, volatility normal;
    * **Stress**: below the 200-day average with volatility in the top 30%.

Historical context
    For the loaded history, the benchmark's forward 21-day return is summarised separately for days that
    were in the same trend state (above / below the 200-day average) as today: mean, median, share of
    positive outcomes and the count. This is descriptive, sample-dependent evidence, not a forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantpulse.core.errors import DomainError

VOL_WINDOW = 21
PERCENTILE_LOOKBACK = 756
HIGH_VOL = 0.70


@dataclass(frozen=True, slots=True)
class Conditional:
    state: str
    n: int
    mean: float | None
    median: float | None
    positive_share: float | None


@dataclass(frozen=True, slots=True)
class Regime:
    label: str
    as_of: pd.Timestamp
    price: float
    above_sma200: bool
    sma50_above_sma200: bool
    distance_sma200: float
    return_3m: float
    drawdown_52w: float
    volatility_21d: float
    volatility_percentile: float
    breadth_above_sma200: float | None
    breadth_above_sma50: float | None
    curve_slope_10y_3m: float | None
    curve_inverted: bool | None
    history: list[Conditional]
    notes: list[str]


def _conditional(bench: pd.Series, above: pd.Series, horizon: int = 21) -> list[Conditional]:
    fwd = bench.shift(-horizon) / bench - 1
    frame = pd.DataFrame({"fwd": fwd, "above": above}).dropna()
    out: list[Conditional] = []
    for state, flag in (("above 200-day average", True), ("below 200-day average", False)):
        x = frame.loc[frame["above"] == flag, "fwd"]
        out.append(
            Conditional(
                state=state,
                n=len(x),
                mean=float(x.mean()) if len(x) else None,
                median=float(x.median()) if len(x) else None,
                positive_share=float((x > 0).mean()) if len(x) else None,
            )
        )
    return out


def market_regime(
    benchmark: pd.Series,
    universe_close: pd.DataFrame | None = None,
    curve_10y: float | None = None,
    curve_3m: float | None = None,
) -> Regime:
    b = benchmark.dropna().astype(float)
    if len(b) < 260:
        raise DomainError(f"need at least 260 benchmark closes, got {len(b)}")
    sma50, sma200 = b.rolling(50).mean(), b.rolling(200).mean()
    vol = np.log(b).diff().rolling(VOL_WINDOW).std() * math.sqrt(252)
    recent_vol = vol.dropna().iloc[-PERCENTILE_LOOKBACK:]
    vol_now = float(vol.iloc[-1])
    pct = float((recent_vol <= vol_now).mean())
    price = float(b.iloc[-1])
    above = bool(price > sma200.iloc[-1])
    high_vol = pct >= HIGH_VOL
    label = {
        (True, False): "Uptrend",
        (True, True): "Volatile uptrend",
        (False, False): "Downtrend",
        (False, True): "Stress",
    }[(above, high_vol)]

    breadth200 = breadth50 = None
    if universe_close is not None and not universe_close.empty:
        last = universe_close.ffill().iloc[-1]
        s200 = universe_close.rolling(200).mean().iloc[-1]
        s50 = universe_close.rolling(50).mean().iloc[-1]
        ok200, ok50 = s200.notna(), s50.notna()
        breadth200 = float((last[ok200] > s200[ok200]).mean()) if ok200.any() else None
        breadth50 = float((last[ok50] > s50[ok50]).mean()) if ok50.any() else None

    slope = None if curve_10y is None or curve_3m is None else curve_10y - curve_3m
    notes: list[str] = []
    if slope is not None and slope < 0:
        notes.append(
            "The 10-year yield is below the 3-month yield (inverted curve). Inversions have preceded most US "
            "recessions, with long and variable lags; they say little about the next month."
        )
    if breadth200 is not None and above and breadth200 < 0.5:
        notes.append(
            "The benchmark is above its 200-day average, but fewer than half the universe is (narrow rally)."
        )
    if not above and breadth200 is not None and breadth200 > 0.5:
        notes.append("The benchmark is below its 200-day average while most of the universe is above theirs.")
    return Regime(
        label=label,
        as_of=b.index[-1],
        price=price,
        above_sma200=above,
        sma50_above_sma200=bool(sma50.iloc[-1] > sma200.iloc[-1]),
        distance_sma200=float(price / sma200.iloc[-1] - 1),
        return_3m=float(price / b.iloc[-64] - 1),
        drawdown_52w=float(price / b.iloc[-252:].max() - 1),
        volatility_21d=vol_now,
        volatility_percentile=pct,
        breadth_above_sma200=breadth200,
        breadth_above_sma50=breadth50,
        curve_slope_10y_3m=slope,
        curve_inverted=None if slope is None else slope < 0,
        history=_conditional(b, (b > sma200).where(sma200.notna())),
        notes=notes,
    )
