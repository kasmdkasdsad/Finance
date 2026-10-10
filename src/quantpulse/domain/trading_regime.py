"""Market regime for the paper-trading strategy: how much capital to deploy and how picky to be.

Inputs (daily closes, the last row may be today's live price)
    * SPY and QQQ trend: price vs the 200-day average, the 50/200-day cross, the 3-month return;
    * breadth: the share of the trading universe above its 50- and 200-day averages;
    * volatility: SPY's 21-day realised volatility, its percentile over the past three years, and the VIX
      when a live VIX quote is available;
    * drawdown: SPY's distance below its 52-week high.

Trend points (−3.5 … +3.5)
    ±1 SPY above/below its 200-day average · ±1 QQQ likewise · ±0.5 SPY 50-day above/below 200-day ·
    ±0.5 SPY 3-month return up/down · +0.5 breadth (above 200-day) > 60% / −0.5 below 40%.

Labels (first match wins)
    ``risk_off``         trend ≤ −1.5 with stressed volatility, or VIX ≥ ``vix_risk_off``, or SPY ≥ 20% below
                         its 52-week high;
    ``bearish``          trend ≤ −1;
    ``high_volatility``  realised-volatility percentile ≥ 80% or VIX ≥ ``vix_high``;
    ``bullish``          trend ≥ 1.5;
    ``neutral``          otherwise.

The label drives the share of maximum exposure deployed and the extra score new positions need (both
configurable), and a per-stock *beta tilt*: high-beta names are favoured in bullish markets and penalised
in bearish ones. Nothing here assumes markets rise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LABELS: tuple[str, ...] = ("bullish", "neutral", "high_volatility", "bearish", "risk_off")
DESCRIPTIONS: dict[str, str] = {
    "bullish": "Bullish: broad uptrend with normal volatility",
    "neutral": "Neutral: mixed trend signals",
    "high_volatility": "High volatility: volatility is elevated",
    "bearish": "Bearish: the major indexes are in downtrends",
    "risk_off": "Risk-off: downtrend with market stress",
}
BETA_TILT: dict[str, float] = {
    "bullish": 0.5,
    "neutral": 0.0,
    "high_volatility": -0.5,
    "bearish": -0.75,
    "risk_off": -1.0,
}
MIN_HISTORY = 210
VOL_LOOKBACK = 756
MIN_VOL_READINGS = 250


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    bull_trend: float = 1.5
    bear_trend: float = -1.0
    risk_off_trend: float = -1.5
    vol_percentile_high: float = 0.80
    vix_high: float = 25.0
    vix_risk_off: float = 32.0
    drawdown_risk_off: float = -0.20
    breadth_strong: float = 0.60
    breadth_weak: float = 0.40


@dataclass(frozen=True, slots=True)
class MarketRegime:
    label: str
    trend_score: float
    stressed: bool
    beta_tilt: float
    metrics: dict[str, float | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        return DESCRIPTIONS[self.label]


def _trend(series: pd.Series) -> tuple[bool | None, bool | None, float | None]:
    s = series.dropna()
    if len(s) < 200:
        return None, None, None
    sma50, sma200 = s.iloc[-50:].mean(), s.iloc[-200:].mean()
    ret_3m = float(s.iloc[-1] / s.iloc[-64] - 1) if len(s) > 64 else None
    return bool(s.iloc[-1] > sma200), bool(sma50 > sma200), ret_3m


def breadth(close: pd.DataFrame) -> tuple[float | None, float | None]:
    """Share of names above their 50- and 200-day averages (names without enough history are skipped)."""
    if close.empty:
        return None, None
    last = close.ffill().iloc[-1]
    out: list[float | None] = []
    for window in (50, 200):
        if len(close) < window:
            out.append(None)
            continue
        tail = close.iloc[-window:]
        ok = (tail.notna().sum() >= int(0.9 * window)) & last.notna()
        sma = tail.mean()
        out.append(float((last[ok] > sma[ok]).mean()) if ok.any() else None)
    return out[0], out[1]


def classify(
    spy: pd.Series,
    qqq: pd.Series | None = None,
    *,
    breadth_50: float | None = None,
    breadth_200: float | None = None,
    vix: float | None = None,
    thresholds: RegimeThresholds | None = None,
) -> MarketRegime:
    t = thresholds or RegimeThresholds()
    s = spy.dropna().astype(float)
    if len(s) < MIN_HISTORY:
        return MarketRegime(
            label="neutral",
            trend_score=0.0,
            stressed=False,
            beta_tilt=BETA_TILT["neutral"],
            metrics={"spy_bars": float(len(s))},
            reasons=[f"only {len(s)} SPY closes (need {MIN_HISTORY}): assuming a neutral market"],
        )
    reasons: list[str] = []
    score = 0.0
    spy_above, spy_cross, spy_3m = _trend(s)
    score += 1.0 if spy_above else -1.0
    reasons.append(f"SPY {'above' if spy_above else 'below'} its 200-day average")
    score += 0.5 if spy_cross else -0.5
    if spy_3m is not None:
        score += 0.5 if spy_3m > 0 else -0.5
    qqq_above: bool | None = None
    if qqq is not None:
        qqq_above, _, _ = _trend(qqq.astype(float))
        if qqq_above is not None:
            score += 1.0 if qqq_above else -1.0
            reasons.append(f"QQQ {'above' if qqq_above else 'below'} its 200-day average")
    if breadth_200 is not None:
        if breadth_200 > t.breadth_strong:
            score += 0.5
            reasons.append(f"broad participation ({breadth_200:.0%} of stocks above their 200-day average)")
        elif breadth_200 < t.breadth_weak:
            score -= 0.5
            reasons.append(f"weak breadth ({breadth_200:.0%} of stocks above their 200-day average)")

    logret = np.log(s).diff()
    vol = logret.rolling(21).std() * math.sqrt(252)
    vol_now = float(vol.iloc[-1])
    recent = vol.dropna().iloc[-VOL_LOOKBACK:]
    # A percentile needs a real history behind it: under a year of volatility readings is not evidence.
    vol_pct = float((recent <= vol_now).mean()) if len(recent) >= MIN_VOL_READINGS else None
    drawdown = float(s.iloc[-1] / s.iloc[-252:].max() - 1)
    vol_high = vol_pct is not None and vol_pct >= t.vol_percentile_high
    stressed = vol_high or (vix is not None and vix >= t.vix_high)
    if vol_high:
        reasons.append(
            f"SPY 21-day volatility {vol_now:.0%} is at the {vol_pct:.0%} percentile of its history"
        )
    if vix is not None:
        reasons.append(f"VIX {vix:.1f}")

    if (
        (score <= t.risk_off_trend and stressed)
        or (vix is not None and vix >= t.vix_risk_off)
        or drawdown <= t.drawdown_risk_off
    ):
        label = "risk_off"
        if drawdown <= t.drawdown_risk_off:
            reasons.append(f"SPY is {drawdown:.0%} below its 52-week high")
    elif score <= t.bear_trend:
        label = "bearish"
    elif stressed:
        label = "high_volatility"
    elif score >= t.bull_trend:
        label = "bullish"
    else:
        label = "neutral"
    return MarketRegime(
        label=label,
        trend_score=score,
        stressed=stressed,
        beta_tilt=BETA_TILT[label],
        metrics={
            "spy_price": float(s.iloc[-1]),
            "spy_above_sma200": float(bool(spy_above)),
            "spy_sma50_above_sma200": float(bool(spy_cross)),
            "spy_return_3m": spy_3m,
            "qqq_above_sma200": None if qqq_above is None else float(qqq_above),
            "breadth_above_sma50": breadth_50,
            "breadth_above_sma200": breadth_200,
            "spy_volatility_21d": vol_now,
            "spy_volatility_percentile": vol_pct,
            "spy_drawdown_52w": drawdown,
            "vix": vix,
        },
        reasons=reasons,
    )
