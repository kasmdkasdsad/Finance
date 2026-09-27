"""Technical agent: price structure over the next week — trend, ADX, MACD, RSI extremes, 20-day breakouts
with volume confirmation, VWAP, support and resistance. Every input is a deterministic indicator from the
cycle's shared table (``raw_signals`` + :mod:`quantpulse.brain.indicators`)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, DataState, Evidence, Opinion, clamp, stance_of
from .base import Agent, symbols_only

WEIGHTS = {
    "trend": 0.30,
    "adx": 0.15,
    "macd": 0.20,
    "rsi": 0.10,
    "breakout": 0.15,
    "vwap": 0.05,
    "levels": 0.05,
}
NEEDED = ("px_vs_sma50", "sma50_vs_sma200", "rsi14", "macd_hist", "adx_signed")


def _t(x: float, scale: float) -> float:
    return math.tanh(x / scale)


class TechnicalAgent(Agent):
    spec = AgentSpec(
        id="technical",
        name="Technical",
        description="Trend, ADX, MACD, RSI, breakouts, VWAP, support and resistance over about a week.",
        family=AgentFamily.SPECIALIST,
        capabilities=(
            "trend",
            "rsi",
            "macd",
            "moving_averages",
            "atr",
            "vwap",
            "breakouts",
            "support_resistance",
        ),
        inputs=("indicators",),
        subjects=("symbol",),
        priority=20,
        horizon_days=5,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if not ctx.indicators.empty else "no price history to analyse"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        return [self._one(ctx, s) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str) -> Opinion:
        g = lambda c: ctx.ind(s, c)  # noqa: E731
        missing = [c for c in NEEDED if g(c) is None]
        if len(missing) > 2 or g("price") is None:
            return self.abstain(s, "not enough indicator history", missing or ["price history"])
        parts: dict[str, float] = {}
        ev: list[Evidence] = []
        state = ctx.state(s)
        q = state if state is not DataState.UNAVAILABLE else DataState.MARKET_CLOSED

        p50, p50_200 = g("px_vs_sma50"), g("sma50_vs_sma200")
        if p50 is not None and p50_200 is not None:
            parts["trend"] = 0.5 * _t(p50, 0.05) + 0.5 * _t(p50_200, 0.05)
            ev.append(
                Evidence(
                    "trend_structure",
                    round(p50, 4),
                    f"price {p50:+.1%} vs 50-day, 50-day {p50_200:+.1%} vs 200-day",
                    direction=int(parts["trend"] > 0) - int(parts["trend"] < 0),
                    strength=0.8,
                    quality=q,
                )
            )
        adx = g("adx_signed")
        if adx is not None:
            parts["adx"] = _t(adx, 25.0)
            ev.append(
                Evidence(
                    "adx_signed",
                    round(adx, 1),
                    f"signed ADX {adx:+.0f} (trend strength and direction)",
                    direction=int(adx > 0) - int(adx < 0),
                    strength=0.5,
                    quality=q,
                )
            )
        hist, cross = g("macd_hist"), g("macd_cross")
        if hist is not None:
            parts["macd"] = clamp(_t(hist, 0.004) + (0.3 * cross / 2 if cross else 0.0))
            label = (
                " (bullish cross)"
                if cross and cross > 0
                else " (bearish cross)"
                if cross and cross < 0
                else ""
            )
            ev.append(
                Evidence(
                    "macd_hist",
                    round(hist, 5),
                    f"MACD histogram {hist:+.3%} of price{label}",
                    direction=int(parts["macd"] > 0) - int(parts["macd"] < 0),
                    strength=0.5,
                    quality=q,
                )
            )
        rsi = g("rsi14")
        if rsi is not None:
            parts["rsi"] = -(rsi - 70) / 30 if rsi > 70 else ((30 - rsi) / 30 if rsi < 30 else 0.0)
            if parts["rsi"]:
                ev.append(
                    Evidence(
                        "rsi14",
                        round(rsi, 1),
                        f"RSI {rsi:.0f}: {'overbought' if rsi > 70 else 'oversold'}",
                        direction=int(parts["rsi"] > 0) - int(parts["rsi"] < 0),
                        strength=0.4,
                        quality=q,
                    )
                )
        vol_ratio = g("volume_ratio_1d") or 1.0
        if g("breakout_20"):
            parts["breakout"] = 0.8 if vol_ratio >= 1.3 else 0.4
            ev.append(
                Evidence(
                    "breakout_20",
                    True,
                    f"above the prior 20-day high on {vol_ratio:.1f}× normal volume",
                    direction=1,
                    strength=0.7,
                    quality=q,
                )
            )
        elif g("breakdown_20"):
            parts["breakout"] = -0.8 if vol_ratio >= 1.3 else -0.4
            ev.append(
                Evidence(
                    "breakdown_20",
                    True,
                    f"below the prior 20-day low on {vol_ratio:.1f}× normal volume",
                    direction=-1,
                    strength=0.7,
                    quality=q,
                )
            )
        vwap = g("px_vs_vwap")
        if vwap is not None and ctx.market_open:
            parts["vwap"] = _t(vwap, 0.01)
            ev.append(
                Evidence(
                    "px_vs_vwap",
                    round(vwap, 4),
                    f"price {vwap:+.2%} vs today's VWAP",
                    direction=int(vwap > 0) - int(vwap < 0),
                    strength=0.3,
                    quality=q,
                )
            )
        up = g("dist_resistance_atr")
        down = g("dist_support_atr")
        if up is not None and 0 <= up < 0.5 and not g("breakout_20"):
            parts["levels"] = -0.5
            ev.append(
                Evidence(
                    "near_resistance",
                    round(up, 2),
                    f"{up:.1f} ATR below 20-day resistance",
                    direction=-1,
                    strength=0.3,
                    quality=q,
                )
            )
        elif down is not None and 0 <= down < 0.5 and (p50 or 0) > 0:
            parts["levels"] = 0.5
            ev.append(
                Evidence(
                    "near_support",
                    round(down, 2),
                    f"{down:.1f} ATR above 20-day support in an uptrend",
                    direction=1,
                    strength=0.3,
                    quality=q,
                )
            )

        used = sum(WEIGHTS[k] for k in parts)
        score = clamp(sum(WEIGHTS[k] * v for k, v in parts.items()) / used) if used else 0.0
        signs = [v for v in parts.values() if abs(v) > 0.05]
        agree = sum(1 for v in signs if (v > 0) == (score > 0)) / len(signs) if signs and score else 0.0
        confidence = (0.25 + 0.55 * agree) * (1.0 - 0.15 * len(missing))
        low20, sma50 = g("low_20"), g("sma50")
        invalidation = (
            f"close below the 20-day low ({low20:,.2f})"
            if score > 0 and low20
            else f"close back above the 50-day average ({sma50:,.2f})"
            if score < 0 and sma50
            else None
        )
        return Opinion(
            agent_id=self.spec.id,
            agent_version=self.spec.version,
            subject=s,
            stance=stance_of(score),
            score=score,
            confidence=confidence,
            horizon_days=self.spec.horizon_days,
            thesis=f"{s} technicals {stance_of(score).value}: "
            + ", ".join(e.detail for e in sorted(ev, key=lambda e: -e.strength)[:3]),
            evidence=ev,
            data_used=["daily bars", *(["live quote"] if s in ctx.quotes else [])],
            data_missing=missing,
            data_quality=q,
            invalidation=invalidation,
            meta={"components": {k: round(v, 3) for k, v in parts.items()}},
        )
