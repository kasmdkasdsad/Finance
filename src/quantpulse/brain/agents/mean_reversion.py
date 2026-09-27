"""Mean-reversion agent: is the price stretched far enough from its recent mean to snap back within a week?

Stretch is measured three ways from the shared indicator table — the 20-day z-score, RSI(14) and the last
five days' move in volatility units — and then filtered by the trend: an oversold pullback inside an
uptrend is the classic buyable dip, while fading a strong trend (overbought in a powerful uptrend, oversold
below a falling 200-day average) is damped rather than trusted.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import agreement, opinion, sgn, squash, symbol_quality

NEEDED = ("z20", "rsi14", "ret_5d_z")


class MeanReversionAgent(Agent):
    spec = AgentSpec(
        id="mean_reversion",
        name="Mean reversion",
        description="Short-term stretch from the 20-day mean (z-score, RSI, 5-day move in sigmas), filtered "
        "by trend strength so strong trends are not faded blindly.",
        family=AgentFamily.SPECIALIST,
        capabilities=("oversold", "overbought", "pullback_in_trend", "stretch"),
        inputs=("indicators",),
        subjects=("symbol",),
        priority=30,
        horizon_days=5,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if not ctx.indicators.empty else "no price history to analyse"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        return [self._one(ctx, s) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str) -> Opinion:
        g = lambda c: ctx.ind(s, c)  # noqa: E731
        missing = [c for c in NEEDED if g(c) is None]
        if len(missing) > 1:
            return self.abstain(s, "not enough history to measure stretch", missing)
        q = symbol_quality(ctx, s)
        ev: list[Evidence] = []
        parts: dict[str, float] = {}
        z20, rsi, r5z = g("z20"), g("rsi14"), g("ret_5d_z")
        if z20 is not None:
            parts["z20"] = -squash(z20, 2.0)
            ev.append(
                Evidence("z20", round(z20, 2), f"{z20:+.1f}σ from the 20-day mean", -sgn(z20), 0.7, quality=q)
            )
        if rsi is not None:
            parts["rsi"] = -squash(rsi - 50, 25.0)
            ev.append(Evidence("rsi14", round(rsi, 1), f"RSI {rsi:.0f}", -sgn(rsi - 50), 0.5, quality=q))
        if r5z is not None:
            parts["move_5d"] = -squash(r5z, 2.0)
            ev.append(
                Evidence("ret_5d_z", round(r5z, 2), f"5-day move {r5z:+.1f}σ", -sgn(r5z), 0.4, quality=q)
            )
        raw = sum(parts.values()) / len(parts)
        stretched = (z20 is not None and abs(z20) >= 1.5) or (rsi is not None and (rsi >= 70 or rsi <= 30))
        adx = g("adx_signed") or 0.0
        trend = sgn(adx) if abs(adx) >= 20 else 0
        strength = min(abs(adx) / 40.0, 1.0)
        below_200 = (g("px_vs_sma200") or 0.0) < 0
        note = "no strong trend"
        if trend and sgn(raw) == trend:
            factor, note = 1.2, "stretched against a trend that should resume (pullback in trend)"
        elif trend and sgn(raw) == -trend:
            factor, note = 1.0 - 0.6 * strength, "fading a strong trend: damped"
        else:
            factor = 1.0
        if raw > 0 and below_200 and trend <= 0:
            factor *= 0.6
            note += "; oversold below a falling 200-day average (falling-knife risk)"
        score = raw * factor if stretched else raw * 0.3
        ev.append(Evidence("trend_filter", round(adx, 1), note, 0, 0.6, quality=q))
        stretch_mag = min(max(abs(z20 or 0.0) / 2.5, abs((rsi or 50) - 50) / 30), 1.0)
        confidence = (0.2 + 0.45 * stretch_mag) * (0.5 + 0.5 * agreement(parts.values(), score))
        if factor < 1.0:
            confidence *= factor  # less sure when fading a trend (or catching a falling knife)
        if not stretched:
            confidence *= 0.5
        if ctx.regime is not None and ctx.regime.label in ("high_volatility", "risk_off"):
            confidence *= 0.8
        lo, hi = g("low_20"), g("high_20")
        invalidation = (
            f"close below the 20-day low ({lo:,.2f})"
            if score > 0 and lo
            else f"close above the 20-day high ({hi:,.2f})"
            if score < 0 and hi
            else None
        )
        label = "oversold" if raw > 0.15 else "overbought" if raw < -0.15 else "not stretched"
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} {label}: {', '.join(e.detail for e in ev[:3])}; {note}",
            ev,
            quality=q,
            used=["daily bars", *(["live quote"] if s in ctx.quotes else [])],
            missing=missing,
            invalidation=invalidation,
            meta={"components": {k: round(v, 3) for k, v in parts.items()}, "trend_factor": round(factor, 2)},
        )
