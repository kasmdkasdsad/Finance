"""Market regime agent: trend, volatility, breadth, correlation and risk appetite — the backdrop every other
view is read against.

Reuses the strategy's own classifier (:func:`quantpulse.domain.trading_regime.classify`, computed during
perception) and adds cross-sectional state from :func:`~quantpulse.brain.indicators.market_statistics`:
rising average correlation and falling breadth are classic signs of a risk-off market.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..context import BrainContext
from ..types import MARKET, AgentFamily, AgentSpec, DataState, Evidence, Opinion, clamp, stance_of
from .base import Agent

LABEL_BIAS = {"bullish": 1.0, "neutral": 0.0, "high_volatility": -0.35, "bearish": -0.8, "risk_off": -1.0}


class MarketRegimeAgent(Agent):
    spec = AgentSpec(
        id="market_regime",
        source="prices",
        failure="no market view this cycle; symbol agents still run and the risk posture falls back to cautious if situational awareness cannot run either",
        name="Market regime",
        description="Classifies the market (bull, bear, choppy, high volatility, risk-off) from trend, "
        "volatility, breadth, correlation and the VIX.",
        family=AgentFamily.SPECIALIST,
        capabilities=("regime", "trend_regime", "volatility_regime", "breadth", "correlation_regime"),
        inputs=("benchmark", "regime", "market_stats", "vix"),
        subjects=("market",),
        priority=10,
        horizon_days=21,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if ctx.regime is not None else "no regime classification (benchmark history missing)"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        r = ctx.regime
        assert r is not None
        stats = ctx.market_stats
        if r.label == "neutral" and r.trend_score == 0 and not r.metrics:
            return [self.abstain(MARKET, "not enough benchmark history to classify", ["benchmark history"])]
        ev: list[Evidence] = [
            Evidence(
                "regime",
                r.label,
                r.description,
                direction=int(LABEL_BIAS.get(r.label, 0) > 0) - int(LABEL_BIAS.get(r.label, 0) < 0),
                strength=0.9,
            ),
            Evidence(
                "trend_score",
                round(r.trend_score, 2),
                "benchmark trend score (-3 … +3)",
                direction=(r.trend_score > 0) - (r.trend_score < 0),
                strength=0.8,
            ),
        ]
        for name, value in r.metrics.items():
            if value is None:
                continue
            ev.append(Evidence(name, round(float(value), 4), f"regime input {name}", strength=0.3))
        corr20, corr120 = stats.get("avg_corr_20"), stats.get("avg_corr_120")
        corr_shift = None
        if corr20 is not None and corr120 is not None:
            corr_shift = corr20 - corr120
            ev.append(
                Evidence(
                    "correlation_regime",
                    round(corr20, 3),
                    f"average pairwise correlation {corr20:.2f} (20d) vs {corr120:.2f} (120d)"
                    + ("; rising correlation is a risk-off sign" if corr_shift > 0.1 else ""),
                    direction=-1 if corr_shift > 0.1 else 0,
                    strength=0.5,
                )
            )
        breadth = stats.get("breadth_50")
        if breadth is not None:
            ev.append(
                Evidence(
                    "breadth_50",
                    round(breadth, 3),
                    f"{breadth:.0%} of stocks above their 50-day average",
                    direction=1 if breadth > 0.6 else (-1 if breadth < 0.4 else 0),
                    strength=0.5,
                )
            )
        if ctx.vix is not None:
            ev.append(
                Evidence(
                    "vix",
                    round(ctx.vix, 2),
                    f"VIX {ctx.vix:.1f}",
                    direction=-1 if ctx.vix > 25 else 0,
                    strength=0.4,
                )
            )
        score = clamp(0.55 * LABEL_BIAS.get(r.label, 0.0) + 0.45 * clamp(r.trend_score / 3.0))
        if corr_shift is not None and corr_shift > 0.1:
            score = clamp(score - 0.15)
        # confidence: how clearly the evidence lines up, lower in stressed markets and without the VIX
        confidence = 0.35 + 0.45 * min(abs(r.trend_score) / 3.0, 1.0)
        if r.stressed:
            confidence -= 0.1
        if ctx.vix is None:
            confidence -= 0.05
        opinion = Opinion(
            agent_id=self.spec.id,
            agent_version=self.spec.version,
            subject=MARKET,
            stance=stance_of(score),
            score=score,
            confidence=confidence,
            horizon_days=self.spec.horizon_days,
            thesis=f"{r.label.replace('_', ' ')} regime: {r.description}. " + "; ".join(r.reasons[:3]),
            evidence=ev,
            data_used=["benchmark daily bars", "breadth", *(["VIX"] if ctx.vix is not None else [])],
            data_missing=[] if ctx.vix is not None else ["VIX"],
            data_quality=DataState.LIVE if ctx.market_open else DataState.MARKET_CLOSED,
            invalidation="benchmark trend score changes sign, or breadth moves across 40%/60%",
            meta={
                "label": r.label,
                "stressed": r.stressed,
                "beta_tilt": r.beta_tilt,
                "trend_score": r.trend_score,
            },
        )
        ctx.working.post("regime", r.label)
        ctx.working.post("regime_score", score)
        return [opinion]
