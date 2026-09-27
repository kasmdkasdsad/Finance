"""Factor agent: the stock model's view and the stock's factor profile.

The direction comes from QuantPulse's walk-forward stock model (``services/model.py``), which already
combines momentum, reversal, trend, volatility, fundamental, earnings and sector features and whose
probability of beating the benchmark is calibrated out of sample — the agent does not refit anything. It
adds the factor exposures behind that view (momentum, value, quality, low volatility, beta), each ranked
against the model universe, so the other agents and the dashboard can see *why* the model likes a stock.
A model that has not been refreshed for several sessions weighs less.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import cross_section_z, opinion, sgn, symbol_quality

# factor -> [(feature, sign)]
FACTORS = {
    "momentum": [("mom_12_1", 1), ("mom_6_1", 1)],
    "value": [("earnings_yield", 1), ("book_to_market", 1), ("fcf_yield", 1)],
    "quality": [("gross_profitability", 1), ("roe", 1), ("accruals", -1)],
    "low_volatility": [("vol_63", -1), ("idio_vol_63", -1)],
    "beta": [("beta_252", 1)],
}
STALE_DAYS = 5


class FactorAgent(Agent):
    spec = AgentSpec(
        id="factor",
        source="model",
        failure="skips without a completed stock-model run; a stale run weighs less",
        name="Factor model",
        description="The walk-forward stock model's calibrated view, with the momentum, value, quality, "
        "low-volatility and beta exposures behind it.",
        family=AgentFamily.SPECIALIST,
        capabilities=("stock_model", "factor_exposure", "calibrated_probability"),
        inputs=("model",),
        subjects=("symbol",),
        priority=30,
        horizon_days=21,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        if ctx.model is None or not ctx.model.live:
            return "the stock model has no completed run this cycle"
        return None

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        assert ctx.model is not None
        feats = ctx.model.features
        profile: dict[str, pd.Series] = {}
        for name, cols in FACTORS.items():
            have = [(c, sgn_) for c, sgn_ in cols if c in feats.columns]
            if have:
                z = cross_section_z(feats[[c for c, _ in have]])
                profile[name] = sum(z[c] * sgn_ for c, sgn_ in have) / len(have)
        exposures = pd.DataFrame(profile)
        return [self._one(ctx, s, exposures) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str, exposures: pd.DataFrame) -> Opinion:
        assert ctx.model is not None
        live = ctx.model.live.get(s)
        if live is None:
            return self.abstain(
                s, "not scored by the stock model (outside its universe or history)", ["model score"]
            )
        q = symbol_quality(ctx, s)
        age = (ctx.as_of.date() - ctx.model.as_of).days
        stale = age > STALE_DAYS
        prob = live.prob_outperform
        score = max(-1.0, min(1.0, (prob - 0.5) * 5))
        ev = [
            Evidence(
                "prob_outperform",
                round(prob, 4),
                f"model: {prob:.0%} chance of beating {ctx.benchmark_symbol} over {ctx.model.label or 'its horizon'} "
                f"(rating {live.rating}/10, z {live.z:+.2f})",
                sgn(score),
                0.9,
                source="stock_model",
                quality=q,
            ),
            Evidence(
                "expected_excess_return",
                round(live.expected_excess_return, 4),
                f"expected excess return {live.expected_excess_return:+.1%}",
                sgn(live.expected_excess_return),
                0.5,
                source="stock_model",
                quality=q,
            ),
        ]
        tilts: dict[str, float] = {}
        if s in exposures.index:
            for name, value in exposures.loc[s].dropna().items():
                tilts[str(name)] = round(float(value), 2)
                if abs(value) >= 0.75:
                    ev.append(
                        Evidence(
                            f"exposure_{name}",
                            round(float(value), 2),
                            f"{'high' if value > 0 else 'low'} {str(name).replace('_', ' ')} exposure ({value:+.1f}σ)",
                            0,
                            0.4,
                            source="stock_model",
                            quality=q,
                        )
                    )
        if stale:
            ev.append(Evidence("model_age", age, f"model run is {age} days old", 0, 0.5, quality=q))
        confidence = (0.3 + 0.4 * min(abs(live.z) / 2.0, 1.0)) * (0.6 if stale else 1.0)
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s}: "
            + ev[0].detail
            + (f"; tilts {', '.join(f'{k} {v:+.1f}' for k, v in tilts.items())}" if tilts else ""),
            ev,
            quality=q,
            used=[f"stock model ({ctx.model.label}, {ctx.model.as_of})"],
            invalidation="the next model run moving it out of the top half of the universe",
            meta={
                "exposures": tilts,
                "rating": live.rating,
                "model_z": live.z,
                "model_age_days": age,
                "sector": live.sector_label,
            },
        )
