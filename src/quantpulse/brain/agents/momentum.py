"""Momentum agent: relative strength and its persistence over about a month.

Scores each focus symbol against the whole universe (cross-sectional z-scores of 12-1, 6-1 and 3-month
momentum, the last month, relative strength vs the benchmark and trend persistence), then notes
acceleration or deterioration (the last month lagging the 3-month pace).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion, clamp, stance_of
from .base import Agent, symbols_only

HORIZONS = {
    "mom_12_1": "12-1 month",
    "mom_6_1": "6-1 month",
    "mom_3m": "3-month",
    "ret_21d": "1-month",
    "rel_strength": "3-month vs benchmark",
    "persistence": "trend persistence",
}
MIN_UNIVERSE = 15


class MomentumAgent(Agent):
    spec = AgentSpec(
        id="momentum",
        name="Momentum",
        description="Relative strength, acceleration, deceleration and persistence across horizons, ranked "
        "against the whole universe.",
        family=AgentFamily.SPECIALIST,
        capabilities=("relative_strength", "acceleration", "persistence", "momentum_deterioration"),
        inputs=("indicators",),
        subjects=("symbol",),
        priority=20,
        horizon_days=21,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        cols = [c for c in HORIZONS if c in ctx.indicators.columns]
        if len(ctx.indicators) < MIN_UNIVERSE or len(cols) < 3:
            return f"cross-section too small ({len(ctx.indicators)} symbols) to rank momentum"
        return None

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        cols = [c for c in HORIZONS if c in ctx.indicators.columns]
        frame = ctx.indicators[cols].apply(pd.to_numeric, errors="coerce")
        z = ((frame - frame.mean()) / frame.std().replace(0, pd.NA)).clip(-3, 3)
        return [self._one(ctx, s, z) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str, z: pd.DataFrame) -> Opinion:
        if s not in z.index:
            return self.abstain(s, "not in the ranked universe", ["price history"])
        row = z.loc[s].dropna()
        if len(row) < 3:
            return self.abstain(
                s, "too few momentum horizons available", [c for c in HORIZONS if c not in row.index]
            )
        ev: list[Evidence] = []
        for col, label in HORIZONS.items():
            if col in row.index:
                raw = ctx.ind(s, col)
                shown = (
                    f"{raw:+.1%}"
                    if raw is not None and col != "persistence"
                    else f"{raw:+.2f}"
                    if raw is not None
                    else "n/a"
                )
                ev.append(
                    Evidence(
                        col,
                        round(float(row[col]), 3),
                        f"{label} {shown} (z {row[col]:+.1f} vs universe)",
                        direction=int(row[col] > 0) - int(row[col] < 0),
                        strength=0.5,
                    )
                )
        score = clamp(float(row.mean()) / 2.0)
        accel, m3, r1 = ctx.ind(s, "mom_accel"), ctx.ind(s, "mom_3m"), ctx.ind(s, "ret_21d")
        deteriorating = m3 is not None and r1 is not None and m3 > 0 and r1 < 0 and (accel or 0) < 0
        if accel is not None:
            ev.append(
                Evidence(
                    "mom_accel",
                    round(accel, 4),
                    f"last month {accel:+.1%} vs the 3-month pace",
                    direction=int(accel > 0) - int(accel < 0),
                    strength=0.4,
                )
            )
        if deteriorating:
            score = clamp(score - 0.2)
            ev.append(
                Evidence(
                    "deterioration",
                    True,
                    "3-month trend up but the last month down and slowing",
                    direction=-1,
                    strength=0.6,
                )
            )
        signs = [v for v in row if abs(v) > 0.1]
        agree = sum(1 for v in signs if (v > 0) == (score > 0)) / len(signs) if signs and score else 0.0
        confidence = 0.2 + 0.6 * agree
        if len(ctx.indicators) < 40:
            confidence *= 0.8  # small cross-sections rank noisily
        return Opinion(
            agent_id=self.spec.id,
            agent_version=self.spec.version,
            subject=s,
            stance=stance_of(score),
            score=score,
            confidence=confidence,
            horizon_days=self.spec.horizon_days,
            thesis=f"{s} momentum {stance_of(score).value}: average z {row.mean():+.2f} across {len(row)} horizons"
            + (", deteriorating" if deteriorating else ", accelerating" if (accel or 0) > 0.02 else ""),
            evidence=ev,
            data_used=["daily bars (whole universe)"],
            data_missing=[c for c in HORIZONS if c not in row.index],
            data_quality=ctx.state(s),
            invalidation="relative strength rank falls into the bottom half of the universe",
            meta={"universe": len(z), "deteriorating": deteriorating},
        )
