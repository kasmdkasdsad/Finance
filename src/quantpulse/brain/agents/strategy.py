"""Strategy-lab agent: the voice of strategies a person has promoted after validation and paper tracking.

Each promoted strategy contributes its current ranking (published by the lab when it updates its paper
portfolios): a focus symbol in its top ``top_n`` is a bullish vote, in its bottom ``top_n`` a bearish one.
Strategies that have not been validated, paper-tracked and promoted have no voice here, and this agent's
own calls are graded like every other agent's.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import opinion, symbol_quality


class StrategyLabAgent(Agent):
    spec = AgentSpec(
        id="strategy_lab",
        source="strategies",
        failure="skips while no strategy is promoted",
        name="Promoted strategies",
        description="Votes from strategies that passed the lab's validation, were paper-tracked and were "
        "promoted by a person.",
        family=AgentFamily.SPECIALIST,
        capabilities=("promoted_strategies",),
        inputs=("strategy_signals",),
        subjects=("symbol",),
        priority=40,
        horizon_days=21,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if ctx.strategy_signals else "no promoted strategy in the lab"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        return [self._one(ctx, s) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str) -> Opinion:
        votes: list[int] = []
        horizons: list[int] = []
        ev: list[Evidence] = []
        q = symbol_quality(ctx, s)
        for key, sig in ctx.strategy_signals.items():
            if s in sig.get("top", []):
                votes.append(1)
                ev.append(
                    Evidence(
                        key,
                        "top",
                        f"{sig.get('name', key)} ({key}) ranks it among its holdings",
                        1,
                        0.7,
                        quality=q,
                    )
                )
            elif s in sig.get("bottom", []):
                votes.append(-1)
                ev.append(
                    Evidence(
                        key,
                        "bottom",
                        f"{sig.get('name', key)} ({key}) ranks it near the bottom",
                        -1,
                        0.6,
                        quality=q,
                    )
                )
            else:
                continue
            horizons.append(int(sig.get("horizon", 21)))
        if not votes:
            return self.abstain(s, "no promoted strategy ranks it at either end")
        score = 0.5 * sum(votes) / len(votes)
        return opinion(
            self,
            s,
            score,
            0.35 + 0.1 * min(len(votes) - 1, 2),
            "; ".join(e.detail for e in ev),
            ev,
            quality=q,
            used=["promoted strategies' current rankings"],
            horizon=int(statistics.median(horizons)),
            meta={"strategies": [e.name for e in ev]},
        )
