"""The agent interface.

An agent is small and specific: it declares what it is (:class:`~quantpulse.brain.types.AgentSpec`), says
whether it can run on this cycle's context (:meth:`Agent.unavailable`), and returns structured
:class:`~quantpulse.brain.types.Opinion` objects for the subjects it is given. It never touches the broker
or the network directly — everything it needs is in the context.

``role`` separates *forecasts* (directional views that are voted on in the consensus and later graded
against what the market did) from *constraints* (data quality, portfolio fit: they veto or shape actions
but are not price forecasts and are never graded as such).

A language-model agent implements the same interface; its spec says ``model_tier`` FAST or STRONG, it
reaches the model only through the context's :class:`~quantpulse.brain.llm.ModelRouter` (budget, cache,
timeout), and it skips itself when no model is available. The briefing agent is the only one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Literal

from ..context import BrainContext
from ..types import MARKET, PORTFOLIO, AgentSpec, Opinion

Role = Literal["forecast", "constraint", "context"]


class Agent(ABC):
    spec: ClassVar[AgentSpec]
    role: ClassVar[Role] = "forecast"

    def unavailable(self, ctx: BrainContext) -> str | None:
        """Why this agent cannot run on this context (``None`` when it can)."""
        return None

    def subjects(self, ctx: BrainContext) -> list[str]:
        """The subjects this agent should look at this cycle."""
        out: list[str] = []
        if "market" in self.spec.subjects:
            out.append(MARKET)
        if "portfolio" in self.spec.subjects:
            out.append(PORTFOLIO)
        if "symbol" in self.spec.subjects:
            out.extend(ctx.focus)
        return out

    @abstractmethod
    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        """Structured opinions for ``subjects`` (abstain rather than guess when evidence is missing)."""

    # helpers ---------------------------------------------------------------------------------------
    def abstain(self, subject: str, reason: str, missing: list[str] | None = None) -> Opinion:
        return Opinion.abstain(self.spec, subject, reason, missing)


def symbols_only(subjects: Sequence[str]) -> list[str]:
    return [s for s in subjects if s not in (MARKET, PORTFOLIO)]
