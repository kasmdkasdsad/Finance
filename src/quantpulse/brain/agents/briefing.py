"""Analyst briefing: the one model-backed agent — a short written summary of what the team found.

It runs last (stage 2), after the specialists and the research checklist, and only when a language model
is configured with budget left; otherwise it skips itself with the reason. For the few focus symbols with
the strongest views it hands the model the findings already in working memory — each agent's stance,
confidence and thesis, the research answers, the opportunity that raised the idea — and asks for a
summary, the points for and against, where the agents conflict, and what to watch, as structured JSON.

The model computes nothing and decides nothing. It is told to use only the facts given; its briefing is
context for the dashboard and the record: it casts no vote, is never graded as a forecast, and no number in
it reaches sizing or the risk engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ..context import BrainContext
from ..llm import ModelRequest, ModelTask
from ..types import AgentFamily, AgentSpec, ModelTier, Opinion
from .base import Agent, Role, symbols_only
from .common import opinion, symbol_quality

SYSTEM = (
    "You write short briefings for the research log of a paper-trading system. Use only the facts in the "
    "message: do not add facts, prices or news, and do not calculate, estimate or recommend numbers, "
    "position sizes or trades. Where the agents disagree, say so plainly. Answer with JSON only."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 700},
        "supporting": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "opposing": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "conflicts": {"type": "string"},
        "watch": {"type": "string"},
    },
    "required": ["summary", "supporting", "opposing", "watch"],
    "additionalProperties": False,
}


def _clip(text: str, n: int = 170) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def facts(ctx: BrainContext, s: str) -> str:
    """The findings the model may use, as plain lines (everything here was computed by the agents)."""
    lines = [f"Symbol: {s}" + (f" (sector: {ctx.sectors[s]})" if s in ctx.sectors else "")]
    lines.append(
        f"Data state: {ctx.state(s).value}; held in the paper account: {'yes' if s in ctx.held else 'no'}"
    )
    for o in ctx.working.opinions.get(s, []):
        if o.agent_id == "briefing":
            continue
        if o.directional:
            lines.append(
                f"- {o.agent_id}: {o.stance.value}, score {o.score:+.2f}, confidence {o.confidence:.2f}, "
                f"{o.horizon_days}-day view. {_clip(o.thesis)}"
            )
        elif o.agent_id == "research":
            for f in (o.meta.get("findings") or [])[:8]:
                lines.append(
                    f"- research: {_clip(str(f.get('question')), 80)} {_clip(str(f.get('answer')), 120)}"
                )
        elif o.veto:
            lines.append(f"- {o.agent_id}: veto — {_clip(o.veto)}")
    for opp in ctx.opportunities:
        if s in opp.symbols[:1]:
            lines.append(f"- detected opportunity ({opp.kind}): {_clip(opp.headline)}")
    return "\n".join(lines)


def strength(ctx: BrainContext, s: str) -> float:
    views = [o for o in ctx.working.opinions.get(s, []) if o.directional]
    return abs(sum(o.score * o.confidence for o in views)) + (1.0 if s in ctx.held else 0.0)


class BriefingAgent(Agent):
    spec = AgentSpec(
        id="briefing",
        name="Analyst briefing",
        description="A language model's short written summary of the team's findings on the strongest ideas "
        "(context only: no vote, no numbers used). Skips itself when no model is configured.",
        family=AgentFamily.META,
        capabilities=("narrative_summary",),
        inputs=("working_memory",),
        subjects=("symbol",),
        cost=25.0,
        priority=90,
        model_tier=ModelTier.FAST,
        horizon_days=5,
        stage=2,
    )
    role: ClassVar[Role] = "context"

    def unavailable(self, ctx: BrainContext) -> str | None:
        if ctx.llm is None:
            return "no language model is attached to the brain"
        if ctx.llm.max_briefings <= 0:
            return "briefings are turned off (QP_BRAIN_LLM_MAX_BRIEFINGS=0)"
        return ctx.llm.unavailable(self.spec.model_tier)

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        assert ctx.llm is not None
        ranked = sorted(symbols_only(subjects), key=lambda s: -strength(ctx, s))
        chosen = [s for s in ranked if ctx.working.opinions.get(s)][: ctx.llm.max_briefings]
        out: list[Opinion] = []
        for s in chosen:  # the router limits concurrency; a cached answer costs nothing
            request = ModelRequest.ask(
                ModelTask.SUMMARIZE,
                SYSTEM,
                "Brief the team on this idea in at most 80 words, then list up to three points for and "
                "three against, any conflict between agents, and one thing to watch.\n\n" + facts(ctx, s),
                schema=SCHEMA,
                max_tokens=500,
                purpose=self.spec.id,
            )
            result = await ctx.llm.complete(request)
            if not result.ok:
                out.append(self.abstain(s, f"no briefing: {result.reason}"))
                continue
            brief = result.parsed
            out.append(
                opinion(
                    self,
                    s,
                    0.0,
                    0.0,
                    f"briefing ({result.model}): {brief['summary']}",
                    [],
                    quality=symbol_quality(ctx, s),
                    used=["the agents' findings this cycle"],
                    meta={
                        "briefing": brief,
                        "model": result.model,
                        "status": result.status,
                        "source": "language model",
                        "note": "context only: casts no vote and sets no number",
                    },
                    directional=False,
                )
            )
        return out
