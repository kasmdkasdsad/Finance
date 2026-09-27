"""Adversarial review: a bull case, a bear case and a devil's advocate for every subject with a consensus.

This is deterministic argument assembly, not text generation. The **bull case** and **bear case** are the
strongest pieces of evidence on each side, taken from every agent that spoke about the subject (the
specialists, research and constraints), weighted by the agent's confidence and the evidence's strength,
plus the risks that do not come from any single forecast (vetoes, an earnings release inside the horizon,
a volatility regime, a value-trap flag). The **devil's advocate** then attacks the *leading* side with a
fixed set of objections that are known ways a consensus goes wrong:

* one idea counted several times (agents that share one source of evidence and nobody else);
* a single voice, or strong opposition from a credible agent;
* chasing an extended move (or selling into an oversold one);
* an earnings release inside the forecast horizon;
* fighting the market regime;
* data that is not live;
* a short-horizon view against a long-horizon one;
* a value trap;
* nobody has a measured track record yet (noted, no penalty — that is every agent today).

Each objection carries a severity and a confidence haircut. The consensus confidence is reduced by the
product of the haircuts; a *high*-severity objection makes the view **challenged**, and the decision step
will not open a new position on a challenged view.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .consensus import Consensus
from .context import BrainContext
from .types import EXECUTABLE_STATES, DataState, Opinion, Stance

MAX_ARGS = 5


@dataclass
class Argument:
    agent_id: str
    text: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "text": self.text, "weight": round(self.weight, 3)}


@dataclass
class Objection:
    code: str
    text: str
    severity: str  # low | medium | high
    haircut: float  # multiplier on the consensus confidence

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "text": self.text, "severity": self.severity, "haircut": self.haircut}


@dataclass
class Debate:
    subject: str
    stance_before: str
    confidence_before: float
    bull: list[Argument] = field(default_factory=list)
    bear: list[Argument] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    confidence_after: float = 0.0
    verdict: str = "stands"
    change_our_mind: list[str] = field(default_factory=list)

    @property
    def challenged(self) -> bool:
        return self.verdict == "challenged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "stance_before": self.stance_before,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "verdict": self.verdict,
            "bull": [a.to_dict() for a in self.bull],
            "bear": [a.to_dict() for a in self.bear],
            "objections": [o.to_dict() for o in self.objections],
            "change_our_mind": self.change_our_mind,
        }


def _arguments(opinions: Sequence[Opinion]) -> tuple[list[Argument], list[Argument]]:
    bull: list[Argument] = []
    bear: list[Argument] = []
    for o in opinions:
        scale = o.confidence if o.stance is not Stance.ABSTAIN else 0.5
        for e in o.evidence:
            if e.direction == 0:
                continue
            arg = Argument(o.agent_id, e.detail, scale * e.strength)
            (bull if e.direction > 0 else bear).append(arg)
        if o.veto:
            bear.append(Argument(o.agent_id, f"veto: {o.veto}", 1.0))
    return bull, bear


def _risks(ctx: BrainContext, subject: str, horizon: int) -> list[Argument]:
    out: list[Argument] = []
    event = (ctx.working.facts.get("event_risk") or {}).get(subject) or {}
    days = event.get("days_to_earnings")
    if days is not None and 0 <= days <= horizon:
        move = event.get("typical_move")
        out.append(
            Argument(
                "catalyst", f"earnings in {days} days" + (f" (typical move ±{move:.1%})" if move else ""), 0.7
            )
        )
    for o in ctx.working.opinions.get(subject, []):
        if o.agent_id == "volatility" and o.meta.get("regime") in ("elevated", "extreme"):
            out.append(
                Argument(
                    "volatility",
                    f"{o.meta['regime']} volatility ({o.meta['forecast_vol']:.0%} forecast)",
                    0.5,
                )
            )
        if o.agent_id == "valuation":
            out.extend(Argument("valuation", c, 0.5) for c in o.meta.get("checks", []) if "trap" in c)
    return out


def _top(args: list[Argument]) -> list[Argument]:
    seen: set[str] = set()
    out: list[Argument] = []
    for a in sorted(args, key=lambda a: -a.weight):
        if a.text not in seen:
            seen.add(a.text)
            out.append(a)
    return out[:MAX_ARGS]


def _objections(
    ctx: BrainContext, c: Consensus, opinions: Sequence[Opinion], horizon: int
) -> list[Objection]:
    lead = 1 if c.score > 0 else -1
    side = "bullish" if lead > 0 else "bearish"
    votes = [v for v in c.votes if abs(v.score) >= 0.15]
    supporters = [v for v in votes if (v.score > 0) == (lead > 0)]
    opponents = [v for v in votes if (v.score > 0) != (lead > 0)]
    out: list[Objection] = []
    sources = {v.source or v.agent_id for v in supporters}
    if len(supporters) >= 2 and len(sources) == 1:
        # already counted once in the consensus (one source = one piece of evidence): no further haircut
        out.append(
            Objection(
                "one_idea",
                f"the case rests on one source of evidence ({next(iter(sources))}) seen by several agents",
                "medium",
                1.0,
            )
        )
    if len(supporters) == 1:
        out.append(
            Objection("single_voice", f"only {supporters[0].agent_id} makes the {side} case", "high", 0.7)
        )
    if supporters and opponents:
        top_for = max(v.weight * abs(v.score) for v in supporters)
        top_against = max(v.weight * abs(v.score) for v in opponents)
        if top_against >= 0.6 * top_for:
            who = max(opponents, key=lambda v: v.weight * abs(v.score)).agent_id
            out.append(Objection("strong_opposition", f"{who} argues credibly the other way", "medium", 0.85))
    rsi, stretch = ctx.ind(c.subject, "rsi14"), ctx.ind(c.subject, "px_vs_sma50")
    if lead > 0 and ((rsi or 0) >= 75 or (stretch or 0) >= 0.15):
        out.append(
            Objection(
                "extended",
                f"chasing an extended move (RSI {rsi or 0:.0f}, {stretch or 0:+.0%} vs 50-day)",
                "medium",
                0.9,
            )
        )
    if lead < 0 and ((rsi or 100) <= 25 or (stretch or 0) <= -0.15):
        out.append(
            Objection("oversold", f"selling into an oversold market (RSI {rsi or 0:.0f})", "medium", 0.9)
        )
    days = ((ctx.working.facts.get("event_risk") or {}).get(c.subject) or {}).get("days_to_earnings")
    if days is not None and 0 <= days <= horizon:
        out.append(
            Objection(
                "event_in_horizon",
                f"an earnings release in {days} days falls inside the {horizon}-day view",
                "medium",
                0.85,
            )
        )
    regime = ctx.working.facts.get("regime")
    if (lead > 0 and regime in ("bearish", "risk_off")) or (lead < 0 and regime == "bullish"):
        out.append(Objection("against_regime", f"a {side} view against a {regime} market", "medium", 0.85))
    if c.data_quality not in EXECUTABLE_STATES:
        closed = c.data_quality is DataState.MARKET_CLOSED
        out.append(
            Objection(
                "data", f"data is {c.data_quality.value}", "low" if closed else "high", 1.0 if closed else 0.6
            )
        )
    short = [
        v for v in supporters if next((o.horizon_days for o in opinions if o.agent_id == v.agent_id), 0) <= 5
    ]
    long_against = [
        v for v in opponents if next((o.horizon_days for o in opinions if o.agent_id == v.agent_id), 0) >= 63
    ]
    if supporters and len(short) == len(supporters) and long_against:
        out.append(
            Objection(
                "horizon",
                f"a short-term {side} case against a long-term view from {long_against[0].agent_id}",
                "low",
                0.95,
            )
        )
    if lead > 0 and any(
        "trap" in chk for o in opinions if o.agent_id == "valuation" for chk in o.meta.get("checks", [])
    ):
        out.append(Objection("value_trap", "the valuation agent suspects a value trap", "medium", 0.85))
    if c.votes and all(v.reliability.status == "unproven" for v in c.votes):
        out.append(Objection("unproven", "no agent here has a measured track record yet", "low", 1.0))
    return out


def review(ctx: BrainContext, consensus: dict[str, Consensus]) -> dict[str, Debate]:
    """Debate every subject with a directional consensus; adjust its confidence in place."""
    horizons = ctx.working.facts.get("horizons") or {}
    out: dict[str, Debate] = {}
    for subject, c in consensus.items():
        if subject.startswith("@"):
            continue
        opinions = ctx.working.opinions.get(subject, [])
        horizon = int(horizons.get(subject, 5))
        bull, bear = _arguments(opinions)
        bear.extend(_risks(ctx, subject, horizon))
        d = Debate(subject, "unknown" if c.unknown else c.stance.value, c.confidence, _top(bull), _top(bear))
        if c.unknown or c.stance is Stance.NEUTRAL:
            d.confidence_after, d.verdict = c.confidence, "no view to challenge"
            out[subject] = d
            continue
        d.objections = _objections(ctx, c, opinions, horizon)
        haircut = 1.0
        for o in d.objections:
            haircut *= o.haircut
        d.confidence_after = c.confidence * haircut
        severities = {o.severity for o in d.objections}
        d.verdict = (
            "challenged" if "high" in severities else "weakened" if "medium" in severities else "stands"
        )
        lead = 1 if c.score > 0 else -1
        d.change_our_mind = [
            o.invalidation
            for o in opinions
            if o.invalidation and o.directional and (o.score > 0) == (lead > 0)
        ][:4]
        c.confidence = d.confidence_after
        serious = [o.text for o in d.objections if o.severity != "low"]
        c.reasons.append(
            f"devil's advocate: {d.verdict}"
            + (f" ({'; '.join(serious)})" if serious else "")
            + f", confidence {d.confidence_before:.2f} → {d.confidence_after:.2f}"
        )
        out[subject] = d
    return out
