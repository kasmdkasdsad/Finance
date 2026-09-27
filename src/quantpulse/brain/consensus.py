"""Consensus: combine the forecasting agents' views on one subject — without hiding their disagreement.

Each directional or neutral vote is weighted by

* the agent's own **confidence**;
* the **quality of the data** behind it (fresh/live 1.0, market closed 0.9, stale 0.6, missing 0.3);
* the agent's **measured reliability** from :class:`ReliabilityBook` — only once it has at least
  ``min_observations`` evaluated predictions; until then it is *unproven* and weighs 1.0 (no invented
  track record, in either direction).

Agents that rest on the same information are **one piece of evidence**, however many of them agree. Each
agent declares its ``source`` (prices, fundamentals, the stock model, options, events, promoted
strategies); the score is a weighted mean over sources (a source weighs as much as its strongest voice and
says what its agents say on average), and full confidence needs at least two independent sources — four
price-based agents agreeing is one idea seen four ways, not four confirmations. Disagreement is the larger
of the split between agents and the split between sources, so a conflict inside one source stays visible.

The result keeps the sides visible: supporting / neutral / opposing / abstaining counts, a disagreement
measure (0 unanimous … 1 evenly split), the strongest voice on each side, the vetoes raised by constraint
agents (data quality), the agents that gave no view and why, and the reasons for uncertainty. With no or
too little evidence, heavy disagreement or low confidence the consensus is *unknown*: "I do not know" is a
legitimate answer and produces no action.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import EXECUTABLE_STATES, DataState, Opinion, Stance, stance_of, worst_state

QUALITY_FACTOR = {
    DataState.FRESH: 1.0,
    DataState.LIVE: 1.0,
    DataState.MARKET_CLOSED: 0.9,
    DataState.STALE: 0.6,
    DataState.UNAVAILABLE: 0.3,
    DataState.PROVIDER_ERROR: 0.3,
    DataState.INVALID: 0.3,
}
# the consensus is graded like an agent: its version changes with its method, so records never mix
# (1: every agent a separate vote; 2: agents sharing a source of information count once)
CONSENSUS_VERSION = "2"
MAX_DISAGREEMENT = 0.5
MIN_CONFIDENCE = 0.2


@dataclass(frozen=True, slots=True)
class Reliability:
    weight: float
    status: str  # "unproven" | "measured"
    n: int


class ReliabilityBook:
    """Measured reliability per (agent, version, regime). Empty until predictions have been evaluated —
    and then only agents with enough observations get a weight other than 1.0."""

    def __init__(self, rows: Iterable[dict[str, Any]] = (), min_observations: int = 30) -> None:
        self.min_observations = min_observations
        self._rows = {(r["agent_id"], r["agent_version"], r.get("regime", "all")): r for r in rows}

    def get(self, agent_id: str, version: str, regime: str | None = None) -> Reliability:
        row = self._rows.get((agent_id, version, regime or "all")) or self._rows.get(
            (agent_id, version, "all")
        )
        n = int(row["n"]) if row else 0
        if row is None or n < self.min_observations or row.get("reliability") is None:
            return Reliability(1.0, "unproven", n)
        return Reliability(float(row["reliability"]), "measured", n)


@dataclass
class Vote:
    agent_id: str
    version: str
    stance: Stance
    score: float
    confidence: float
    weight: float
    reliability: Reliability
    thesis: str
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "source": self.source,
            "version": self.version,
            "stance": self.stance.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "weight": round(self.weight, 4),
            "reliability": {
                "weight": self.reliability.weight,
                "status": self.reliability.status,
                "n": self.reliability.n,
            },
            "thesis": self.thesis,
        }


@dataclass
class Consensus:
    subject: str
    stance: Stance
    score: float
    confidence: float
    unknown: bool
    supporting: int
    neutral: int
    opposing: int
    abstaining: int
    disagreement: float
    votes: list[Vote] = field(default_factory=list)
    primary_disagreement: dict[str, Any] | None = None
    vetoes: list[dict[str, str]] = field(default_factory=list)
    data_quality: DataState = DataState.LIVE
    reasons: list[str] = field(default_factory=list)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)  # evidence per information source
    independent: int = 0  # sources with a directional view
    missing: list[dict[str, str]] = field(default_factory=list)  # forecasting agents with no view, and why
    uncertainty: list[str] = field(default_factory=list)

    @property
    def actionable_view(self) -> bool:
        return not self.unknown and self.stance in (Stance.BULLISH, Stance.BEARISH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "stance": "unknown" if self.unknown else self.stance.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "unknown": self.unknown,
            "supporting": self.supporting,
            "neutral": self.neutral,
            "opposing": self.opposing,
            "abstaining": self.abstaining,
            "disagreement": round(self.disagreement, 4),
            "votes": [v.to_dict() for v in self.votes],
            "primary_disagreement": self.primary_disagreement,
            "vetoes": self.vetoes,
            "data_quality": self.data_quality.value,
            "reasons": self.reasons,
            "sources": self.sources,
            "independent_sources": self.independent,
            "missing": self.missing,
            "uncertainty": self.uncertainty,
        }


def _split(items: Sequence[tuple[float, float]]) -> float:
    """0 when every weighted score points the same way … 1 when they cancel out."""
    spread = sum(w * abs(x) for w, x in items)
    return 1.0 - abs(sum(w * x for w, x in items)) / spread if spread > 0 else 0.0


def build_consensus(
    subject: str,
    forecasts: Sequence[Opinion],
    constraints: Sequence[Opinion] = (),
    reliability: ReliabilityBook | None = None,
    regime: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    sources: Mapping[str, str] | None = None,
    missing: Sequence[dict[str, str]] = (),
) -> Consensus:
    """``sources`` maps agent ids to the information they rest on (an agent without one is its own
    source); ``missing`` lists the forecasting agents that gave no view on this subject, with the reason."""
    book = reliability or ReliabilityBook()
    source_of = dict(sources or {})
    voters = [o for o in forecasts if o.stance is not Stance.ABSTAIN]
    abstaining = len(forecasts) - len(voters)
    vetoes = [{"agent_id": o.agent_id, "reason": o.veto} for o in constraints if o.veto]
    quality = (
        worst_state([o.data_quality for o in [*voters, *constraints]])
        if (voters or constraints)
        else DataState.UNAVAILABLE
    )

    votes: list[Vote] = []
    for o in voters:
        rel = book.get(o.agent_id, o.agent_version, regime)
        weight = o.confidence * QUALITY_FACTOR.get(o.data_quality, 0.5) * rel.weight
        src = source_of.get(o.agent_id) or o.agent_id
        votes.append(
            Vote(o.agent_id, o.agent_version, o.stance, o.score, o.confidence, weight, rel, o.thesis, src)
        )

    reasons: list[str] = []
    missing = list(missing)
    total = sum(v.weight for v in votes)
    if not votes or total <= 0:
        reasons.append("no agent had a view" if not voters else "the views carry no weight")
        return Consensus(
            subject,
            Stance.NEUTRAL,
            0.0,
            0.0,
            True,
            0,
            0,
            0,
            abstaining,
            0.0,
            votes,
            None,
            vetoes,
            quality,
            reasons,
            missing=missing,
            uncertainty=[*reasons, *_missing_note(missing)],
        )

    # one piece of evidence per information source
    groups: dict[str, list[Vote]] = {}
    for v in votes:
        groups.setdefault(v.source, []).append(v)
    by_source: dict[str, dict[str, Any]] = {}
    for src, members in groups.items():
        w = sum(v.weight for v in members)
        by_source[src] = {
            "score": sum(v.weight * v.score for v in members) / w if w > 0 else 0.0,
            "weight": max(v.weight for v in members),
            "confidence": sum(v.weight * v.confidence for v in members) / w if w > 0 else 0.0,
            "agents": [v.agent_id for v in members],
        }
    source_total = sum(g["weight"] for g in by_source.values())
    score = sum(g["weight"] * g["score"] for g in by_source.values()) / source_total
    disagreement = max(
        _split([(v.weight, v.score) for v in votes]),
        _split([(g["weight"], g["score"]) for g in by_source.values()]),
    )
    independent = sum(1 for g in by_source.values() if abs(g["score"]) >= 0.15)
    stance = stance_of(score, 0.1)
    # sides relative to the weighted lean — even when it is too weak to be a stance, a split stays visible
    lean = 1.0 if score >= 0.0 else -1.0
    supporting = sum(1 for v in votes if v.score * lean >= 0.15)
    opposing = sum(1 for v in votes if v.score * lean <= -0.15)
    neutral = len(votes) - supporting - opposing
    mean_conf = sum(g["weight"] * g["confidence"] for g in by_source.values()) / source_total
    coverage = min(1.0, max(independent, 1 if votes else 0) / 2.0)
    confidence = mean_conf * (1.0 - disagreement) * coverage * QUALITY_FACTOR.get(quality, 0.5)

    bulls = [v for v in votes if v.score > 0.15]
    bears = [v for v in votes if v.score < -0.15]
    primary = None
    if bulls and bears:
        top_bull = max(bulls, key=lambda v: v.weight * v.score)
        top_bear = min(bears, key=lambda v: v.weight * v.score)
        primary = {
            "for": {
                "agent_id": top_bull.agent_id,
                "score": round(top_bull.score, 3),
                "thesis": top_bull.thesis,
            },
            "against": {
                "agent_id": top_bear.agent_id,
                "score": round(top_bear.score, 3),
                "thesis": top_bear.thesis,
            },
            "summary": f"{top_bull.agent_id} vs {top_bear.agent_id}",
        }

    unknown = False
    if len(votes) == 1 and votes[0].confidence < 0.5:
        unknown = True
        reasons.append(f"only one view ({votes[0].agent_id}) and it is not confident")
    if disagreement > MAX_DISAGREEMENT:
        unknown = True
        reasons.append(f"agents disagree (disagreement {disagreement:.2f})")
    if confidence < min_confidence:
        unknown = True
        reasons.append(f"combined confidence {confidence:.2f} is too low")
    reasons.append(f"{supporting} supporting, {neutral} neutral, {opposing} opposing")

    uncertainty = list(reasons[:-1])
    if independent <= 1:
        only = next((s for s, g in by_source.items() if abs(g["score"]) >= 0.15), None) or "none"
        uncertainty.append(
            f"evidence from one source only ({only}): agents that share it count once"
            if independent == 1
            else "no source has a directional view"
        )
    if votes and all(v.reliability.status == "unproven" for v in votes):
        uncertainty.append("no voting agent has a measured track record yet (all unproven)")
    if quality not in EXECUTABLE_STATES:
        uncertainty.append(f"data {quality.value}")
    if 0.3 < disagreement <= MAX_DISAGREEMENT:
        uncertainty.append(f"partial disagreement ({disagreement:.2f})")
    uncertainty.extend(_missing_note(missing))
    return Consensus(
        subject=subject,
        stance=Stance.NEUTRAL if unknown else stance,
        score=score,
        confidence=confidence,
        unknown=unknown,
        supporting=supporting,
        neutral=neutral,
        opposing=opposing,
        abstaining=abstaining,
        disagreement=disagreement,
        votes=votes,
        primary_disagreement=primary,
        vetoes=vetoes,
        data_quality=quality,
        reasons=reasons,
        sources={src: _rounded(g) for src, g in by_source.items()},
        independent=independent,
        missing=missing,
        uncertainty=uncertainty,
    )


def _rounded(g: dict[str, Any]) -> dict[str, Any]:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in g.items()}


def _missing_note(missing: Sequence[dict[str, str]]) -> list[str]:
    gone = [m for m in missing if m.get("kind") != "abstained"]
    if not gone:
        return []
    names = ", ".join(f"{m['agent_id']} ({m['reason'][:60]})" for m in gone[:4])
    return [f"{len(gone)} forecasting agent(s) gave no view: {names}"]
