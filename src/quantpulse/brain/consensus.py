"""Consensus: combine the forecasting agents' views on one subject — without hiding their disagreement.

Each directional or neutral vote is weighted by

* the agent's own **confidence**;
* the **quality of the data** behind it (fresh/live 1.0, market closed 0.9, stale 0.6, missing 0.3);
* the agent's **measured reliability** from :class:`ReliabilityBook` — only once it has at least
  ``min_observations`` evaluated predictions; until then it is *unproven* and weighs 1.0 (no invented
  track record, in either direction).

The result keeps the sides visible: supporting / neutral / opposing / abstaining counts, a disagreement
measure (0 unanimous … 1 evenly split), the strongest voice on each side, and the vetoes raised by
constraint agents (data quality). With no or too little evidence, heavy disagreement or low confidence the
consensus is *unknown*: "I do not know" is a legitimate answer and produces no action.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .types import DataState, Opinion, Stance, stance_of, worst_state

QUALITY_FACTOR = {
    DataState.FRESH: 1.0,
    DataState.LIVE: 1.0,
    DataState.MARKET_CLOSED: 0.9,
    DataState.STALE: 0.6,
    DataState.UNAVAILABLE: 0.3,
    DataState.PROVIDER_ERROR: 0.3,
}
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
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
        }


def build_consensus(
    subject: str,
    forecasts: Sequence[Opinion],
    constraints: Sequence[Opinion] = (),
    reliability: ReliabilityBook | None = None,
    regime: str | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> Consensus:
    book = reliability or ReliabilityBook()
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
        votes.append(
            Vote(o.agent_id, o.agent_version, o.stance, o.score, o.confidence, weight, rel, o.thesis)
        )

    reasons: list[str] = []
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
        )

    score = sum(v.weight * v.score for v in votes) / total
    spread = sum(v.weight * abs(v.score) for v in votes)
    disagreement = 1.0 - abs(sum(v.weight * v.score for v in votes)) / spread if spread > 0 else 0.0
    stance = stance_of(score, 0.1)
    # sides relative to the weighted lean — even when it is too weak to be a stance, a split stays visible
    lean = 1.0 if score >= 0.0 else -1.0
    supporting = sum(1 for v in votes if v.score * lean >= 0.15)
    opposing = sum(1 for v in votes if v.score * lean <= -0.15)
    neutral = len(votes) - supporting - opposing
    mean_conf = sum(v.weight * v.confidence for v in votes) / total
    coverage = min(1.0, len(votes) / 2.0)
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
    )
