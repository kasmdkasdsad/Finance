"""The brain's vocabulary: agent specs, opinions, evidence, data states, actions and modes.

Everything here is plain data (JSON-serialisable through ``to_dict``) so it can be stored, compared across
cycles and graded later.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

MARKET = "@market"  # subject of market-wide opinions
PORTFOLIO = "@portfolio"  # subject of portfolio-wide opinions


class Stance(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    ABSTAIN = "abstain"  # the agent has no view (missing data, not applicable): never counted as neutral


class DataState(StrEnum):
    """Trustworthiness of the data behind a symbol or an opinion (worst first in :data:`STATE_RANK`)."""

    FRESH = "fresh"  # a live quote seconds old
    LIVE = "live"  # a live quote within the allowed age
    STALE = "stale"  # older than allowed: analysis only, never an executable action
    MARKET_CLOSED = "market_closed"  # outside the regular session: daily data only
    UNAVAILABLE = "unavailable"  # no data
    PROVIDER_ERROR = "provider_error"  # the source failed
    INVALID = "invalid"  # timestamps that cannot be right (stamped in the future): age unknown


STATE_RANK: dict[DataState, int] = {
    DataState.INVALID: 0,
    DataState.PROVIDER_ERROR: 0,
    DataState.UNAVAILABLE: 1,
    DataState.STALE: 2,
    DataState.MARKET_CLOSED: 3,
    DataState.LIVE: 4,
    DataState.FRESH: 5,
}
EXECUTABLE_STATES = frozenset({DataState.FRESH, DataState.LIVE})


def worst_state(states: list[DataState] | tuple[DataState, ...]) -> DataState:
    return min(states, key=STATE_RANK.__getitem__) if states else DataState.UNAVAILABLE


class BrainSession(StrEnum):
    OPEN = "market_open"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


class ModelTier(StrEnum):
    DETERMINISTIC = "deterministic"  # plain Python: indicators, statistics, risk
    FAST = "fast"  # a small, cheap language model: classification, extraction, short summaries
    STRONG = "strong"  # a strong reasoning model: research synthesis, debate


class AgentFamily(StrEnum):
    PERCEPTION = "perception"
    SPECIALIST = "specialist"
    OPPORTUNITY = "opportunity"
    ADVERSARIAL = "adversarial"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    META = "meta"


class Action(StrEnum):
    BUY = "buy"
    INCREASE = "increase"
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"
    SELL = "sell"
    REBALANCE = "rebalance"
    DE_RISK = "de_risk"
    WATCH = "watch"
    NO_ACTION = "no_action"


BUYING = frozenset({Action.BUY, Action.INCREASE})
SELLING = frozenset({Action.SELL, Action.REDUCE, Action.CLOSE, Action.DE_RISK, Action.REBALANCE})  # trims


class BrainMode(StrEnum):
    RESEARCH_ONLY = "research_only"  # analysis and memory, no decisions
    DRY_RUN = "dry_run"  # decisions and risk review, clearly marked as a dry run
    PAPER_RECOMMENDATION = "paper_recommendation"  # decisions + risk review, recommended to the user
    PAPER_EXECUTION = "paper_execution"  # approved decisions are sent to the Alpaca PAPER account


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x != x or math.isinf(x):  # NaN / inf never leak into a score
        return 0.0
    return max(lo, min(hi, x))


def stance_of(score: float, threshold: float = 0.15) -> Stance:
    if score >= threshold:
        return Stance.BULLISH
    if score <= -threshold:
        return Stance.BEARISH
    return Stance.NEUTRAL


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """What an agent is, needs and costs. ``version`` changes whenever its logic or parameters change, so
    its track record is never mixed across versions."""

    id: str
    name: str
    description: str
    family: AgentFamily
    capabilities: tuple[str, ...]
    inputs: tuple[str, ...]  # BrainContext fields it reads (used to skip it when they are missing)
    outputs: tuple[str, ...] = ("opinion",)
    subjects: tuple[str, ...] = ("symbol",)  # "symbol", "market", "portfolio"
    cost: float = 1.0  # relative cost units (deterministic agents ≈ 1; LLM calls far more)
    priority: int = 50  # lower runs first when a budget forces a choice
    dependencies: tuple[str, ...] = ()
    model_tier: ModelTier = ModelTier.DETERMINISTIC
    version: str = "1.0.0"
    horizon_days: int = 5  # the horizon its directional opinions are graded on
    stage: int = 0  # 0 specialists; 1 agents that read the specialists' findings (research, situation)
    # the information it rests on: agents sharing a source are one piece of evidence in the consensus,
    # however many of them agree ("prices", "fundamentals", "model", "options", "events", "strategies", …)
    source: str = ""
    failure: str = ""  # what happens when it cannot run or fails (its charter's failure behaviour)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["family"] = self.family.value
        d["model_tier"] = self.model_tier.value
        return d


@dataclass(frozen=True, slots=True)
class Evidence:
    """One fact behind an opinion. ``direction`` +1 supports a rise (or the thesis), −1 a fall."""

    name: str
    value: float | str | bool | None
    detail: str
    direction: int = 0
    strength: float = 0.5  # 0..1, how much this fact matters to the agent
    source: str = "computed"
    quality: DataState = DataState.LIVE

    def to_dict(self) -> dict[str, Any]:
        v = self.value
        if isinstance(v, float) and (v != v or math.isinf(v)):
            v = None
        return {
            "name": self.name,
            "value": v,
            "detail": self.detail,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "source": self.source,
            "quality": self.quality.value,
        }


@dataclass(slots=True)
class Opinion:
    """An agent's structured view on one subject (a symbol, the market or the portfolio)."""

    agent_id: str
    agent_version: str
    subject: str
    stance: Stance
    score: float  # −1 (strongly bearish) … +1 (strongly bullish)
    confidence: float  # 0 … 1: the agent's own certainty given its evidence and data quality
    horizon_days: int
    thesis: str
    evidence: list[Evidence] = field(default_factory=list)
    data_used: list[str] = field(default_factory=list)
    data_missing: list[str] = field(default_factory=list)
    data_quality: DataState = DataState.LIVE
    invalidation: str | None = None  # what would prove the view wrong
    veto: str | None = None  # data-quality / risk agents: why no action may be taken on this subject
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = clamp(self.score)
        self.confidence = clamp(self.confidence, 0.0, 1.0)
        if self.stance is Stance.ABSTAIN:
            self.score, self.confidence = 0.0, 0.0

    @property
    def directional(self) -> bool:
        return self.stance in (Stance.BULLISH, Stance.BEARISH)

    @classmethod
    def abstain(cls, spec: AgentSpec, subject: str, reason: str, missing: list[str] | None = None) -> Opinion:
        return cls(
            agent_id=spec.id,
            agent_version=spec.version,
            subject=subject,
            stance=Stance.ABSTAIN,
            score=0.0,
            confidence=0.0,
            horizon_days=spec.horizon_days,
            thesis=f"No view: {reason}",
            data_missing=list(missing or []),
            data_quality=DataState.UNAVAILABLE if missing else DataState.LIVE,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "subject": self.subject,
            "stance": self.stance.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "horizon_days": self.horizon_days,
            "thesis": self.thesis,
            "evidence": [e.to_dict() for e in self.evidence],
            "data_used": self.data_used,
            "data_missing": self.data_missing,
            "data_quality": self.data_quality.value,
            "invalidation": self.invalidation,
            "veto": self.veto,
            "meta": self.meta,
        }


def finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def utc_iso(d: datetime | None) -> str | None:
    return d.isoformat() if d is not None else None
