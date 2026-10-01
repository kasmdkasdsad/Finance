"""The Brain's 24/7 research subsystem: the persistent research queue (every job and its result), the learning
ledger (conclusions with their evidence) and the improvement lifecycle (hypotheses that only a person can promote).
See quantpulse.brain.research."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Date, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from quantpulse.db.base import Base, UTCDateTime


class BrainResearchJobRow(Base):
    """One research question and its run: queued, running, done, failed or cancelled. Kept for good: the
    experiment history."""

    __tablename__ = "brain_research_jobs"
    __table_args__ = (Index("ix_brain_research_jobs_queue", "status", "priority"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    key: Mapped[str] = mapped_column(String(160), index=True)  # the same question is never queued twice
    question: Mapped[str] = mapped_column(Text)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(16))  # system | follow_up | person | event
    parent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[str] = mapped_column(String(8))  # light | medium | heavy
    priority: Mapped[float] = mapped_column(Float)
    priority_detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(12))  # queued | running | done | failed | cancelled
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    not_before: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    holder: Mapped[str | None] = mapped_column(String(96), nullable=True)  # the process running it
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peak_rss_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class BrainLearningRow(Base):
    """A conclusion and its evidence. Its status is computed from the evidence (never asserted): UNPROVEN while
    the sample is too small or untested."""

    __tablename__ = "brain_learnings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(40))
    topic: Mapped[str] = mapped_column(String(160), index=True)
    claim: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))  # UNPROVEN | SUPPORTED | REFUTED | INCONCLUSIVE
    sample_size: Mapped[int] = mapped_column(Integer)
    min_sample: Mapped[int] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    regime: Mapped[str] = mapped_column(String(48))
    benchmark: Mapped[str] = mapped_column(String(64))
    method: Mapped[str] = mapped_column(String(96))
    statistics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float)
    limitations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    supersedes_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BrainHypothesisRow(Base):
    """An improvement on its way through DISCOVERED → HYPOTHESIS → BACKTEST → WALK_FORWARD → STRESS_TEST →
    PAPER_SHADOW → EVALUATION → PRODUCTION, one stage at a time; PRODUCTION only by a person."""

    __tablename__ = "brain_hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(160), unique=True)
    kind: Mapped[str] = mapped_column(String(32))  # strategy | feature | agent_combination | process
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(24))  # research | improvement | lab | person
    source_ref: Mapped[str | None] = mapped_column(String(96), nullable=True)
    stage: Mapped[str] = mapped_column(String(20), index=True)
    protected_control: Mapped[str | None] = mapped_column(String(48), nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    history: Mapped[list[Any]] = mapped_column(JSON, default=list)
    next_step_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(48), nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
