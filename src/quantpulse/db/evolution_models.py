"""Tables of market evolution and the model registry (revision 0024).

All append-only in spirit: a metric is recorded per day, a change is recorded when detected (its status may
move from open to confirmed or faded), every relationship estimate is a new row (its history is the record),
and a model version's stage history grows but is never rewritten.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from quantpulse.db.base import Base, UTCDateTime


class EvolutionMetricRow(Base):
    __tablename__ = "evolution_metrics"
    __table_args__ = (
        UniqueConstraint("day", "dimension", "subject", "metric", "timescale"),
        Index("ix_evolution_metrics_series", "dimension", "subject", "metric", "timescale"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    dimension: Mapped[str] = mapped_column(String(24))
    subject: Mapped[str] = mapped_column(String(64))
    metric: Mapped[str] = mapped_column(String(48))
    timescale: Mapped[str] = mapped_column(String(12))
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    n: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(24))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())


class EvolutionChangeRow(Base):
    __tablename__ = "evolution_changes"
    __table_args__ = (Index("ix_evolution_changes_series", "dimension", "subject", "metric"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    detected_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    dimension: Mapped[str] = mapped_column(String(24))
    subject: Mapped[str] = mapped_column(String(64))
    metric: Mapped[str] = mapped_column(String(48))
    timescale: Mapped[str] = mapped_column(String(12))
    reference_start: Mapped[date] = mapped_column(Date)
    reference_end: Mapped[date] = mapped_column(Date)
    recent_start: Mapped[date] = mapped_column(Date)
    recent_end: Mapped[date] = mapped_column(Date)
    kind: Mapped[str] = mapped_column(String(24))
    effect_sd: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float] = mapped_column(Float)
    q_value: Mapped[float] = mapped_column(Float)
    significant: Mapped[bool] = mapped_column(Boolean)
    persisted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    change_points: Mapped[list[Any]] = mapped_column(JSON, default=list)
    test: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    hypotheses_summary: Mapped[str] = mapped_column(Text, default="")
    revalidation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # open | confirmed (persisted) | faded (did not persist)
    status: Mapped[str] = mapped_column(String(16), default="open")


class EvolutionHypothesisRow(Base):
    __tablename__ = "evolution_hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    change_id: Mapped[int] = mapped_column(ForeignKey("evolution_changes.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(32))
    statement: Mapped[str] = mapped_column(Text)
    predicts: Mapped[str] = mapped_column(Text)
    # consistent | inconsistent | untestable — never "proven"
    verdict: Mapped[str] = mapped_column(String(16))
    detail: Mapped[str] = mapped_column(Text)
    identifiable_from_prices: Mapped[bool] = mapped_column(Boolean, default=True)
    evaluated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class EvolutionRelationshipRow(Base):
    """One estimate of one relationship on one window: the history is every row, never an update."""

    __tablename__ = "evolution_relationships"
    __table_args__ = (Index("ix_evolution_relationships_key", "key", "subject", "window_end"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(48))
    subject: Mapped[str] = mapped_column(String(64))
    window_start: Mapped[date] = mapped_column(Date)
    window_end: Mapped[date] = mapped_column(Date)
    slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    se: Mapped[float | None] = mapped_column(Float, nullable=True)
    r: Mapped[float | None] = mapped_column(Float, nullable=True)
    n: Mapped[int] = mapped_column(Integer)
    # STABLE | STRENGTHENED | WEAKENED | DISAPPEARED | INVERTED | EMERGED | INSUFFICIENT | ESTABLISHED
    status: Mapped[str] = mapped_column(String(16))
    z: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ModelRegistryRow(Base):
    """A model version in a slot. Its stage history only grows; the champion of a slot has role 'champion'."""

    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("slot", "version"),
        Index("ix_model_registry_slot_stage", "slot", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slot: Mapped[str] = mapped_column(String(48))
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))  # statistical | ml | ai | rule
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # what it was fitted on (period, universe, hash)
    stage: Mapped[str] = mapped_column(String(24), default="CANDIDATE")
    stage_history: Mapped[list[Any]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict
    )  # out-of-sample, walk-forward, stress, shadow
    in_sample: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # recorded, never used to promote
    role: Mapped[str] = mapped_column(String(12), default="none")  # champion | challenger | none
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
