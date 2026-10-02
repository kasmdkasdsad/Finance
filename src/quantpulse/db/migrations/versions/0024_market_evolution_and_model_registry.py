"""Market evolution (metrics, detected changes, competing hypotheses, relationship history), the versioned
model registry (see quantpulse.db.evolution_models), and option orders in the order record: the symbol
columns widened for OCC contracts (21 characters) and multi-leg record symbols, and each order's asset
class, order class, position intent and legs. Additive (widening only).

Revision ID: 0024
Revises: 0023
Create Date: 2026-09-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evolution_changes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("detected_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("dimension", sa.String(length=24), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=False),
        sa.Column("metric", sa.String(length=48), nullable=False),
        sa.Column("timescale", sa.String(length=12), nullable=False),
        sa.Column("reference_start", sa.Date(), nullable=False),
        sa.Column("reference_end", sa.Date(), nullable=False),
        sa.Column("recent_start", sa.Date(), nullable=False),
        sa.Column("recent_end", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("effect_sd", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("q_value", sa.Float(), nullable=False),
        sa.Column("significant", sa.Boolean(), nullable=False),
        sa.Column("persisted", sa.Boolean(), nullable=True),
        sa.Column("change_points", sa.JSON(), nullable=False),
        sa.Column("test", sa.JSON(), nullable=False),
        sa.Column("hypotheses_summary", sa.Text(), nullable=False),
        sa.Column("revalidation", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evolution_changes")),
    )
    with op.batch_alter_table("evolution_changes", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_evolution_changes_detected_at"), ["detected_at"], unique=False)
        batch_op.create_index("ix_evolution_changes_series", ["dimension", "subject", "metric"], unique=False)

    op.create_table(
        "evolution_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("dimension", sa.String(length=24), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=False),
        sa.Column("metric", sa.String(length=48), nullable=False),
        sa.Column("timescale", sa.String(length=12), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("recorded_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evolution_metrics")),
        sa.UniqueConstraint(
            "day",
            "dimension",
            "subject",
            "metric",
            "timescale",
            name=op.f("uq_evolution_metrics_day_dimension_subject_metric_timescale"),
        ),
    )
    with op.batch_alter_table("evolution_metrics", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_evolution_metrics_day"), ["day"], unique=False)
        batch_op.create_index(
            "ix_evolution_metrics_series", ["dimension", "subject", "metric", "timescale"], unique=False
        )

    op.create_table(
        "evolution_relationships",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=48), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=False),
        sa.Column("window_start", sa.Date(), nullable=False),
        sa.Column("window_end", sa.Date(), nullable=False),
        sa.Column("slope", sa.Float(), nullable=True),
        sa.Column("se", sa.Float(), nullable=True),
        sa.Column("r", sa.Float(), nullable=True),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("z", sa.Float(), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("recorded_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evolution_relationships")),
    )
    with op.batch_alter_table("evolution_relationships", schema=None) as batch_op:
        batch_op.create_index(
            "ix_evolution_relationships_key", ["key", "subject", "window_end"], unique=False
        )

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slot", sa.String(length=48), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("stage", sa.String(length=24), nullable=False),
        sa.Column("stage_history", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("in_sample", sa.JSON(), nullable=False),
        sa.Column("role", sa.String(length=12), nullable=False),
        sa.Column("approved_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_registry")),
        sa.UniqueConstraint("slot", "version", name=op.f("uq_model_registry_slot_version")),
    )
    with op.batch_alter_table("model_registry", schema=None) as batch_op:
        batch_op.create_index("ix_model_registry_slot_stage", ["slot", "stage"], unique=False)

    op.create_table(
        "evolution_hypotheses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("change_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("predicts", sa.Text(), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("identifiable_from_prices", sa.Boolean(), nullable=False),
        sa.Column("evaluated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["change_id"],
            ["evolution_changes.id"],
            name=op.f("fk_evolution_hypotheses_change_id_evolution_changes"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evolution_hypotheses")),
    )
    with op.batch_alter_table("evolution_hypotheses", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_evolution_hypotheses_change_id"), ["change_id"], unique=False)
    with op.batch_alter_table("broker_orders", schema=None) as batch_op:
        batch_op.alter_column("symbol", existing_type=sa.String(length=16), type_=sa.String(length=32))
        batch_op.add_column(
            sa.Column("asset_class", sa.String(length=12), server_default="us_equity", nullable=False)
        )
        batch_op.add_column(
            sa.Column("order_class", sa.String(length=8), server_default="simple", nullable=False)
        )
        batch_op.add_column(sa.Column("position_intent", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("legs", sa.JSON(), nullable=True))
    with op.batch_alter_table("trading_events", schema=None) as batch_op:
        batch_op.alter_column(
            "symbol", existing_type=sa.String(length=16), type_=sa.String(length=32), existing_nullable=True
        )


def downgrade() -> None:
    with op.batch_alter_table("trading_events", schema=None) as batch_op:
        batch_op.alter_column(
            "symbol", existing_type=sa.String(length=32), type_=sa.String(length=16), existing_nullable=True
        )
    with op.batch_alter_table("broker_orders", schema=None) as batch_op:
        batch_op.drop_column("legs")
        batch_op.drop_column("position_intent")
        batch_op.drop_column("order_class")
        batch_op.drop_column("asset_class")
        batch_op.alter_column("symbol", existing_type=sa.String(length=32), type_=sa.String(length=16))
    with op.batch_alter_table("evolution_hypotheses", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_evolution_hypotheses_change_id"))

    op.drop_table("evolution_hypotheses")
    with op.batch_alter_table("model_registry", schema=None) as batch_op:
        batch_op.drop_index("ix_model_registry_slot_stage")

    op.drop_table("model_registry")
    with op.batch_alter_table("evolution_relationships", schema=None) as batch_op:
        batch_op.drop_index("ix_evolution_relationships_key")

    op.drop_table("evolution_relationships")
    with op.batch_alter_table("evolution_metrics", schema=None) as batch_op:
        batch_op.drop_index("ix_evolution_metrics_series")
        batch_op.drop_index(batch_op.f("ix_evolution_metrics_day"))

    op.drop_table("evolution_metrics")
    with op.batch_alter_table("evolution_changes", schema=None) as batch_op:
        batch_op.drop_index("ix_evolution_changes_series")
        batch_op.drop_index(batch_op.f("ix_evolution_changes_detected_at"))

    op.drop_table("evolution_changes")
