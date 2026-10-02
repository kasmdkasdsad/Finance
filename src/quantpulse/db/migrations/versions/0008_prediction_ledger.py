"""Prediction ledger: logged forecasts and model predictions with their graded outcomes.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("made_on", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("reference_price", sa.Float(), nullable=False),
        sa.Column("benchmark", sa.String(length=16), nullable=False),
        sa.Column("benchmark_reference", sa.Float(), nullable=True),
        sa.Column("prob_up", sa.Float(), nullable=True),
        sa.Column("prob_outperform", sa.Float(), nullable=True),
        sa.Column("expected_return", sa.Float(), nullable=True),
        sa.Column("q05", sa.Float(), nullable=True),
        sa.Column("q25", sa.Float(), nullable=True),
        sa.Column("q50", sa.Float(), nullable=True),
        sa.Column("q75", sa.Float(), nullable=True),
        sa.Column("q95", sa.Float(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(length=40), nullable=False),
        sa.Column("data_status", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=10), nullable=False),
        sa.Column("resolved_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("realized_price", sa.Float(), nullable=True),
        sa.Column("benchmark_realized", sa.Float(), nullable=True),
        sa.Column("realized_return", sa.Float(), nullable=True),
        sa.Column("benchmark_return", sa.Float(), nullable=True),
        sa.Column("outcome_up", sa.Boolean(), nullable=True),
        sa.Column("outcome_outperform", sa.Boolean(), nullable=True),
        sa.Column("in_50", sa.Boolean(), nullable=True),
        sa.Column("in_90", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.UniqueConstraint(
            "symbol",
            "source",
            "horizon_days",
            "made_on",
            name=op.f("uq_predictions_symbol_source_horizon_days_made_on"),
        ),
    )
    op.create_index("ix_predictions_status_target_date", "predictions", ["status", "target_date"])
    op.create_index("ix_predictions_symbol_made_on", "predictions", ["symbol", "made_on"])


def downgrade() -> None:
    op.drop_index("ix_predictions_symbol_made_on", table_name="predictions")
    op.drop_index("ix_predictions_status_target_date", table_name="predictions")
    op.drop_table("predictions")
