"""Ideas the Brain considered — taken or not, and why not — graded later against the benchmark.

Revision ID: 0019
Revises: 0018
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_opportunity_outcomes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("opportunity_id", sa.Integer(), nullable=True),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("direction", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("detected_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("strength", sa.Float(), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("taken", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("reason_detail", sa.Text(), nullable=True),
        sa.Column("stopped_at", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("repeats", sa.Integer(), nullable=False),
        sa.Column("regime", sa.String(length=24), nullable=True),
        sa.Column("vol_env", sa.String(length=12), nullable=True),
        sa.Column("market_open", sa.Boolean(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("entry_benchmark", sa.Float(), nullable=True),
        sa.Column("vol", sa.Float(), nullable=True),
        sa.Column("state", sa.String(length=12), nullable=False),
        sa.Column("realized_return", sa.Float(), nullable=True),
        sa.Column("relative", sa.Float(), nullable=True),
        sa.Column("favourable", sa.Float(), nullable=True),
        sa.Column("z", sa.Float(), nullable=True),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("evaluated_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_opportunity_outcomes")),
    )
    op.create_index(
        "ix_brain_opportunity_outcomes_key",
        "brain_opportunity_outcomes",
        ["kind", "symbol", "direction", "day"],
        unique=True,
    )
    op.create_index(
        "ix_brain_opportunity_outcomes_state", "brain_opportunity_outcomes", ["state", "due_date"]
    )


def downgrade() -> None:
    op.drop_index("ix_brain_opportunity_outcomes_state", table_name="brain_opportunity_outcomes")
    op.drop_index("ix_brain_opportunity_outcomes_key", table_name="brain_opportunity_outcomes")
    op.drop_table("brain_opportunity_outcomes")
