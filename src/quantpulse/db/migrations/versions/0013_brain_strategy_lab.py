"""The brain's strategy lab: versioned strategies and their validation and paper runs.

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_strategies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_id", sa.String(length=48), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("parent_version", sa.Integer(), nullable=True),
        sa.Column("validation", sa.JSON(), nullable=False),
        sa.Column("paper", sa.JSON(), nullable=False),
        sa.Column("decided_by", sa.String(length=24), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("promoted_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_strategies")),
        sa.UniqueConstraint("strategy_id", "version", name="uq_brain_strategies_version"),
    )
    op.create_table(
        "brain_strategy_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_row_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["strategy_row_id"],
            ["brain_strategies.id"],
            name=op.f("fk_brain_strategy_runs_strategy_row_id_brain_strategies"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_strategy_runs")),
    )
    with op.batch_alter_table("brain_strategy_runs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_brain_strategy_runs_kind", ["strategy_row_id", "kind", "created_at"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_brain_strategy_runs_strategy_row_id"), ["strategy_row_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("brain_strategy_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_brain_strategy_runs_strategy_row_id"))
        batch_op.drop_index("ix_brain_strategy_runs_kind")

    op.drop_table("brain_strategy_runs")
    op.drop_table("brain_strategies")
