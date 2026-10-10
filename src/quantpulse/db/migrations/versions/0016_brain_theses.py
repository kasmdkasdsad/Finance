"""Position theses for the Alpaca paper account the Brain owns.

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_theses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("origin", sa.String(length=12), nullable=False),
        sa.Column("opened_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("closed_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("entry_qty", sa.Float(), nullable=False),
        sa.Column("entry_decision_id", sa.Integer(), nullable=True),
        sa.Column("entry_order_id", sa.String(length=128), nullable=True),
        sa.Column("thesis", sa.Text(), nullable=False),
        sa.Column("invalidation", sa.Text(), nullable=True),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("target_price", sa.Float(), nullable=True),
        sa.Column("expected_return", sa.Float(), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("supporting", sa.JSON(), nullable=False),
        sa.Column("opposing", sa.JSON(), nullable=False),
        sa.Column("regime", sa.String(length=24), nullable=True),
        sa.Column("sector", sa.String(length=64), nullable=True),
        sa.Column("benchmark_entry", sa.Float(), nullable=True),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("avg_price", sa.Float(), nullable=False),
        sa.Column("last_price", sa.Float(), nullable=True),
        sa.Column("market_value", sa.Float(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("unrealized_pnl", sa.Float(), nullable=True),
        sa.Column("return_pct", sa.Float(), nullable=True),
        sa.Column("benchmark_return", sa.Float(), nullable=True),
        sa.Column("check", sa.JSON(), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.Text(), nullable=True),
        sa.Column("exit_decision_id", sa.Integer(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
        sa.Column("history", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_theses")),
    )
    op.create_index("ix_brain_theses_symbol_status", "brain_theses", ["symbol", "status"])


def downgrade() -> None:
    op.drop_index("ix_brain_theses_symbol_status", table_name="brain_theses")
    op.drop_table("brain_theses")
