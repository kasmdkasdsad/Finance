"""The Brain's execution ledger (one row per order) and the near-close review of each trading day.

Revision ID: 0018
Revises: 0017
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_executions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("client_order_id", sa.String(length=128), nullable=False),
        sa.Column("alpaca_order_id", sa.String(length=64), nullable=True),
        sa.Column("decision_id", sa.Integer(), nullable=True),
        sa.Column("brain_cycle_id", sa.Integer(), nullable=True),
        sa.Column("trading_cycle_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("consensus", sa.JSON(), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("order_type", sa.String(length=20), nullable=True),
        sa.Column("expected_price", sa.Float(), nullable=True),
        sa.Column("submitted_price", sa.Float(), nullable=True),
        sa.Column("quote_price", sa.Float(), nullable=True),
        sa.Column("quote_bid", sa.Float(), nullable=True),
        sa.Column("quote_ask", sa.Float(), nullable=True),
        sa.Column("spread_bps", sa.Float(), nullable=True),
        sa.Column("quote_age_s", sa.Float(), nullable=True),
        sa.Column("quote_source", sa.String(length=96), nullable=True),
        sa.Column("decided_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("submitted_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("submit_latency_ms", sa.Float(), nullable=True),
        sa.Column("decision_to_submit_s", sa.Float(), nullable=True),
        sa.Column("filled_qty", sa.Float(), nullable=False),
        sa.Column("filled_avg_price", sa.Float(), nullable=True),
        sa.Column("filled_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("seconds_to_fill", sa.Float(), nullable=True),
        sa.Column("partial", sa.Boolean(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("final", sa.Boolean(), nullable=False),
        sa.Column("slippage_bps", sa.Float(), nullable=True),
        sa.Column("cost_vs_quote_bps", sa.Float(), nullable=True),
        sa.Column("grade", sa.String(length=12), nullable=True),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_executions")),
        sa.UniqueConstraint("client_order_id", name=op.f("uq_brain_executions_client_order_id")),
    )
    op.create_index(op.f("ix_brain_executions_decision_id"), "brain_executions", ["decision_id"])
    op.create_index("ix_brain_executions_symbol", "brain_executions", ["symbol", "submitted_at"])
    with op.batch_alter_table("brain_sessions") as batch:
        batch.add_column(sa.Column("near_close", sa.JSON(), nullable=False, server_default="{}"))


def downgrade() -> None:
    with op.batch_alter_table("brain_sessions") as batch:
        batch.drop_column("near_close")
    op.drop_index("ix_brain_executions_symbol", table_name="brain_executions")
    op.drop_index(op.f("ix_brain_executions_decision_id"), table_name="brain_executions")
    op.drop_table("brain_executions")
