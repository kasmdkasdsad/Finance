"""The Brain's hypothetical paper book: positions, simulated fills and the equity curve.

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_book_positions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("avg_cost", sa.Float(), nullable=False),
        sa.Column("last_price", sa.Float(), nullable=False),
        sa.Column("opened_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("invalidation", sa.Text(), nullable=True),
        sa.Column("thesis", sa.Text(), nullable=True),
        sa.Column("expected_return", sa.Float(), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=True),
        sa.Column("review_after", sa.Date(), nullable=True),
        sa.Column("entry_decision_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_book_positions")),
        sa.UniqueConstraint("symbol", name=op.f("uq_brain_book_positions_symbol")),
    )
    op.create_table(
        "brain_book_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("decision_id", sa.Integer(), nullable=True),
        sa.Column("executed_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("symbol", sa.String(length=24), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("proposed_price", sa.Float(), nullable=False),
        sa.Column("fill_price", sa.Float(), nullable=False),
        sa.Column("notional", sa.Float(), nullable=False),
        sa.Column("slippage_bps", sa.Float(), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.Column("price_source", sa.String(length=64), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
        sa.Column("holding_days", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_book_trades")),
    )
    op.create_index(op.f("ix_brain_book_trades_cycle_id"), "brain_book_trades", ["cycle_id"])
    op.create_index("ix_brain_book_trades_symbol", "brain_book_trades", ["symbol", "executed_at"])
    op.create_table(
        "brain_book_equity",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("invested", sa.Float(), nullable=False),
        sa.Column("positions", sa.Integer(), nullable=False),
        sa.Column("benchmark_price", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_book_equity")),
    )
    op.create_index(op.f("ix_brain_book_equity_at"), "brain_book_equity", ["at"])
    op.create_index(op.f("ix_brain_book_equity_day"), "brain_book_equity", ["day"])


def downgrade() -> None:
    op.drop_index(op.f("ix_brain_book_equity_day"), table_name="brain_book_equity")
    op.drop_index(op.f("ix_brain_book_equity_at"), table_name="brain_book_equity")
    op.drop_table("brain_book_equity")
    op.drop_index("ix_brain_book_trades_symbol", table_name="brain_book_trades")
    op.drop_index(op.f("ix_brain_book_trades_cycle_id"), table_name="brain_book_trades")
    op.drop_table("brain_book_trades")
    op.drop_table("brain_book_positions")
