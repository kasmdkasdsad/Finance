"""One row per trading day of the Alpaca paper account (pre-market check and close).

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("owner", sa.String(length=16), nullable=False),
        sa.Column("equity_open", sa.Float(), nullable=True),
        sa.Column("equity_close", sa.Float(), nullable=True),
        sa.Column("day_return", sa.Float(), nullable=True),
        sa.Column("benchmark_close", sa.Float(), nullable=True),
        sa.Column("benchmark_return", sa.Float(), nullable=True),
        sa.Column("exposure", sa.Float(), nullable=True),
        sa.Column("positions", sa.Integer(), nullable=True),
        sa.Column("orders_sent", sa.Integer(), nullable=False),
        sa.Column("orders_filled", sa.Integer(), nullable=False),
        sa.Column("traded_notional", sa.Float(), nullable=False),
        sa.Column("cycles", sa.Integer(), nullable=False),
        sa.Column("data_blocked_cycles", sa.Integer(), nullable=False),
        sa.Column("halts", sa.JSON(), nullable=False),
        sa.Column("premarket", sa.JSON(), nullable=False),
        sa.Column("close", sa.JSON(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_sessions")),
        sa.UniqueConstraint("day", name=op.f("uq_brain_sessions_day")),
    )


def downgrade() -> None:
    op.drop_table("brain_sessions")
