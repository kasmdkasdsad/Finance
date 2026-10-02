"""Risk-free yield curve points and option-chain snapshots.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "yield_curve_points",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("curve_date", sa.Date(), nullable=False),
        sa.Column("tenor", sa.String(length=16), nullable=False),
        sa.Column("years", sa.Float(), nullable=False),
        sa.Column("rate", sa.Float(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("ingested_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_yield_curve_points")),
        sa.UniqueConstraint("curve_date", "tenor", name=op.f("uq_yield_curve_points_curve_date_tenor")),
    )
    op.create_index(op.f("ix_yield_curve_points_curve_date"), "yield_curve_points", ["curve_date"])

    op.create_table(
        "option_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("contract_symbol", sa.String(length=40), nullable=False),
        sa.Column("kind", sa.String(length=4), nullable=False),
        sa.Column("strike", sa.Float(), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("last", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("open_interest", sa.Float(), nullable=True),
        sa.Column("implied_volatility", sa.Float(), nullable=True),
        sa.Column("underlying_price", sa.Float(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("snapshot_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_option_snapshots")),
        sa.UniqueConstraint(
            "contract_symbol", "snapshot_at", name=op.f("uq_option_snapshots_contract_symbol_snapshot_at")
        ),
    )
    op.create_index(
        "ix_option_snapshots_underlying_snapshot_at", "option_snapshots", ["underlying", "snapshot_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_option_snapshots_underlying_snapshot_at", table_name="option_snapshots")
    op.drop_table("option_snapshots")
    op.drop_index(op.f("ix_yield_curve_points_curve_date"), table_name="yield_curve_points")
    op.drop_table("yield_curve_points")
