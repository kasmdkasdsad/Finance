"""Market data core: price bars, quote snapshots, ingestion audit log.

Revision ID: 0001
Revises:
Create Date: 2026-09-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "price_bars",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("interval", sa.String(length=8), nullable=False),
        sa.Column("ts", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("ingested_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_price_bars")),
        sa.UniqueConstraint("symbol", "interval", "ts", name=op.f("uq_price_bars_symbol_interval_ts")),
    )
    op.create_index("ix_price_bars_symbol_interval_ts", "price_bars", ["symbol", "interval", "ts"])

    op.create_table(
        "quote_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("previous_close", sa.Float(), nullable=True),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("quoted_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("ingested_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_quote_snapshots")),
    )
    op.create_index("ix_quote_snapshots_symbol_quoted_at", "quote_snapshots", ["symbol", "quoted_at"])

    op.create_table(
        "ingestion_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset", sa.String(length=40), nullable=False),
        sa.Column("key", sa.String(length=160), nullable=False),
        sa.Column("provider", sa.String(length=40), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_events")),
    )
    op.create_index("ix_ingestion_events_dataset_created_at", "ingestion_events", ["dataset", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_events_dataset_created_at", table_name="ingestion_events")
    op.drop_table("ingestion_events")
    op.drop_index("ix_quote_snapshots_symbol_quoted_at", table_name="quote_snapshots")
    op.drop_table("quote_snapshots")
    op.drop_index("ix_price_bars_symbol_interval_ts", table_name="price_bars")
    op.drop_table("price_bars")
