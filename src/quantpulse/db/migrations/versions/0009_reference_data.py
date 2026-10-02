"""Reference data (company profiles, earnings events, reference blobs, XBRL frame facts) and the
prediction ledger's origin column (live vs backfilled).

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "company_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("sic", sa.String(length=8), nullable=True),
        sa.Column("sic_description", sa.String(length=200), nullable=True),
        sa.Column("sector", sa.String(length=8), nullable=False),
        sa.Column("earnings_since", sa.Date(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_company_profiles")),
        sa.UniqueConstraint("symbol", name=op.f("uq_company_profiles_symbol")),
    )
    op.create_table(
        "earnings_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("announced_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_earnings_events")),
        sa.UniqueConstraint("symbol", "announced_at", name=op.f("uq_earnings_events_symbol_announced_at")),
    )
    op.create_index(op.f("ix_earnings_events_symbol"), "earnings_events", ["symbol"])
    op.create_table(
        "reference_blobs",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("fetched_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_reference_blobs")),
    )
    op.create_table(
        "fundamental_facts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tag", sa.String(length=96), nullable=False),
        sa.Column("frame", sa.String(length=12), nullable=False),
        sa.Column("cik", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("accn", sa.String(length=25), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fundamental_facts")),
        sa.UniqueConstraint("tag", "frame", "cik", name=op.f("uq_fundamental_facts_tag_frame_cik")),
    )
    op.create_index("ix_fundamental_facts_cik_tag", "fundamental_facts", ["cik", "tag"])
    with op.batch_alter_table("predictions") as batch:
        batch.add_column(sa.Column("origin", sa.String(length=10), server_default="live", nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.drop_column("origin")
    op.drop_index("ix_fundamental_facts_cik_tag", table_name="fundamental_facts")
    op.drop_table("fundamental_facts")
    op.drop_table("reference_blobs")
    op.drop_index(op.f("ix_earnings_events_symbol"), table_name="earnings_events")
    op.drop_table("earnings_events")
    op.drop_table("company_profiles")
