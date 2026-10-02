"""The Brain's automatic daily and weekly reviews of its own results.

Revision ID: 0020
Revises: 0019
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=12), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("body", sa.JSON(), nullable=False),
        sa.Column("lessons", sa.JSON(), nullable=False),
        sa.Column("proposals", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_reviews")),
    )
    op.create_index("ix_brain_reviews_kind_period", "brain_reviews", ["kind", "period_end"])


def downgrade() -> None:
    op.drop_index("ix_brain_reviews_kind_period", table_name="brain_reviews")
    op.drop_table("brain_reviews")
