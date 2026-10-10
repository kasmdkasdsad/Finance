"""A lease table: at most one process supervises the Brain and sends orders.

Revision ID: 0022
Revises: 0021
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "service_leases",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("holder", sa.String(length=128), nullable=False),
        sa.Column("acquired_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("heartbeat_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("expires_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("name", name=op.f("pk_service_leases")),
    )


def downgrade() -> None:
    op.drop_table("service_leases")
