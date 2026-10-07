"""The trailing stop: each position thesis keeps the highest price seen since it opened (``peak_price``); once
the position is up 10% on what was paid, its stop follows that high and never moves down. Additive: one
nullable column.

Revision ID: 0026
Revises: 0025
Create Date: 2026-10-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("brain_theses", schema=None) as batch:
        batch.add_column(sa.Column("peak_price", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("brain_theses", schema=None) as batch:
        batch.drop_column("peak_price")
