"""Widen two columns that held longer values than declared. SQLite ignores VARCHAR lengths; PostgreSQL (the
cloud deployment's database) enforces them, so these writes would fail there: a Brain cycle's trigger
("supervisor: PriceMoveDetected") and a reference blob's cache key.

Revision ID: 0021
Revises: 0020
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("brain_cycles") as batch:
        batch.alter_column("trigger", existing_type=sa.String(length=16), type_=sa.String(length=96))
    with op.batch_alter_table("reference_blobs") as batch:
        batch.alter_column("key", existing_type=sa.String(length=64), type_=sa.String(length=160))


def downgrade() -> None:
    with op.batch_alter_table("reference_blobs") as batch:
        batch.alter_column("key", existing_type=sa.String(length=160), type_=sa.String(length=64))
    with op.batch_alter_table("brain_cycles") as batch:
        batch.alter_column("trigger", existing_type=sa.String(length=96), type_=sa.String(length=16))
