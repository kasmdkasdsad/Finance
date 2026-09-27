"""Prediction expected returns and statistically honest track records.

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PERFORMANCE_COLUMNS: tuple[tuple[str, sa.types.TypeEngine], ...] = (
    ("n_effective", sa.Integer()),
    ("ci_low", sa.Float()),
    ("ci_high", sa.Float()),
    ("p_value", sa.Float()),
    ("q_value", sa.Float()),
    ("verdict", sa.String(length=24)),
    ("mean_excess", sa.Float()),
    ("mean_excess_z", sa.Float()),
)


def upgrade() -> None:
    with op.batch_alter_table("brain_predictions") as batch:
        batch.add_column(sa.Column("expected_return", sa.Float(), nullable=True))
    with op.batch_alter_table("brain_agent_performance") as batch:
        for name, kind in PERFORMANCE_COLUMNS:
            batch.add_column(sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("brain_agent_performance") as batch:
        for name, _ in reversed(PERFORMANCE_COLUMNS):
            batch.drop_column(name)
    with op.batch_alter_table("brain_predictions") as batch:
        batch.drop_column("expected_return")
