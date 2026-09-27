"""The brain's research team: detected opportunities (with their pipeline trace) and adversarial debates.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_debates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=24), nullable=False),
        sa.Column("stance_before", sa.String(length=10), nullable=False),
        sa.Column("confidence_before", sa.Float(), nullable=False),
        sa.Column("confidence_after", sa.Float(), nullable=False),
        sa.Column("verdict", sa.String(length=24), nullable=False),
        sa.Column("bull", sa.JSON(), nullable=False),
        sa.Column("bear", sa.JSON(), nullable=False),
        sa.Column("objections", sa.JSON(), nullable=False),
        sa.Column("change_our_mind", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_debates_cycle_id_brain_cycles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_debates")),
    )
    with op.batch_alter_table("brain_debates", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_debates_cycle_id"), ["cycle_id"], unique=False)
        batch_op.create_index("ix_brain_debates_subject", ["subject", "created_at"], unique=False)

    op.create_table(
        "brain_opportunities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("subject", sa.String(length=48), nullable=False),
        sa.Column("symbols", sa.JSON(), nullable=False),
        sa.Column("direction", sa.Integer(), nullable=False),
        sa.Column("strength", sa.Float(), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("stages", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_opportunities_cycle_id_brain_cycles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_opportunities")),
    )
    with op.batch_alter_table("brain_opportunities", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_opportunities_cycle_id"), ["cycle_id"], unique=False)
        batch_op.create_index("ix_brain_opportunities_kind_created", ["kind", "created_at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("brain_opportunities", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_opportunities_kind_created")
        batch_op.drop_index(batch_op.f("ix_brain_opportunities_cycle_id"))

    op.drop_table("brain_opportunities")
    with op.batch_alter_table("brain_debates", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_debates_subject")
        batch_op.drop_index(batch_op.f("ix_brain_debates_cycle_id"))

    op.drop_table("brain_debates")
