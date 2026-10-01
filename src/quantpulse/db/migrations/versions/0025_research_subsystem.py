"""The Brain's 24/7 research subsystem (see quantpulse.db.research_models): the persistent research queue (every
job and its result), the learning ledger (conclusions with their evidence, UNPROVEN until the sample suffices) and
the improvement lifecycle (hypotheses only a person can promote). Additive: three new tables.

Revision ID: 0025
Revises: 0024
Create Date: 2026-10-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_hypotheses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=160), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("source_ref", sa.String(length=96), nullable=True),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("protected_control", sa.String(length=48), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("history", sa.JSON(), nullable=False),
        sa.Column("next_step_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("decided_by", sa.String(length=48), nullable=True),
        sa.Column("promoted_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_hypotheses")),
        sa.UniqueConstraint("key", name=op.f("uq_brain_hypotheses_key")),
    )
    with op.batch_alter_table("brain_hypotheses", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_hypotheses_stage"), ["stage"], unique=False)

    op.create_table(
        "brain_learnings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("topic", sa.String(length=160), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("min_sample", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=True),
        sa.Column("period_end", sa.Date(), nullable=True),
        sa.Column("regime", sa.String(length=48), nullable=False),
        sa.Column("benchmark", sa.String(length=64), nullable=False),
        sa.Column("method", sa.String(length=96), nullable=False),
        sa.Column("statistics", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("limitations", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("supersedes_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_learnings")),
    )
    with op.batch_alter_table("brain_learnings", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_learnings_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_brain_learnings_job_id"), ["job_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_brain_learnings_topic"), ["topic"], unique=False)

    op.create_table(
        "brain_research_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("key", sa.String(length=160), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("cost", sa.String(length=8), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("priority_detail", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("not_before", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("started_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("heartbeat_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("finished_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("holder", sa.String(length=96), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("peak_rss_mb", sa.Float(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_research_jobs")),
    )
    with op.batch_alter_table("brain_research_jobs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_research_jobs_key"), ["key"], unique=False)
        batch_op.create_index(batch_op.f("ix_brain_research_jobs_kind"), ["kind"], unique=False)
        batch_op.create_index("ix_brain_research_jobs_queue", ["status", "priority"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("brain_research_jobs", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_research_jobs_queue")
        batch_op.drop_index(batch_op.f("ix_brain_research_jobs_kind"))
        batch_op.drop_index(batch_op.f("ix_brain_research_jobs_key"))

    op.drop_table("brain_research_jobs")
    with op.batch_alter_table("brain_learnings", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_brain_learnings_topic"))
        batch_op.drop_index(batch_op.f("ix_brain_learnings_job_id"))
        batch_op.drop_index(batch_op.f("ix_brain_learnings_created_at"))

    op.drop_table("brain_learnings")
    with op.batch_alter_table("brain_hypotheses", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_brain_hypotheses_stage"))

    op.drop_table("brain_hypotheses")
