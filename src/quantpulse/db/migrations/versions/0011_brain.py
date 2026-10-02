"""The multi-agent brain: agents, cycles, agent runs, opinions, consensus, decisions, predictions,
reflections, agent performance, improvements, memory, events and state.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-27
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_agent_performance",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.String(length=48), nullable=False),
        sa.Column("agent_version", sa.String(length=16), nullable=False),
        sa.Column("regime", sa.String(length=24), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("window", sa.String(length=16), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False),
        sa.Column("hit_rate", sa.Float(), nullable=True),
        sa.Column("brier", sa.Float(), nullable=True),
        sa.Column("ic", sa.Float(), nullable=True),
        sa.Column("calibration", sa.JSON(), nullable=False),
        sa.Column("reliability", sa.Float(), nullable=True),
        sa.Column("computed_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_agent_performance")),
        sa.UniqueConstraint(
            "agent_id", "agent_version", "regime", "horizon_days", "window", name="uq_brain_agent_performance"
        ),
    )
    op.create_table(
        "brain_agents",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("family", sa.String(length=16), nullable=False),
        sa.Column("version", sa.String(length=16), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("registered_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_agents")),
    )
    op.create_table(
        "brain_cycles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False),
        sa.Column("session", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("started_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("finished_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("regime", sa.JSON(), nullable=False),
        sa.Column("market", sa.JSON(), nullable=False),
        sa.Column("portfolio", sa.JSON(), nullable=False),
        sa.Column("data_quality", sa.JSON(), nullable=False),
        sa.Column("focus", sa.JSON(), nullable=False),
        sa.Column("agents", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("notes", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_cycles")),
    )
    with op.batch_alter_table("brain_cycles", schema=None) as batch_op:
        batch_op.create_index("ix_brain_cycles_started_at", ["started_at"], unique=False)

    op.create_table(
        "brain_improvements",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("target", sa.String(length=48), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("proposal", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("test_result", sa.JSON(), nullable=False),
        sa.Column("decided_by", sa.String(length=16), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_improvements")),
    )
    op.create_table(
        "brain_reflections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("subject_type", sa.String(length=16), nullable=False),
        sa.Column("subject_id", sa.Integer(), nullable=True),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("decision_quality", sa.String(length=16), nullable=True),
        sa.Column("outcome_quality", sa.String(length=16), nullable=True),
        sa.Column("questions", sa.JSON(), nullable=False),
        sa.Column("lessons", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_reflections")),
    )
    with op.batch_alter_table("brain_reflections", schema=None) as batch_op:
        batch_op.create_index("ix_brain_reflections_subject", ["subject_type", "subject_id"], unique=False)

    op.create_table(
        "brain_state",
        sa.Column("key", sa.String(length=40), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_brain_state")),
    )
    op.create_table(
        "brain_agent_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.String(length=48), nullable=False),
        sa.Column("agent_version", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("started_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("subjects", sa.Integer(), nullable=False),
        sa.Column("opinions", sa.Integer(), nullable=False),
        sa.Column("model_tier", sa.String(length=16), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_agent_runs_cycle_id_brain_cycles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_agent_runs")),
    )
    with op.batch_alter_table("brain_agent_runs", schema=None) as batch_op:
        batch_op.create_index("ix_brain_agent_runs_agent", ["agent_id", "started_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_brain_agent_runs_cycle_id"), ["cycle_id"], unique=False)

    op.create_table(
        "brain_consensus",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=False),
        sa.Column("subject", sa.String(length=24), nullable=False),
        sa.Column("stance", sa.String(length=10), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("unknown", sa.Boolean(), nullable=False),
        sa.Column("supporting", sa.Integer(), nullable=False),
        sa.Column("neutral", sa.Integer(), nullable=False),
        sa.Column("opposing", sa.Integer(), nullable=False),
        sa.Column("abstaining", sa.Integer(), nullable=False),
        sa.Column("disagreement", sa.Float(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("vetoes", sa.JSON(), nullable=False),
        sa.Column("data_quality", sa.String(length=16), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_consensus_cycle_id_brain_cycles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_consensus")),
    )
    with op.batch_alter_table("brain_consensus", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_consensus_cycle_id"), ["cycle_id"], unique=False)
        batch_op.create_index("ix_brain_consensus_subject", ["subject", "created_at"], unique=False)

    op.create_table(
        "brain_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=40), nullable=False),
        sa.Column("subject", sa.String(length=24), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_events_cycle_id_brain_cycles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_events")),
    )
    with op.batch_alter_table("brain_events", schema=None) as batch_op:
        batch_op.create_index("ix_brain_events_type_created", ["type", "created_at"], unique=False)

    op.create_table(
        "brain_memory",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("tier", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("subject", sa.String(length=24), nullable=False),
        sa.Column("key", sa.String(length=96), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("expires_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_memory_cycle_id_brain_cycles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_memory")),
    )
    with op.batch_alter_table("brain_memory", schema=None) as batch_op:
        batch_op.create_index("ix_brain_memory_key", ["tier", "key"], unique=False)
        batch_op.create_index("ix_brain_memory_tier_subject", ["tier", "subject"], unique=False)

    op.create_table(
        "brain_predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("source_id", sa.String(length=48), nullable=False),
        sa.Column("source_version", sa.String(length=16), nullable=False),
        sa.Column("subject", sa.String(length=24), nullable=False),
        sa.Column("direction", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("benchmark", sa.String(length=16), nullable=False),
        sa.Column("regime", sa.String(length=24), nullable=True),
        sa.Column("made_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("entry_benchmark", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("realized_return", sa.Float(), nullable=True),
        sa.Column("realized_relative", sa.Float(), nullable=True),
        sa.Column("hit", sa.Boolean(), nullable=True),
        sa.Column("evaluated_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_predictions_cycle_id_brain_cycles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_predictions")),
    )
    with op.batch_alter_table("brain_predictions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_predictions_cycle_id"), ["cycle_id"], unique=False)
        batch_op.create_index("ix_brain_predictions_due", ["status", "due_date"], unique=False)
        batch_op.create_index("ix_brain_predictions_source", ["source_type", "source_id"], unique=False)

    op.create_table(
        "brain_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=False),
        sa.Column("consensus_id", sa.Integer(), nullable=True),
        sa.Column("subject", sa.String(length=24), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("est_price", sa.Float(), nullable=True),
        sa.Column("notional", sa.Float(), nullable=True),
        sa.Column("current_weight", sa.Float(), nullable=True),
        sa.Column("target_weight", sa.Float(), nullable=True),
        sa.Column("rationale", sa.JSON(), nullable=False),
        sa.Column("risk_approved", sa.Boolean(), nullable=True),
        sa.Column("risk", sa.JSON(), nullable=False),
        sa.Column("execution", sa.JSON(), nullable=False),
        sa.Column("outcome", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("evaluated_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["consensus_id"],
            ["brain_consensus.id"],
            name=op.f("fk_brain_decisions_consensus_id_brain_consensus"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_decisions_cycle_id_brain_cycles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_decisions")),
    )
    with op.batch_alter_table("brain_decisions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_brain_decisions_cycle_id"), ["cycle_id"], unique=False)
        batch_op.create_index("ix_brain_decisions_subject", ["subject", "created_at"], unique=False)

    op.create_table(
        "brain_opinions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("agent_id", sa.String(length=48), nullable=False),
        sa.Column("agent_version", sa.String(length=16), nullable=False),
        sa.Column("subject", sa.String(length=24), nullable=False),
        sa.Column("stance", sa.String(length=10), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("thesis", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("data_missing", sa.JSON(), nullable=False),
        sa.Column("data_quality", sa.String(length=16), nullable=False),
        sa.Column("invalidation", sa.Text(), nullable=True),
        sa.Column("veto", sa.Text(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["brain_cycles.id"],
            name=op.f("fk_brain_opinions_cycle_id_brain_cycles"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["brain_agent_runs.id"],
            name=op.f("fk_brain_opinions_run_id_brain_agent_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_brain_opinions")),
    )
    with op.batch_alter_table("brain_opinions", schema=None) as batch_op:
        batch_op.create_index("ix_brain_opinions_agent", ["agent_id", "created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_brain_opinions_cycle_id"), ["cycle_id"], unique=False)
        batch_op.create_index("ix_brain_opinions_subject", ["subject", "created_at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("brain_opinions", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_opinions_subject")
        batch_op.drop_index(batch_op.f("ix_brain_opinions_cycle_id"))
        batch_op.drop_index("ix_brain_opinions_agent")

    op.drop_table("brain_opinions")
    with op.batch_alter_table("brain_decisions", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_decisions_subject")
        batch_op.drop_index(batch_op.f("ix_brain_decisions_cycle_id"))

    op.drop_table("brain_decisions")
    with op.batch_alter_table("brain_predictions", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_predictions_source")
        batch_op.drop_index("ix_brain_predictions_due")
        batch_op.drop_index(batch_op.f("ix_brain_predictions_cycle_id"))

    op.drop_table("brain_predictions")
    with op.batch_alter_table("brain_memory", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_memory_tier_subject")
        batch_op.drop_index("ix_brain_memory_key")

    op.drop_table("brain_memory")
    with op.batch_alter_table("brain_events", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_events_type_created")

    op.drop_table("brain_events")
    with op.batch_alter_table("brain_consensus", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_consensus_subject")
        batch_op.drop_index(batch_op.f("ix_brain_consensus_cycle_id"))

    op.drop_table("brain_consensus")
    with op.batch_alter_table("brain_agent_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_brain_agent_runs_cycle_id"))
        batch_op.drop_index("ix_brain_agent_runs_agent")

    op.drop_table("brain_agent_runs")
    op.drop_table("brain_state")
    with op.batch_alter_table("brain_reflections", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_reflections_subject")

    op.drop_table("brain_reflections")
    op.drop_table("brain_improvements")
    with op.batch_alter_table("brain_cycles", schema=None) as batch_op:
        batch_op.drop_index("ix_brain_cycles_started_at")

    op.drop_table("brain_cycles")
    op.drop_table("brain_agents")
    op.drop_table("brain_agent_performance")
