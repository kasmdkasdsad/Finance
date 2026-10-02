"""The options layer: market data, the research library, strategy versions and their evidence, candidates,
theses, positions and their lifecycle, execution quality, counterfactuals, missed opportunities, learning,
experiments and the knowledge graph (see quantpulse.db.options_models).

Additive only: no existing table changes, so the previous version keeps running on this schema during a deploy.

Revision ID: 0023
Revises: 0022
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "options_chain_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("fetched_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("feed", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("underlying_price", sa.Float(), nullable=False),
        sa.Column("underlying_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("contracts", sa.Integer(), nullable=False),
        sa.Column("quality", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_chain_snapshots")),
    )
    with op.batch_alter_table("options_chain_snapshots", schema=None) as batch_op:
        batch_op.create_index(
            "ix_options_chain_snapshots_underlying_fetched", ["underlying", "fetched_at"], unique=False
        )

    op.create_table(
        "options_contracts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=4), nullable=False),
        sa.Column("strike", sa.Float(), nullable=False),
        sa.Column("multiplier", sa.Integer(), nullable=False),
        sa.Column("style", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("tradable", sa.Boolean(), nullable=False),
        sa.Column("open_interest", sa.Float(), nullable=True),
        sa.Column("open_interest_date", sa.Date(), nullable=True),
        sa.Column("first_seen_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("last_seen_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_contracts")),
        sa.UniqueConstraint("symbol", name=op.f("uq_options_contracts_symbol")),
    )
    with op.batch_alter_table("options_contracts", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_options_contracts_expiration"), ["expiration"], unique=False)
        batch_op.create_index(
            "ix_options_contracts_underlying_expiration", ["underlying", "expiration"], unique=False
        )

    op.create_table(
        "options_feature_importance",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("computed_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("feature", sa.String(length=96), nullable=False),
        sa.Column("interaction_with", sa.String(length=96), nullable=True),
        sa.Column("target", sa.String(length=32), nullable=False),
        sa.Column("importance", sa.Float(), nullable=True),
        sa.Column("oos_importance", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("validated", sa.Boolean(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_feature_importance")),
    )
    op.create_table(
        "options_feature_observations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("outcome", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_feature_observations")),
    )
    with op.batch_alter_table("options_feature_observations", schema=None) as batch_op:
        batch_op.create_index(
            "ix_options_feature_observations_underlying_day", ["underlying", "day"], unique=False
        )

    op.create_table(
        "options_generation_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("started_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("finished_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("created", sa.Integer(), nullable=False),
        sa.Column("evaluated", sa.Integer(), nullable=False),
        sa.Column("promoted", sa.Integer(), nullable=False),
        sa.Column("budget", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_generation_runs")),
    )
    op.create_table(
        "options_iv_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("atm_iv_30d", sa.Float(), nullable=True),
        sa.Column("iv_rank", sa.Float(), nullable=True),
        sa.Column("iv_percentile", sa.Float(), nullable=True),
        sa.Column("rv_20", sa.Float(), nullable=True),
        sa.Column("rv_60", sa.Float(), nullable=True),
        sa.Column("term_slope", sa.Float(), nullable=True),
        sa.Column("term_shape", sa.String(length=16), nullable=True),
        sa.Column("skew_25d", sa.Float(), nullable=True),
        sa.Column("implied_move", sa.Float(), nullable=True),
        sa.Column("feed", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_iv_history")),
        sa.UniqueConstraint("underlying", "day", name=op.f("uq_options_iv_history_underlying_day")),
    )
    with op.batch_alter_table("options_iv_history", schema=None) as batch_op:
        batch_op.create_index("ix_options_iv_history_iv_rank", ["iv_rank"], unique=False)

    op.create_table(
        "options_knowledge_edges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("src_type", sa.String(length=16), nullable=False),
        sa.Column("src_id", sa.String(length=64), nullable=False),
        sa.Column("relation", sa.String(length=24), nullable=False),
        sa.Column("dst_type", sa.String(length=16), nullable=False),
        sa.Column("dst_id", sa.String(length=64), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_knowledge_edges")),
        sa.UniqueConstraint(
            "src_type",
            "src_id",
            "relation",
            "dst_type",
            "dst_id",
            name=op.f("uq_options_knowledge_edges_src_type_src_id_relation_dst_type_dst_id"),
        ),
    )
    with op.batch_alter_table("options_knowledge_edges", schema=None) as batch_op:
        batch_op.create_index("ix_options_knowledge_edges_dst", ["dst_type", "dst_id"], unique=False)

    op.create_table(
        "options_learning_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("evidence", sa.String(length=16), nullable=False),
        sa.Column("subject", sa.String(length=64), nullable=False),
        sa.Column("dims", sa.JSON(), nullable=False),
        sa.Column("predicted", sa.Float(), nullable=True),
        sa.Column("actual", sa.Float(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_learning_events")),
    )
    with op.batch_alter_table("options_learning_events", schema=None) as batch_op:
        batch_op.create_index("ix_options_learning_events_kind_at", ["kind", "at"], unique=False)

    op.create_table(
        "options_lessons",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("memory", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("observation", sa.Text(), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("date_from", sa.Date(), nullable=True),
        sa.Column("date_to", sa.Date(), nullable=True),
        sa.Column("applicability", sa.JSON(), nullable=False),
        sa.Column("expires_on", sa.Date(), nullable=True),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_lessons")),
    )
    with op.batch_alter_table("options_lessons", schema=None) as batch_op:
        batch_op.create_index("ix_options_lessons_memory_status", ["memory", "status"], unique=False)

    op.create_table(
        "options_strategy_genomes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("genome_hash", sa.String(length=64), nullable=False),
        sa.Column("family", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("parameter_count", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_genomes")),
        sa.UniqueConstraint("genome_hash", name=op.f("uq_options_strategy_genomes_genome_hash")),
    )
    with op.batch_alter_table("options_strategy_genomes", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_options_strategy_genomes_family"), ["family"], unique=False)

    op.create_table(
        "options_strategy_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_key", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("author", sa.String(length=200), nullable=False),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("quality", sa.String(length=24), nullable=False),
        sa.Column("reference", sa.Text(), nullable=False),
        sa.Column("market", sa.String(length=120), nullable=False),
        sa.Column("time_period", sa.String(length=64), nullable=False),
        sa.Column("limitations", sa.Text(), nullable=False),
        sa.Column("evidence_grade", sa.JSON(), nullable=False),
        sa.Column("extraction_confidence", sa.Float(), nullable=False),
        sa.Column("reproducibility", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_sources")),
        sa.UniqueConstraint("source_key", name=op.f("uq_options_strategy_sources_source_key")),
    )
    op.create_table(
        "options_strategy_weights",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_key", sa.String(length=64), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=False),
        sa.Column("structure", sa.String(length=32), nullable=False),
        sa.Column("underlying_class", sa.String(length=32), nullable=False),
        sa.Column("vol_state", sa.String(length=32), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("mean", sa.Float(), nullable=False),
        sa.Column("sd", sa.Float(), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_weights")),
        sa.UniqueConstraint(
            "strategy_key",
            "regime",
            "structure",
            "underlying_class",
            "vol_state",
            name=op.f("uq_options_strategy_weights_strategy_key_regime_structure_underlying_class_vol_state"),
        ),
    )
    op.create_table(
        "options_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("size", sa.Float(), nullable=False),
        sa.Column("exchange", sa.String(length=8), nullable=True),
        sa.Column("feed", sa.String(length=16), nullable=False),
        sa.Column("recorded_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_trades")),
    )
    with op.batch_alter_table("options_trades", schema=None) as batch_op:
        batch_op.create_index("ix_options_trades_symbol_at", ["symbol", "at"], unique=False)

    op.create_table(
        "options_quotes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("kind", sa.String(length=4), nullable=False),
        sa.Column("strike", sa.Float(), nullable=False),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("bid_size", sa.Float(), nullable=True),
        sa.Column("ask_size", sa.Float(), nullable=True),
        sa.Column("last", sa.Float(), nullable=True),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.Column("open_interest", sa.Float(), nullable=True),
        sa.Column("iv", sa.Float(), nullable=True),
        sa.Column("quote_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("feed", sa.String(length=16), nullable=False),
        sa.Column("recorded_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["options_chain_snapshots.id"],
            name=op.f("fk_options_quotes_snapshot_id_options_chain_snapshots"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_quotes")),
    )
    with op.batch_alter_table("options_quotes", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_options_quotes_snapshot_id"), ["snapshot_id"], unique=False)
        batch_op.create_index("ix_options_quotes_symbol_quote_at", ["symbol", "quote_at"], unique=False)
        batch_op.create_index(
            "ix_options_quotes_underlying_expiration", ["underlying", "expiration"], unique=False
        )

    op.create_table(
        "options_strategy_claims",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("assumptions", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("test", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["options_strategy_sources.id"],
            name=op.f("fk_options_strategy_claims_source_id_options_strategy_sources"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_claims")),
    )
    with op.batch_alter_table("options_strategy_claims", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_options_strategy_claims_source_id"), ["source_id"], unique=False)

    op.create_table(
        "options_strategy_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_key", sa.String(length=64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("genome_id", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("second_parent_id", sa.Integer(), nullable=True),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("origin", sa.String(length=24), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("stage_history", sa.JSON(), nullable=False),
        sa.Column("role", sa.String(length=12), nullable=False),
        sa.Column("is_baseline", sa.Boolean(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("stage_changed_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["genome_id"],
            ["options_strategy_genomes.id"],
            name=op.f("fk_options_strategy_versions_genome_id_options_strategy_genomes"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_versions_parent_id_options_strategy_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["second_parent_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_versions_second_parent_id_options_strategy_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["options_strategy_sources.id"],
            name=op.f("fk_options_strategy_versions_source_id_options_strategy_sources"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_versions")),
        sa.UniqueConstraint(
            "strategy_key", "version", name=op.f("uq_options_strategy_versions_strategy_key_version")
        ),
    )
    with op.batch_alter_table("options_strategy_versions", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_strategy_versions_genome_id"), ["genome_id"], unique=False
        )
        batch_op.create_index("ix_options_strategy_versions_stage", ["stage"], unique=False)

    op.create_table(
        "options_greeks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("quote_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column("gamma", sa.Float(), nullable=True),
        sa.Column("theta", sa.Float(), nullable=True),
        sa.Column("vega", sa.Float(), nullable=True),
        sa.Column("rho", sa.Float(), nullable=True),
        sa.Column("iv", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["quote_id"],
            ["options_quotes.id"],
            name=op.f("fk_options_greeks_quote_id_options_quotes"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_greeks")),
    )
    with op.batch_alter_table("options_greeks", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_options_greeks_quote_id"), ["quote_id"], unique=False)
        batch_op.create_index("ix_options_greeks_symbol_at", ["symbol", "at"], unique=False)

    op.create_table(
        "options_hypotheses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("source", sa.String(length=24), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("parent_version_id", sa.Integer(), nullable=True),
        sa.Column("rationale", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_hypotheses_parent_version_id_options_strategy_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_hypotheses")),
    )
    op.create_table(
        "options_strategy_backtests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("run_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("purpose", sa.String(length=16), nullable=False),
        sa.Column("data_source", sa.String(length=16), nullable=False),
        sa.Column("execution_model", sa.String(length=16), nullable=False),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column("period_end", sa.Date(), nullable=False),
        sa.Column("universe", sa.JSON(), nullable=False),
        sa.Column("trades", sa.Integer(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_backtests_version_id_options_strategy_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_backtests")),
    )
    with op.batch_alter_table("options_strategy_backtests", schema=None) as batch_op:
        batch_op.create_index(
            "ix_options_strategy_backtests_version_run", ["version_id", "run_at"], unique=False
        )

    op.create_table(
        "options_strategy_decay",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_decay_version_id_options_strategy_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_decay")),
    )
    with op.batch_alter_table("options_strategy_decay", schema=None) as batch_op:
        batch_op.create_index("ix_options_strategy_decay_version_at", ["version_id", "at"], unique=False)

    op.create_table(
        "options_strategy_regimes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("trades", sa.Integer(), nullable=False),
        sa.Column("expectancy", sa.Float(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_regimes_version_id_options_strategy_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_regimes")),
        sa.UniqueConstraint(
            "version_id",
            "regime",
            "source",
            name=op.f("uq_options_strategy_regimes_version_id_regime_source"),
        ),
    )
    with op.batch_alter_table("options_strategy_regimes", schema=None) as batch_op:
        batch_op.create_index("ix_options_strategy_regimes_regime", ["regime"], unique=False)

    op.create_table(
        "options_strategy_rules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.Integer(), nullable=False),
        sa.Column("genome_id", sa.Integer(), nullable=True),
        sa.Column("rule_type", sa.String(length=24), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("expression", sa.JSON(), nullable=False),
        sa.Column("explicit", sa.Boolean(), nullable=False),
        sa.Column("assumed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["options_strategy_claims.id"],
            name=op.f("fk_options_strategy_rules_claim_id_options_strategy_claims"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["genome_id"],
            ["options_strategy_genomes.id"],
            name=op.f("fk_options_strategy_rules_genome_id_options_strategy_genomes"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_rules")),
    )
    with op.batch_alter_table("options_strategy_rules", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_options_strategy_rules_claim_id"), ["claim_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_options_strategy_rules_genome_id"), ["genome_id"], unique=False)

    op.create_table(
        "options_strategy_scores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("scored_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("dimensions", sa.JSON(), nullable=False),
        sa.Column("overfit_risk", sa.Float(), nullable=True),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_scores_version_id_options_strategy_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_scores")),
    )
    with op.batch_alter_table("options_strategy_scores", schema=None) as batch_op:
        batch_op.create_index(
            "ix_options_strategy_scores_version_at", ["version_id", "scored_at"], unique=False
        )

    op.create_table(
        "options_strategy_stress_tests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("run_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("scenarios", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_stress_tests_version_id_options_strategy_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_stress_tests")),
    )
    with op.batch_alter_table("options_strategy_stress_tests", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_strategy_stress_tests_version_id"), ["version_id"], unique=False
        )

    op.create_table(
        "options_strategy_walkforwards",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("run_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("data_source", sa.String(length=16), nullable=False),
        sa.Column("windows", sa.JSON(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_strategy_walkforwards_version_id_options_strategy_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_strategy_walkforwards")),
    )
    with op.batch_alter_table("options_strategy_walkforwards", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_strategy_walkforwards_version_id"), ["version_id"], unique=False
        )

    op.create_table(
        "options_trade_candidates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_key", sa.String(length=64), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=True),
        sa.Column("family", sa.String(length=32), nullable=False),
        sa.Column("structure_key", sa.String(length=200), nullable=False),
        sa.Column("structure", sa.JSON(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=True),
        sa.Column("dte", sa.Integer(), nullable=True),
        sa.Column("iv_rank", sa.Float(), nullable=True),
        sa.Column("mode", sa.String(length=12), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("gate", sa.String(length=48), nullable=True),
        sa.Column("reject_reason", sa.Text(), nullable=True),
        sa.Column("data_quality", sa.JSON(), nullable=False),
        sa.Column("audit", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_trade_candidates_version_id_options_strategy_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_trade_candidates")),
    )
    with op.batch_alter_table("options_trade_candidates", schema=None) as batch_op:
        batch_op.create_index("ix_options_trade_candidates_status", ["status"], unique=False)
        batch_op.create_index(
            "ix_options_trade_candidates_underlying_created", ["underlying", "created_at"], unique=False
        )
        batch_op.create_index("ix_options_trade_candidates_version", ["version_id"], unique=False)

    op.create_table(
        "options_experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("hypothesis_id", sa.Integer(), nullable=True),
        sa.Column("parent_version_id", sa.Integer(), nullable=True),
        sa.Column("child_version_id", sa.Integer(), nullable=True),
        sa.Column("feature_changes", sa.JSON(), nullable=False),
        sa.Column("parameter_changes", sa.JSON(), nullable=False),
        sa.Column("dataset", sa.JSON(), nullable=False),
        sa.Column("train_period", sa.String(length=32), nullable=True),
        sa.Column("test_period", sa.String(length=32), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("priority", sa.Float(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("started_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("finished_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["child_version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_experiments_child_version_id_options_strategy_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["hypothesis_id"],
            ["options_hypotheses.id"],
            name=op.f("fk_options_experiments_hypothesis_id_options_hypotheses"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_experiments_parent_version_id_options_strategy_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_experiments")),
    )
    with op.batch_alter_table("options_experiments", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_experiments_hypothesis_id"), ["hypothesis_id"], unique=False
        )
        batch_op.create_index("ix_options_experiments_status_priority", ["status", "priority"], unique=False)

    op.create_table(
        "options_missed_opportunities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("family", sa.String(length=32), nullable=False),
        sa.Column("reject_reason", sa.Text(), nullable=False),
        sa.Column("gate", sa.String(length=48), nullable=True),
        sa.Column("strategy_confidence", sa.Float(), nullable=True),
        sa.Column("grade_after", sa.Date(), nullable=False),
        sa.Column("outcome_pnl", sa.Float(), nullable=True),
        sa.Column("underlying_return", sa.Float(), nullable=True),
        sa.Column("classification", sa.String(length=32), nullable=True),
        sa.Column("graded_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["options_trade_candidates.id"],
            name=op.f("fk_options_missed_opportunities_candidate_id_options_trade_candidates"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_missed_opportunities")),
    )
    with op.batch_alter_table("options_missed_opportunities", schema=None) as batch_op:
        batch_op.create_index("ix_options_missed_classification", ["classification"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_options_missed_opportunities_candidate_id"), ["candidate_id"], unique=False
        )

    op.create_table(
        "options_trade_theses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("market_regime", sa.String(length=32), nullable=True),
        sa.Column("iv_regime", sa.String(length=32), nullable=True),
        sa.Column("thesis", sa.Text(), nullable=False),
        sa.Column("body", sa.JSON(), nullable=False),
        sa.Column("debate", sa.JSON(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["options_trade_candidates.id"],
            name=op.f("fk_options_trade_theses_candidate_id_options_trade_candidates"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_trade_theses")),
    )
    with op.batch_alter_table("options_trade_theses", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_trade_theses_candidate_id"), ["candidate_id"], unique=False
        )

    op.create_table(
        "options_positions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_id", sa.Integer(), nullable=True),
        sa.Column("thesis_id", sa.Integer(), nullable=True),
        sa.Column("version_id", sa.Integer(), nullable=True),
        sa.Column("underlying", sa.String(length=16), nullable=False),
        sa.Column("family", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=12), nullable=False),
        sa.Column("structure", sa.JSON(), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("expiry_state", sa.String(length=20), nullable=False),
        sa.Column("first_expiration", sa.Date(), nullable=True),
        sa.Column("opened_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("entry_value", sa.Float(), nullable=False),
        sa.Column("entry_mid", sa.Float(), nullable=True),
        sa.Column("entry_underlying", sa.Float(), nullable=False),
        sa.Column("entry_iv", sa.Float(), nullable=True),
        sa.Column("entry_greeks", sa.JSON(), nullable=False),
        sa.Column("max_loss", sa.Float(), nullable=False),
        sa.Column("max_profit", sa.Float(), nullable=True),
        sa.Column("marks", sa.JSON(), nullable=False),
        sa.Column("client_order_id", sa.String(length=64), nullable=True),
        sa.Column("exit_client_order_id", sa.String(length=64), nullable=True),
        sa.Column("closed_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("exit_value", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.Text(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=True),
        sa.Column("attribution", sa.JSON(), nullable=False),
        sa.Column("critique", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["options_trade_candidates.id"],
            name=op.f("fk_options_positions_candidate_id_options_trade_candidates"),
        ),
        sa.ForeignKeyConstraint(
            ["thesis_id"],
            ["options_trade_theses.id"],
            name=op.f("fk_options_positions_thesis_id_options_trade_theses"),
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["options_strategy_versions.id"],
            name=op.f("fk_options_positions_version_id_options_strategy_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_positions")),
        sa.UniqueConstraint("client_order_id", name=op.f("uq_options_positions_client_order_id")),
        sa.UniqueConstraint("exit_client_order_id", name=op.f("uq_options_positions_exit_client_order_id")),
    )
    with op.batch_alter_table("options_positions", schema=None) as batch_op:
        batch_op.create_index("ix_options_positions_first_expiration", ["first_expiration"], unique=False)
        batch_op.create_index(
            "ix_options_positions_underlying_status", ["underlying", "status"], unique=False
        )
        batch_op.create_index("ix_options_positions_version", ["version_id"], unique=False)

    op.create_table(
        "options_assignment_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("contracts", sa.Integer(), nullable=False),
        sa.Column("share_delivery", sa.Integer(), nullable=False),
        sa.Column("cash_flow", sa.Float(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["options_positions.id"],
            name=op.f("fk_options_assignment_events_position_id_options_positions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_assignment_events")),
    )
    with op.batch_alter_table("options_assignment_events", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_assignment_events_position_id"), ["position_id"], unique=False
        )

    op.create_table(
        "options_counterfactuals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("candidate_id", sa.Integer(), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("alternative", sa.String(length=48), nullable=False),
        sa.Column("structure", sa.JSON(), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("pnl_on_risk", sa.Float(), nullable=True),
        sa.Column("chosen_pnl", sa.Float(), nullable=True),
        sa.Column("better_than_chosen", sa.Boolean(), nullable=True),
        sa.Column("data_source", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["options_trade_candidates.id"],
            name=op.f("fk_options_counterfactuals_candidate_id_options_trade_candidates"),
        ),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["options_positions.id"],
            name=op.f("fk_options_counterfactuals_position_id_options_positions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_counterfactuals")),
    )
    with op.batch_alter_table("options_counterfactuals", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_counterfactuals_position_id"), ["position_id"], unique=False
        )

    op.create_table(
        "options_execution_ledger",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("candidate_id", sa.Integer(), nullable=True),
        sa.Column("client_order_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=8), nullable=False),
        sa.Column("legs", sa.JSON(), nullable=False),
        sa.Column("decision_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("quote_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("submitted_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("filled_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("decision_price", sa.Float(), nullable=True),
        sa.Column("mid", sa.Float(), nullable=True),
        sa.Column("bid", sa.Float(), nullable=True),
        sa.Column("ask", sa.Float(), nullable=True),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("fill_price", sa.Float(), nullable=True),
        sa.Column("expected_price", sa.Float(), nullable=True),
        sa.Column("slippage_dollars", sa.Float(), nullable=True),
        sa.Column("slippage_bps", sa.Float(), nullable=True),
        sa.Column("spread", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.ForeignKeyConstraint(
            ["candidate_id"],
            ["options_trade_candidates.id"],
            name=op.f("fk_options_execution_ledger_candidate_id_options_trade_candidates"),
        ),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["options_positions.id"],
            name=op.f("fk_options_execution_ledger_position_id_options_positions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_execution_ledger")),
        sa.UniqueConstraint("client_order_id", name=op.f("uq_options_execution_ledger_client_order_id")),
    )
    with op.batch_alter_table("options_execution_ledger", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_execution_ledger_position_id"), ["position_id"], unique=False
        )

    op.create_table(
        "options_exercise_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=True),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("contracts", sa.Integer(), nullable=False),
        sa.Column("share_delivery", sa.Integer(), nullable=False),
        sa.Column("cash_flow", sa.Float(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["options_positions.id"],
            name=op.f("fk_options_exercise_events_position_id_options_positions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_exercise_events")),
    )
    with op.batch_alter_table("options_exercise_events", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_exercise_events_position_id"), ["position_id"], unique=False
        )

    op.create_table(
        "options_position_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position_id", sa.Integer(), nullable=False),
        sa.Column("at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["options_positions.id"],
            name=op.f("fk_options_position_events_position_id_options_positions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_options_position_events")),
    )
    with op.batch_alter_table("options_position_events", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_options_position_events_position_id"), ["position_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("options_position_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_position_events_position_id"))

    op.drop_table("options_position_events")
    with op.batch_alter_table("options_exercise_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_exercise_events_position_id"))

    op.drop_table("options_exercise_events")
    with op.batch_alter_table("options_execution_ledger", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_execution_ledger_position_id"))

    op.drop_table("options_execution_ledger")
    with op.batch_alter_table("options_counterfactuals", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_counterfactuals_position_id"))

    op.drop_table("options_counterfactuals")
    with op.batch_alter_table("options_assignment_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_assignment_events_position_id"))

    op.drop_table("options_assignment_events")
    with op.batch_alter_table("options_positions", schema=None) as batch_op:
        batch_op.drop_index("ix_options_positions_version")
        batch_op.drop_index("ix_options_positions_underlying_status")
        batch_op.drop_index("ix_options_positions_first_expiration")

    op.drop_table("options_positions")
    with op.batch_alter_table("options_trade_theses", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_trade_theses_candidate_id"))

    op.drop_table("options_trade_theses")
    with op.batch_alter_table("options_missed_opportunities", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_missed_opportunities_candidate_id"))
        batch_op.drop_index("ix_options_missed_classification")

    op.drop_table("options_missed_opportunities")
    with op.batch_alter_table("options_experiments", schema=None) as batch_op:
        batch_op.drop_index("ix_options_experiments_status_priority")
        batch_op.drop_index(batch_op.f("ix_options_experiments_hypothesis_id"))

    op.drop_table("options_experiments")
    with op.batch_alter_table("options_trade_candidates", schema=None) as batch_op:
        batch_op.drop_index("ix_options_trade_candidates_version")
        batch_op.drop_index("ix_options_trade_candidates_underlying_created")
        batch_op.drop_index("ix_options_trade_candidates_status")

    op.drop_table("options_trade_candidates")
    with op.batch_alter_table("options_strategy_walkforwards", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_strategy_walkforwards_version_id"))

    op.drop_table("options_strategy_walkforwards")
    with op.batch_alter_table("options_strategy_stress_tests", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_strategy_stress_tests_version_id"))

    op.drop_table("options_strategy_stress_tests")
    with op.batch_alter_table("options_strategy_scores", schema=None) as batch_op:
        batch_op.drop_index("ix_options_strategy_scores_version_at")

    op.drop_table("options_strategy_scores")
    with op.batch_alter_table("options_strategy_rules", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_strategy_rules_genome_id"))
        batch_op.drop_index(batch_op.f("ix_options_strategy_rules_claim_id"))

    op.drop_table("options_strategy_rules")
    with op.batch_alter_table("options_strategy_regimes", schema=None) as batch_op:
        batch_op.drop_index("ix_options_strategy_regimes_regime")

    op.drop_table("options_strategy_regimes")
    with op.batch_alter_table("options_strategy_decay", schema=None) as batch_op:
        batch_op.drop_index("ix_options_strategy_decay_version_at")

    op.drop_table("options_strategy_decay")
    with op.batch_alter_table("options_strategy_backtests", schema=None) as batch_op:
        batch_op.drop_index("ix_options_strategy_backtests_version_run")

    op.drop_table("options_strategy_backtests")
    op.drop_table("options_hypotheses")
    with op.batch_alter_table("options_greeks", schema=None) as batch_op:
        batch_op.drop_index("ix_options_greeks_symbol_at")
        batch_op.drop_index(batch_op.f("ix_options_greeks_quote_id"))

    op.drop_table("options_greeks")
    with op.batch_alter_table("options_strategy_versions", schema=None) as batch_op:
        batch_op.drop_index("ix_options_strategy_versions_stage")
        batch_op.drop_index(batch_op.f("ix_options_strategy_versions_genome_id"))

    op.drop_table("options_strategy_versions")
    with op.batch_alter_table("options_strategy_claims", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_strategy_claims_source_id"))

    op.drop_table("options_strategy_claims")
    with op.batch_alter_table("options_quotes", schema=None) as batch_op:
        batch_op.drop_index("ix_options_quotes_underlying_expiration")
        batch_op.drop_index("ix_options_quotes_symbol_quote_at")
        batch_op.drop_index(batch_op.f("ix_options_quotes_snapshot_id"))

    op.drop_table("options_quotes")
    with op.batch_alter_table("options_trades", schema=None) as batch_op:
        batch_op.drop_index("ix_options_trades_symbol_at")

    op.drop_table("options_trades")
    op.drop_table("options_strategy_weights")
    op.drop_table("options_strategy_sources")
    with op.batch_alter_table("options_strategy_genomes", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_options_strategy_genomes_family"))

    op.drop_table("options_strategy_genomes")
    with op.batch_alter_table("options_lessons", schema=None) as batch_op:
        batch_op.drop_index("ix_options_lessons_memory_status")

    op.drop_table("options_lessons")
    with op.batch_alter_table("options_learning_events", schema=None) as batch_op:
        batch_op.drop_index("ix_options_learning_events_kind_at")

    op.drop_table("options_learning_events")
    with op.batch_alter_table("options_knowledge_edges", schema=None) as batch_op:
        batch_op.drop_index("ix_options_knowledge_edges_dst")

    op.drop_table("options_knowledge_edges")
    with op.batch_alter_table("options_iv_history", schema=None) as batch_op:
        batch_op.drop_index("ix_options_iv_history_iv_rank")

    op.drop_table("options_iv_history")
    op.drop_table("options_generation_runs")
    with op.batch_alter_table("options_feature_observations", schema=None) as batch_op:
        batch_op.drop_index("ix_options_feature_observations_underlying_day")

    op.drop_table("options_feature_observations")
    op.drop_table("options_feature_importance")
    with op.batch_alter_table("options_contracts", schema=None) as batch_op:
        batch_op.drop_index("ix_options_contracts_underlying_expiration")
        batch_op.drop_index(batch_op.f("ix_options_contracts_expiration"))

    op.drop_table("options_contracts")
    with op.batch_alter_table("options_chain_snapshots", schema=None) as batch_op:
        batch_op.drop_index("ix_options_chain_snapshots_underlying_fetched")

    op.drop_table("options_chain_snapshots")
