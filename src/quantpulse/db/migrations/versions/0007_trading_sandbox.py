"""Trading sandbox: paper accounts, positions, trades, equity snapshots and the agent's journal.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _fk(table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["account_id"],
        ["sandbox_accounts.id"],
        name=op.f(f"fk_{table}_account_id_sandbox_accounts"),
        ondelete="CASCADE",
    )


def upgrade() -> None:
    op.create_table(
        "sandbox_accounts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("mode", sa.String(length=10), nullable=False),
        sa.Column("starting_cash", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("auto_trade", sa.Boolean(), nullable=False),
        sa.Column("allow_synthetic", sa.Boolean(), nullable=False),
        sa.Column("strategy", sa.JSON(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sandbox_accounts")),
        sa.UniqueConstraint("name", name=op.f("uq_sandbox_accounts_name")),
    )
    op.create_table(
        "sandbox_positions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("avg_cost", sa.Float(), nullable=False),
        _fk("sandbox_positions"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sandbox_positions")),
        sa.UniqueConstraint("account_id", "symbol", name=op.f("uq_sandbox_positions_account_id_symbol")),
    )
    op.create_index(op.f("ix_sandbox_positions_account_id"), "sandbox_positions", ["account_id"])
    op.create_table(
        "sandbox_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("executed_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("reference_price", sa.Float(), nullable=False),
        sa.Column("commission", sa.Float(), nullable=False),
        sa.Column("realized_pnl", sa.Float(), nullable=False),
        sa.Column("data_status", sa.String(length=10), nullable=False),
        sa.Column("source", sa.String(length=10), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        _fk("sandbox_trades"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sandbox_trades")),
    )
    op.create_index(
        "ix_sandbox_trades_account_id_executed_at", "sandbox_trades", ["account_id", "executed_at"]
    )
    op.create_table(
        "sandbox_equity",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("recorded_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("benchmark_price", sa.Float(), nullable=True),
        sa.Column("data_status", sa.String(length=10), nullable=False),
        _fk("sandbox_equity"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sandbox_equity")),
    )
    op.create_index(
        "ix_sandbox_equity_account_id_recorded_at", "sandbox_equity", ["account_id", "recorded_at"]
    )
    op.create_table(
        "sandbox_journal",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        _fk("sandbox_journal"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sandbox_journal")),
    )
    op.create_index(
        "ix_sandbox_journal_account_id_created_at", "sandbox_journal", ["account_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_sandbox_journal_account_id_created_at", table_name="sandbox_journal")
    op.drop_table("sandbox_journal")
    op.drop_index("ix_sandbox_equity_account_id_recorded_at", table_name="sandbox_equity")
    op.drop_table("sandbox_equity")
    op.drop_index("ix_sandbox_trades_account_id_executed_at", table_name="sandbox_trades")
    op.drop_table("sandbox_trades")
    op.drop_index(op.f("ix_sandbox_positions_account_id"), table_name="sandbox_positions")
    op.drop_table("sandbox_positions")
    op.drop_table("sandbox_accounts")
