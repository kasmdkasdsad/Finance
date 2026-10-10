"""Alpaca paper trading: strategy cycles, broker orders, the trading audit trail and trading state.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "trading_cycles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cycle_key", sa.String(length=48), nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False),
        sa.Column("mode", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=12), nullable=False),
        sa.Column("started_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("finished_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("skip_reason", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("equity", sa.Float(), nullable=True),
        sa.Column("last_equity", sa.Float(), nullable=True),
        sa.Column("cash", sa.Float(), nullable=True),
        sa.Column("buying_power", sa.Float(), nullable=True),
        sa.Column("long_market_value", sa.Float(), nullable=True),
        sa.Column("data_status", sa.String(length=10), nullable=True),
        sa.Column("regime", sa.JSON(), nullable=False),
        sa.Column("positions", sa.JSON(), nullable=False),
        sa.Column("signals", sa.JSON(), nullable=False),
        sa.Column("targets", sa.JSON(), nullable=False),
        sa.Column("trades", sa.JSON(), nullable=False),
        sa.Column("plan", sa.JSON(), nullable=False),
        sa.Column("notes", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_trading_cycles")),
        sa.UniqueConstraint("cycle_key", name=op.f("uq_trading_cycles_cycle_key")),
    )
    op.create_index("ix_trading_cycles_started_at", "trading_cycles", ["started_at"])
    op.create_table(
        "trading_state",
        sa.Column("key", sa.String(length=40), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_trading_state")),
    )
    op.create_table(
        "broker_orders",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("client_order_id", sa.String(length=128), nullable=False),
        sa.Column("alpaca_order_id", sa.String(length=64), nullable=True),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("notional", sa.Float(), nullable=True),
        sa.Column("order_type", sa.String(length=20), nullable=False),
        sa.Column("time_in_force", sa.String(length=8), nullable=False),
        sa.Column("limit_price", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("filled_quantity", sa.Float(), nullable=False),
        sa.Column("average_fill_price", sa.Float(), nullable=True),
        sa.Column("submitted_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("filled_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("canceled_at", quantpulse.db.base.UTCDateTime(), nullable=True),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=True),
        sa.Column("signal_score", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("updated_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["trading_cycles.id"],
            name=op.f("fk_broker_orders_cycle_id_trading_cycles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_broker_orders")),
        sa.UniqueConstraint("client_order_id", name=op.f("uq_broker_orders_client_order_id")),
    )
    op.create_index(op.f("ix_broker_orders_alpaca_order_id"), "broker_orders", ["alpaca_order_id"])
    op.create_index(op.f("ix_broker_orders_cycle_id"), "broker_orders", ["cycle_id"])
    op.create_index("ix_broker_orders_status", "broker_orders", ["status"])
    op.create_index("ix_broker_orders_symbol_created_at", "broker_orders", ["symbol", "created_at"])
    op.create_table(
        "trading_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("cycle_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("client_order_id", sa.String(length=128), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cycle_id"],
            ["trading_cycles.id"],
            name=op.f("fk_trading_events_cycle_id_trading_cycles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_trading_events")),
    )
    op.create_index("ix_trading_events_created_at", "trading_events", ["created_at"])
    op.create_index(op.f("ix_trading_events_cycle_id"), "trading_events", ["cycle_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_trading_events_cycle_id"), table_name="trading_events")
    op.drop_index("ix_trading_events_created_at", table_name="trading_events")
    op.drop_table("trading_events")
    op.drop_index("ix_broker_orders_symbol_created_at", table_name="broker_orders")
    op.drop_index("ix_broker_orders_status", table_name="broker_orders")
    op.drop_index(op.f("ix_broker_orders_cycle_id"), table_name="broker_orders")
    op.drop_index(op.f("ix_broker_orders_alpaca_order_id"), table_name="broker_orders")
    op.drop_table("broker_orders")
    op.drop_table("trading_state")
    op.drop_index("ix_trading_cycles_started_at", table_name="trading_cycles")
    op.drop_table("trading_cycles")
