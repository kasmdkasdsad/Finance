"""Vehicle asset lifecycle: vehicles, telemetry, fuel logs, maintenance, regional fuel prices.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import quantpulse.db.base

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vehicles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nickname", sa.String(length=80), nullable=False),
        sa.Column("profile_id", sa.String(length=80), nullable=False),
        sa.Column("purchase_price", sa.Float(), nullable=True),
        sa.Column("purchase_date", sa.Date(), nullable=False),
        sa.Column("purchase_odometer", sa.Float(), nullable=False),
        sa.Column("annual_miles", sa.Float(), nullable=False),
        sa.Column("city_share", sa.Float(), nullable=False),
        sa.Column("fuel_region", sa.String(length=16), nullable=True),
        sa.Column("created_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vehicles")),
    )

    op.create_table(
        "telemetry_readings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vehicle_id", sa.Integer(), nullable=False),
        sa.Column("recorded_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("odometer", sa.Float(), nullable=False),
        sa.Column("fuel_level_pct", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.ForeignKeyConstraint(
            ["vehicle_id"],
            ["vehicles.id"],
            name=op.f("fk_telemetry_readings_vehicle_id_vehicles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_telemetry_readings")),
    )
    op.create_index(
        "ix_telemetry_readings_vehicle_id_recorded_at", "telemetry_readings", ["vehicle_id", "recorded_at"]
    )

    op.create_table(
        "fuel_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vehicle_id", sa.Integer(), nullable=False),
        sa.Column("filled_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.Column("odometer", sa.Float(), nullable=False),
        sa.Column("gallons", sa.Float(), nullable=False),
        sa.Column("price_per_gallon", sa.Float(), nullable=False),
        sa.Column("full_tank", sa.Boolean(), nullable=False),
        sa.Column("station", sa.String(length=120), nullable=True),
        sa.ForeignKeyConstraint(
            ["vehicle_id"], ["vehicles.id"], name=op.f("fk_fuel_logs_vehicle_id_vehicles"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fuel_logs")),
    )
    op.create_index("ix_fuel_logs_vehicle_id_filled_at", "fuel_logs", ["vehicle_id", "filled_at"])

    op.create_table(
        "maintenance_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vehicle_id", sa.Integer(), nullable=False),
        sa.Column("service_code", sa.String(length=40), nullable=False),
        sa.Column("performed_on", sa.Date(), nullable=False),
        sa.Column("odometer", sa.Float(), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["vehicle_id"],
            ["vehicles.id"],
            name=op.f("fk_maintenance_records_vehicle_id_vehicles"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_maintenance_records")),
    )
    op.create_index(
        "ix_maintenance_records_vehicle_id_service_code",
        "maintenance_records",
        ["vehicle_id", "service_code"],
    )

    op.create_table(
        "fuel_price_observations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("region", sa.String(length=16), nullable=False),
        sa.Column("region_name", sa.String(length=80), nullable=False),
        sa.Column("grade", sa.String(length=16), nullable=False),
        sa.Column("period", sa.Date(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("series_id", sa.String(length=40), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("ingested_at", quantpulse.db.base.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_fuel_price_observations")),
        sa.UniqueConstraint(
            "region", "grade", "period", name=op.f("uq_fuel_price_observations_region_grade_period")
        ),
    )


def downgrade() -> None:
    op.drop_table("fuel_price_observations")
    op.drop_index("ix_maintenance_records_vehicle_id_service_code", table_name="maintenance_records")
    op.drop_table("maintenance_records")
    op.drop_index("ix_fuel_logs_vehicle_id_filled_at", table_name="fuel_logs")
    op.drop_table("fuel_logs")
    op.drop_index("ix_telemetry_readings_vehicle_id_recorded_at", table_name="telemetry_readings")
    op.drop_table("telemetry_readings")
    op.drop_table("vehicles")
