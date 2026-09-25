"""ORM models for the QuantPulse data warehouse.

Each group of tables is introduced by its own Alembic revision (see ``db/migrations/versions``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from quantpulse.core.clock import utcnow
from quantpulse.db.base import Base, UTCDateTime

# ----------------------------------------------------------------------------- 0001 market core


class PriceBarRow(Base):
    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "ts"),
        Index("ix_price_bars_symbol_interval_ts", "symbol", "interval", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16))
    interval: Mapped[str] = mapped_column(String(8))
    ts: Mapped[datetime] = mapped_column(UTCDateTime())
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    provider: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class QuoteSnapshotRow(Base):
    __tablename__ = "quote_snapshots"
    __table_args__ = (Index("ix_quote_snapshots_symbol_quoted_at", "symbol", "quoted_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16))
    price: Mapped[float] = mapped_column(Float)
    previous_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    provider: Mapped[str] = mapped_column(String(32))
    quoted_at: Mapped[datetime] = mapped_column(UTCDateTime())
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class IngestionEventRow(Base):
    __tablename__ = "ingestion_events"
    __table_args__ = (Index("ix_ingestion_events_dataset_created_at", "dataset", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(40))
    key: Mapped[str] = mapped_column(String(160))
    provider: Mapped[str] = mapped_column(String(40))
    rows: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


# ----------------------------------------------------------------------------- 0002 rates & options


class YieldCurvePointRow(Base):
    __tablename__ = "yield_curve_points"
    __table_args__ = (UniqueConstraint("curve_date", "tenor"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    curve_date: Mapped[date] = mapped_column(Date, index=True)
    tenor: Mapped[str] = mapped_column(String(16))
    years: Mapped[float] = mapped_column(Float)
    rate: Mapped[float] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class OptionSnapshotRow(Base):
    __tablename__ = "option_snapshots"
    __table_args__ = (
        UniqueConstraint("contract_symbol", "snapshot_at"),
        Index("ix_option_snapshots_underlying_snapshot_at", "underlying", "snapshot_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    underlying: Mapped[str] = mapped_column(String(16))
    contract_symbol: Mapped[str] = mapped_column(String(40))
    kind: Mapped[str] = mapped_column(String(4))
    strike: Mapped[float] = mapped_column(Float)
    expiration: Mapped[date] = mapped_column(Date)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    last: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    underlying_price: Mapped[float] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(32))
    snapshot_at: Mapped[datetime] = mapped_column(UTCDateTime())


# ----------------------------------------------------------------------------- 0003 fundamentals


class CompanyRow(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True)
    cik: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    shares_outstanding: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares_as_of: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class FinancialStatementRow(Base):
    __tablename__ = "financial_statements"
    __table_args__ = (UniqueConstraint("symbol", "fiscal_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    fiscal_year: Mapped[int] = mapped_column(Integer)
    period_end: Mapped[date] = mapped_column(Date)
    form: Mapped[str] = mapped_column(String(12))
    filed: Mapped[date | None] = mapped_column(Date, nullable=True)
    accession: Mapped[str | None] = mapped_column(String(25), nullable=True)
    revenue: Mapped[float | None] = mapped_column(Float, nullable=True)
    gross_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    pretax_income: Mapped[float | None] = mapped_column(Float, nullable=True)
    income_tax: Mapped[float | None] = mapped_column(Float, nullable=True)
    interest_expense: Mapped[float | None] = mapped_column(Float, nullable=True)
    depreciation_amortization: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_liabilities: Mapped[float | None] = mapped_column(Float, nullable=True)
    stockholders_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_debt: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_assets: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_liabilities: Mapped[float | None] = mapped_column(Float, nullable=True)
    operating_cash_flow: Mapped[float | None] = mapped_column(Float, nullable=True)
    capital_expenditure: Mapped[float | None] = mapped_column(Float, nullable=True)
    diluted_eps: Mapped[float | None] = mapped_column(Float, nullable=True)
    diluted_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class SecFilingRow(Base):
    __tablename__ = "sec_filings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    cik: Mapped[str] = mapped_column(String(10))
    accession: Mapped[str] = mapped_column(String(25), unique=True)
    form: Mapped[str] = mapped_column(String(16))
    filing_date: Mapped[date] = mapped_column(Date)
    report_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    primary_document: Mapped[str | None] = mapped_column(String(200), nullable=True)
    url: Mapped[str | None] = mapped_column(String(400), nullable=True)


class EstimateSnapshotRow(Base):
    __tablename__ = "analyst_estimate_snapshots"
    __table_args__ = (Index("ix_analyst_estimate_snapshots_symbol_captured_at", "symbol", "captured_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    provider: Mapped[str] = mapped_column(String(32))
    captured_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


# ----------------------------------------------------------------------------- 0004 portfolio


class PortfolioRow(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
    holdings: Mapped[list[HoldingRow]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", lazy="selectin", order_by="HoldingRow.id"
    )


class HoldingRow(Base):
    __tablename__ = "holdings"
    __table_args__ = (UniqueConstraint("portfolio_id", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[float] = mapped_column(Float)
    cost_basis: Mapped[float | None] = mapped_column(Float, nullable=True)
    portfolio: Mapped[PortfolioRow] = relationship(back_populates="holdings")


# ----------------------------------------------------------------------------- 0005 vehicle


class VehicleRow(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nickname: Mapped[str] = mapped_column(String(80))
    profile_id: Mapped[str] = mapped_column(String(80))
    purchase_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    purchase_date: Mapped[date] = mapped_column(Date)
    purchase_odometer: Mapped[float] = mapped_column(Float, default=0.0)
    annual_miles: Mapped[float] = mapped_column(Float, default=12000.0)
    city_share: Mapped[float] = mapped_column(Float, default=0.55)
    fuel_region: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class TelemetryRow(Base):
    __tablename__ = "telemetry_readings"
    __table_args__ = (Index("ix_telemetry_readings_vehicle_id_recorded_at", "vehicle_id", "recorded_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicles.id", ondelete="CASCADE"))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())
    odometer: Mapped[float] = mapped_column(Float)
    fuel_level_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="manual")


class FuelLogRow(Base):
    __tablename__ = "fuel_logs"
    __table_args__ = (Index("ix_fuel_logs_vehicle_id_filled_at", "vehicle_id", "filled_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicles.id", ondelete="CASCADE"))
    filled_at: Mapped[datetime] = mapped_column(UTCDateTime())
    odometer: Mapped[float] = mapped_column(Float)
    gallons: Mapped[float] = mapped_column(Float)
    price_per_gallon: Mapped[float] = mapped_column(Float)
    full_tank: Mapped[bool] = mapped_column(Boolean, default=True)
    station: Mapped[str | None] = mapped_column(String(120), nullable=True)


class MaintenanceRecordRow(Base):
    __tablename__ = "maintenance_records"
    __table_args__ = (Index("ix_maintenance_records_vehicle_id_service_code", "vehicle_id", "service_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vehicle_id: Mapped[int] = mapped_column(ForeignKey("vehicles.id", ondelete="CASCADE"))
    service_code: Mapped[str] = mapped_column(String(40))
    performed_on: Mapped[date] = mapped_column(Date)
    odometer: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class FuelPriceRow(Base):
    __tablename__ = "fuel_price_observations"
    __table_args__ = (UniqueConstraint("region", "grade", "period"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region: Mapped[str] = mapped_column(String(16))
    region_name: Mapped[str] = mapped_column(String(80))
    grade: Mapped[str] = mapped_column(String(16))
    period: Mapped[date] = mapped_column(Date)
    price: Mapped[float] = mapped_column(Float)
    series_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    provider: Mapped[str] = mapped_column(String(32))
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


# ----------------------------------------------------------------------------- 0006 sports


class SportsGameRow(Base):
    __tablename__ = "sports_games"
    __table_args__ = (Index("ix_sports_games_league_season", "league", "season"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(24), unique=True)
    league: Mapped[str] = mapped_column(String(24))
    season: Mapped[int] = mapped_column(Integer)
    season_type: Mapped[int] = mapped_column(Integer)
    week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_time: Mapped[datetime] = mapped_column(UTCDateTime())
    home_team_id: Mapped[str] = mapped_column(String(16))
    away_team_id: Mapped[str] = mapped_column(String(16))
    home_name: Mapped[str] = mapped_column(String(80))
    away_name: Mapped[str] = mapped_column(String(80))
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state: Mapped[str] = mapped_column(String(8))
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class TeamRatingRow(Base):
    __tablename__ = "team_ratings"
    __table_args__ = (UniqueConstraint("league", "season", "team_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league: Mapped[str] = mapped_column(String(24))
    season: Mapped[int] = mapped_column(Integer)
    team_id: Mapped[str] = mapped_column(String(16))
    team_name: Mapped[str] = mapped_column(String(80))
    rating: Mapped[float] = mapped_column(Float)
    games: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    ties: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


# ----------------------------------------------------------------------------- 0007 trading sandbox


class SandboxAccountRow(Base):
    __tablename__ = "sandbox_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    mode: Mapped[str] = mapped_column(String(10))
    starting_cash: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    auto_trade: Mapped[bool] = mapped_column(Boolean, default=True)
    allow_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    strategy: Mapped[dict[str, Any]] = mapped_column(JSON)
    state: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class SandboxPositionRow(Base):
    __tablename__ = "sandbox_positions"
    __table_args__ = (UniqueConstraint("account_id", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("sandbox_accounts.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float] = mapped_column(Float)


class SandboxTradeRow(Base):
    __tablename__ = "sandbox_trades"
    __table_args__ = (Index("ix_sandbox_trades_account_id_executed_at", "account_id", "executed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("sandbox_accounts.id", ondelete="CASCADE"))
    executed_at: Mapped[datetime] = mapped_column(UTCDateTime())
    symbol: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    reference_price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    data_status: Mapped[str] = mapped_column(String(10))
    source: Mapped[str] = mapped_column(String(10))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class SandboxEquityRow(Base):
    __tablename__ = "sandbox_equity"
    __table_args__ = (Index("ix_sandbox_equity_account_id_recorded_at", "account_id", "recorded_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("sandbox_accounts.id", ondelete="CASCADE"))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    benchmark_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_status: Mapped[str] = mapped_column(String(10))


class SandboxJournalRow(Base):
    __tablename__ = "sandbox_journal"
    __table_args__ = (Index("ix_sandbox_journal_account_id_created_at", "account_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("sandbox_accounts.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    kind: Mapped[str] = mapped_column(String(16))
    summary: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSON)


class PredictionRow(Base):
    """One logged prediction and, once its target date has passed, how it turned out."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("symbol", "source", "horizon_days", "made_on"),
        Index("ix_predictions_status_target_date", "status", "target_date"),
        Index("ix_predictions_symbol_made_on", "symbol", "made_on"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    made_on: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date)
    symbol: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16))
    horizon_days: Mapped[int] = mapped_column(Integer)
    reference_price: Mapped[float] = mapped_column(Float)
    benchmark: Mapped[str] = mapped_column(String(16))
    benchmark_reference: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_up: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_outperform: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    q05: Mapped[float | None] = mapped_column(Float, nullable=True)
    q25: Mapped[float | None] = mapped_column(Float, nullable=True)
    q50: Mapped[float | None] = mapped_column(Float, nullable=True)
    q75: Mapped[float | None] = mapped_column(Float, nullable=True)
    q95: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_version: Mapped[str] = mapped_column(String(40))
    data_status: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(10), default="open")
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    realized_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_realized: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_up: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    outcome_outperform: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    in_50: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    in_90: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
