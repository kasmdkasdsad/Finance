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
    origin: Mapped[str] = mapped_column(String(10), default="live", server_default="live")
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


class CompanyProfileRow(Base):
    """SEC registrant profile (SIC and Fama-French sector) plus the scan window of its earnings events."""

    __tablename__ = "company_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True)
    cik: Mapped[str] = mapped_column(String(10))
    name: Mapped[str] = mapped_column(String(200))
    sic: Mapped[str | None] = mapped_column(String(8), nullable=True)
    sic_description: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sector: Mapped[str] = mapped_column(String(8))
    earnings_since: Mapped[date] = mapped_column(Date)
    provider: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class EarningsEventRow(Base):
    __tablename__ = "earnings_events"
    __table_args__ = (UniqueConstraint("symbol", "announced_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    announced_at: Mapped[datetime] = mapped_column(UTCDateTime())


class ReferenceBlobRow(Base):
    """Small reference datasets stored whole (e.g. the S&P 500 constituents and change log)."""

    __tablename__ = "reference_blobs"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    provider: Mapped[str] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)


class FundamentalFactRow(Base):
    """One XBRL value from an SEC frame: (tag, calendar frame, company) → value for the period."""

    __tablename__ = "fundamental_facts"
    __table_args__ = (
        UniqueConstraint("tag", "frame", "cik"),
        Index("ix_fundamental_facts_cik_tag", "cik", "tag"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tag: Mapped[str] = mapped_column(String(96))
    frame: Mapped[str] = mapped_column(String(12))
    cik: Mapped[int] = mapped_column(Integer)
    period_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    period_end: Mapped[date] = mapped_column(Date)
    value: Mapped[float] = mapped_column(Float)
    accn: Mapped[str] = mapped_column(String(25))


# ----------------------------------------------------------------------------- 0010 Alpaca paper trading


class TradingCycleRow(Base):
    """One strategy cycle against the Alpaca paper account (dry run or executed) and everything it saw."""

    __tablename__ = "trading_cycles"
    __table_args__ = (Index("ix_trading_cycles_started_at", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_key: Mapped[str] = mapped_column(String(48), unique=True)
    trigger: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(12))
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    skip_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash: Mapped[float | None] = mapped_column(Float, nullable=True)
    buying_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    long_market_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_status: Mapped[str | None] = mapped_column(String(10), nullable=True)
    regime: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    positions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    signals: Mapped[list[Any]] = mapped_column(JSON, default=list)
    targets: Mapped[list[Any]] = mapped_column(JSON, default=list)
    trades: Mapped[list[Any]] = mapped_column(JSON, default=list)
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[list[Any]] = mapped_column(JSON, default=list)


class BrokerOrderRow(Base):
    """An order QuantPulse sent to (or found on) the Alpaca paper account. Alpaca is authoritative; this
    row is kept in step with it by reconciliation."""

    __tablename__ = "broker_orders"
    __table_args__ = (
        Index("ix_broker_orders_symbol_created_at", "symbol", "created_at"),
        Index("ix_broker_orders_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(128), unique=True)
    alpaca_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("trading_cycles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_type: Mapped[str] = mapped_column(String(20))
    time_in_force: Mapped[str] = mapped_column(String(8), default="day")
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24))
    filled_quantity: Mapped[float] = mapped_column(Float, default=0.0)
    average_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    canceled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    strategy: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str | None] = mapped_column(String(24), nullable=True)
    signal_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class TradingEventRow(Base):
    """Audit trail: signals, proposals, risk decisions, orders, fills, kill switch, reconciliation."""

    __tablename__ = "trading_events"
    __table_args__ = (Index("ix_trading_events_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("trading_cycles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(40))
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True)
    client_order_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class TradingStateRow(Base):
    """Small persistent trading state (runtime kill switch, per-position memory, P/L baseline)."""

    __tablename__ = "trading_state"

    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


# ----------------------------------------------------------------------------- brain (multi-agent)
class BrainAgentRow(Base):
    """A registered agent (code-defined) and whether it is enabled; its spec is stored per version."""

    __tablename__ = "brain_agents"

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    family: Mapped[str] = mapped_column(String(16))
    version: Mapped[str] = mapped_column(String(16))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    registered_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)


class BrainCycleRow(Base):
    """One brain cycle: what it perceived, which agents it chose and why, and what it concluded."""

    __tablename__ = "brain_cycles"
    __table_args__ = (Index("ix_brain_cycles_started_at", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(24))
    trigger: Mapped[str] = mapped_column(String(16))
    session: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(12))
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    market: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    portfolio: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_quality: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    focus: Mapped[list[Any]] = mapped_column(JSON, default=list)
    agents: Mapped[list[Any]] = mapped_column(JSON, default=list)  # selected / skipped, with reasons
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class BrainAgentRunRow(Base):
    """One agent's run within a cycle: status, timing and cost (failures are recorded, never hidden)."""

    __tablename__ = "brain_agent_runs"
    __table_args__ = (Index("ix_brain_agent_runs_agent", "agent_id", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(ForeignKey("brain_cycles.id", ondelete="CASCADE"), index=True)
    agent_id: Mapped[str] = mapped_column(String(48))
    agent_version: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(12))  # ok | failed | timeout | skipped
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    subjects: Mapped[int] = mapped_column(Integer, default=0)
    opinions: Mapped[int] = mapped_column(Integer, default=0)
    model_tier: Mapped[str] = mapped_column(String(16))
    cost: Mapped[float] = mapped_column(Float, default=0.0)


class BrainOpinionRow(Base):
    """An agent's structured finding on one subject (a symbol, @market or @portfolio)."""

    __tablename__ = "brain_opinions"
    __table_args__ = (
        Index("ix_brain_opinions_subject", "subject", "created_at"),
        Index("ix_brain_opinions_agent", "agent_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(ForeignKey("brain_cycles.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("brain_agent_runs.id", ondelete="SET NULL"), nullable=True
    )
    agent_id: Mapped[str] = mapped_column(String(48))
    agent_version: Mapped[str] = mapped_column(String(16))
    subject: Mapped[str] = mapped_column(String(24))
    stance: Mapped[str] = mapped_column(String(10))
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    horizon_days: Mapped[int] = mapped_column(Integer)
    thesis: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list)
    data_missing: Mapped[list[Any]] = mapped_column(JSON, default=list)
    data_quality: Mapped[str] = mapped_column(String(16))
    invalidation: Mapped[str | None] = mapped_column(Text, nullable=True)
    veto: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainConsensusRow(Base):
    """The team's combined view on one subject, with the disagreement kept visible."""

    __tablename__ = "brain_consensus"
    __table_args__ = (Index("ix_brain_consensus_subject", "subject", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(ForeignKey("brain_cycles.id", ondelete="CASCADE"), index=True)
    subject: Mapped[str] = mapped_column(String(24))
    stance: Mapped[str] = mapped_column(String(10))
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    unknown: Mapped[bool] = mapped_column(Boolean, default=False)
    supporting: Mapped[int] = mapped_column(Integer, default=0)
    neutral: Mapped[int] = mapped_column(Integer, default=0)
    opposing: Mapped[int] = mapped_column(Integer, default=0)
    abstaining: Mapped[int] = mapped_column(Integer, default=0)
    disagreement: Mapped[float] = mapped_column(Float, default=0.0)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # sides, weights, primary conflict
    vetoes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    data_quality: Mapped[str] = mapped_column(String(16))
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainDecisionRow(Base):
    """A proposed portfolio action, the deterministic risk engine's verdict on it, and (later) what happened.
    The brain never sends an order itself."""

    __tablename__ = "brain_decisions"
    __table_args__ = (Index("ix_brain_decisions_subject", "subject", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(ForeignKey("brain_cycles.id", ondelete="CASCADE"), index=True)
    consensus_id: Mapped[int | None] = mapped_column(
        ForeignKey("brain_consensus.id", ondelete="SET NULL"), nullable=True
    )
    subject: Mapped[str] = mapped_column(String(24))
    action: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(24))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    quantity: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    rationale: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    risk_approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    risk: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    execution: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    outcome: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    evaluated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class BrainPredictionRow(Base):
    """A gradeable claim: direction of ``subject`` relative to the benchmark over ``horizon_days`` sessions.
    Written when made; the outcome columns stay empty until the horizon has passed and it is evaluated."""

    __tablename__ = "brain_predictions"
    __table_args__ = (
        Index("ix_brain_predictions_due", "status", "due_date"),
        Index("ix_brain_predictions_source", "source_type", "source_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("brain_cycles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    source_type: Mapped[str] = mapped_column(String(16))  # agent | consensus | decision | strategy
    source_id: Mapped[str] = mapped_column(String(48))
    source_version: Mapped[str] = mapped_column(String(16))
    subject: Mapped[str] = mapped_column(String(24))
    direction: Mapped[int] = mapped_column(Integer)  # +1 outperform, −1 underperform the benchmark
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    horizon_days: Mapped[int] = mapped_column(Integer)
    benchmark: Mapped[str] = mapped_column(String(16))
    regime: Mapped[str | None] = mapped_column(String(24), nullable=True)
    made_at: Mapped[datetime] = mapped_column(UTCDateTime())
    due_date: Mapped[date] = mapped_column(Date)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_benchmark: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(12))  # open | evaluated | void
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_relative: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class BrainReflectionRow(Base):
    """An append-only lesson about a decision, prediction or cycle (the original reasoning is never edited)."""

    __tablename__ = "brain_reflections"
    __table_args__ = (Index("ix_brain_reflections_subject", "subject_type", "subject_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(16))  # decision | prediction | cycle | system
    subject_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category: Mapped[str] = mapped_column(String(32))
    decision_quality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    outcome_quality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    questions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    lessons: Mapped[list[Any]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainAgentPerformanceRow(Base):
    """Measured track record of one agent version (optionally per regime) — computed only from evaluated
    predictions; nothing is written until there are observations."""

    __tablename__ = "brain_agent_performance"
    __table_args__ = (
        UniqueConstraint(
            "agent_id", "agent_version", "regime", "horizon_days", "window", name="uq_brain_agent_performance"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(48))
    agent_version: Mapped[str] = mapped_column(String(16))
    regime: Mapped[str] = mapped_column(String(24))  # "all" or a regime label
    horizon_days: Mapped[int] = mapped_column(Integer)
    window: Mapped[str] = mapped_column(String(16))  # e.g. "all", "90d"
    n: Mapped[int] = mapped_column(Integer)
    hits: Mapped[int] = mapped_column(Integer)
    hit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier: Mapped[float | None] = mapped_column(Float, nullable=True)
    ic: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration: Mapped[list[Any]] = mapped_column(JSON, default=list)
    reliability: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainImprovementRow(Base):
    """A proposed improvement (agent, data, routing, strategy) and its test result; never self-applied to
    risk controls."""

    __tablename__ = "brain_improvements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(24))
    target: Mapped[str] = mapped_column(String(48))
    title: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    proposal: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16))  # proposed | testing | validated | rejected | applied
    test_result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    decided_by: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainMemoryRow(Base):
    """Structured, searchable memory (short-term, working, long-term, strategy and agent tiers). Short
    summaries plus data — never raw model transcripts."""

    __tablename__ = "brain_memory"
    __table_args__ = (
        Index("ix_brain_memory_tier_subject", "tier", "subject"),
        Index("ix_brain_memory_key", "tier", "key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tier: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(24))
    subject: Mapped[str] = mapped_column(String(24))
    key: Mapped[str | None] = mapped_column(String(96), nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tags: Mapped[list[Any]] = mapped_column(JSON, default=list)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("brain_cycles.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class BrainEventRow(Base):
    """Something that happened (a quote went stale, a regime changed, an order filled, a prediction matured)."""

    __tablename__ = "brain_events"
    __table_args__ = (Index("ix_brain_events_type_created", "type", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(40))
    subject: Mapped[str | None] = mapped_column(String(24), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("brain_cycles.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainStateRow(Base):
    """Brain controls and scheduler state (started / paused, last job runs)."""

    __tablename__ = "brain_state"

    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainOpportunityRow(Base):
    """An idea the brain found itself, and how far it got: detection → data validation → agents → research →
    bull/bear/devil's advocate → consensus → portfolio fit → risk preview."""

    __tablename__ = "brain_opportunities"
    __table_args__ = (Index("ix_brain_opportunities_kind_created", "kind", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(ForeignKey("brain_cycles.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    subject: Mapped[str] = mapped_column(String(48))
    symbols: Mapped[list[Any]] = mapped_column(JSON, default=list)
    direction: Mapped[int] = mapped_column(Integer, default=0)
    strength: Mapped[float] = mapped_column(Float)
    headline: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(24))
    stages: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BrainDebateRow(Base):
    """The adversarial review of one subject's consensus: bull case, bear case, devil's advocate."""

    __tablename__ = "brain_debates"
    __table_args__ = (Index("ix_brain_debates_subject", "subject", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_id: Mapped[int] = mapped_column(ForeignKey("brain_cycles.id", ondelete="CASCADE"), index=True)
    subject: Mapped[str] = mapped_column(String(24))
    stance_before: Mapped[str] = mapped_column(String(10))
    confidence_before: Mapped[float] = mapped_column(Float)
    confidence_after: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(24))
    bull: Mapped[list[Any]] = mapped_column(JSON, default=list)
    bear: Mapped[list[Any]] = mapped_column(JSON, default=list)
    objections: Mapped[list[Any]] = mapped_column(JSON, default=list)
    change_our_mind: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
