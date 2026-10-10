"""Tables of the options layer (revision 0023): market data, the research library, strategies and their
evidence, trade candidates and theses, positions and their lifecycle, execution quality, counterfactuals,
missed opportunities, learning, experiments and the knowledge graph.

Normalised where it matters (a contract, a strategy version, a position are single rows others point to),
JSON where a record is a document read whole (metrics, a thesis's arguments). Nothing here is ever rewritten
in place to change history: strategies get new versions, positions get events, lessons get new evidence.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from quantpulse.db.base import Base, UTCDateTime

SYM = 32  # an OCC symbol is at most 21 characters
U = 16  # an underlying


# ----------------------------------------------------------------------------- market data
class OptionsContractRow(Base):
    __tablename__ = "options_contracts"
    __table_args__ = (Index("ix_options_contracts_underlying_expiration", "underlying", "expiration"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(SYM), unique=True)
    underlying: Mapped[str] = mapped_column(String(U))
    expiration: Mapped[date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(4))
    strike: Mapped[float] = mapped_column(Float)
    multiplier: Mapped[int] = mapped_column(Integer, default=100)
    style: Mapped[str] = mapped_column(String(16), default="american")
    status: Mapped[str] = mapped_column(String(16), default="active")
    tradable: Mapped[bool] = mapped_column(Boolean, default=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime())
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsChainSnapshotRow(Base):
    __tablename__ = "options_chain_snapshots"
    __table_args__ = (Index("ix_options_chain_snapshots_underlying_fetched", "underlying", "fetched_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    underlying: Mapped[str] = mapped_column(String(U))
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime())
    feed: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(32))
    underlying_price: Mapped[float] = mapped_column(Float)
    underlying_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    contracts: Mapped[int] = mapped_column(Integer, default=0)
    quality: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsQuoteRow(Base):
    __tablename__ = "options_quotes"
    __table_args__ = (
        Index("ix_options_quotes_symbol_quote_at", "symbol", "quote_at"),
        Index("ix_options_quotes_underlying_expiration", "underlying", "expiration"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_chain_snapshots.id", ondelete="CASCADE"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(SYM))
    underlying: Mapped[str] = mapped_column(String(U))
    expiration: Mapped[date] = mapped_column(Date)
    kind: Mapped[str] = mapped_column(String(4))
    strike: Mapped[float] = mapped_column(Float)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    last: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    quote_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    feed: Mapped[str] = mapped_column(String(16))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsTradeRow(Base):
    """Trade prints from the market-data feed (not QuantPulse's own trades: see positions and the ledger)."""

    __tablename__ = "options_trades"
    __table_args__ = (Index("ix_options_trades_symbol_at", "symbol", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(SYM))
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    exchange: Mapped[str | None] = mapped_column(String(8), nullable=True)
    feed: Mapped[str] = mapped_column(String(16))
    recorded_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsGreeksRow(Base):
    __tablename__ = "options_greeks"
    __table_args__ = (Index("ix_options_greeks_symbol_at", "symbol", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    quote_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_quotes.id", ondelete="CASCADE"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(SYM))
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    gamma: Mapped[float | None] = mapped_column(Float, nullable=True)
    theta: Mapped[float | None] = mapped_column(Float, nullable=True)
    vega: Mapped[float | None] = mapped_column(Float, nullable=True)
    rho: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(16))  # vendor | model


class OptionsIVHistoryRow(Base):
    """One row per underlying per day: the implied-volatility standing the Brain decided with."""

    __tablename__ = "options_iv_history"
    __table_args__ = (
        UniqueConstraint("underlying", "day"),
        Index("ix_options_iv_history_iv_rank", "iv_rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    underlying: Mapped[str] = mapped_column(String(U))
    day: Mapped[date] = mapped_column(Date)
    atm_iv_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    rv_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    rv_60: Mapped[float | None] = mapped_column(Float, nullable=True)
    term_slope: Mapped[float | None] = mapped_column(Float, nullable=True)
    term_shape: Mapped[str | None] = mapped_column(String(16), nullable=True)
    skew_25d: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_move: Mapped[float | None] = mapped_column(Float, nullable=True)
    feed: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(32))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# ----------------------------------------------------------------------------- research library
class OptionsStrategySourceRow(Base):
    __tablename__ = "options_strategy_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_key: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(String(300))
    author: Mapped[str] = mapped_column(String(200))
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    # paper | index_methodology | book | letter | dataset | …
    source_type: Mapped[str] = mapped_column(String(32))
    quality: Mapped[str] = mapped_column(String(24))  # PRIMARY | ACADEMIC | REGULATORY | EXCHANGE | …
    reference: Mapped[str] = mapped_column(Text)
    market: Mapped[str] = mapped_column(String(120))
    time_period: Mapped[str] = mapped_column(String(64))
    limitations: Mapped[str] = mapped_column(Text, default="")
    evidence_grade: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reproducibility: Mapped[str] = mapped_column(String(32), default="unknown")
    status: Mapped[str] = mapped_column(String(32), default="UNVERIFIED_RESEARCH")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsStrategyClaimRow(Base):
    __tablename__ = "options_strategy_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("options_strategy_sources.id", ondelete="CASCADE"), index=True
    )
    claim: Mapped[str] = mapped_column(Text)
    assumptions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    # UNTESTED | TESTING | SUPPORTED | NOT_REPRODUCED | INCONCLUSIVE
    status: Mapped[str] = mapped_column(String(24), default="UNTESTED")
    # the claim's test: effect size, uncertainty
    test: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsStrategyRuleRow(Base):
    __tablename__ = "options_strategy_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("options_strategy_claims.id", ondelete="CASCADE"), index=True
    )
    genome_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_strategy_genomes.id"), nullable=True, index=True
    )
    rule_type: Mapped[str] = mapped_column(String(24))  # entry | exit | sizing | expiration | strike | filter
    text: Mapped[str] = mapped_column(Text)
    expression: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    explicit: Mapped[bool] = mapped_column(Boolean, default=False)
    # QuantPulse chose the value, the source did not
    assumed: Mapped[bool] = mapped_column(Boolean, default=False)


class OptionsStrategyGenomeRow(Base):
    __tablename__ = "options_strategy_genomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    genome_hash: Mapped[str] = mapped_column(String(64), unique=True)
    family: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(16))
    params: Mapped[dict[str, Any]] = mapped_column(JSON)
    parameter_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsStrategyVersionRow(Base):
    """A strategy version: immutable once created (a change is a new version, attributed to its reason)."""

    __tablename__ = "options_strategy_versions"
    __table_args__ = (
        UniqueConstraint("strategy_key", "version"),
        Index("ix_options_strategy_versions_stage", "stage"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_key: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(160))
    genome_id: Mapped[int] = mapped_column(ForeignKey("options_strategy_genomes.id"), index=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("options_strategy_versions.id"), nullable=True)
    second_parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_strategy_versions.id"), nullable=True
    )
    generation: Mapped[int] = mapped_column(Integer, default=0)
    # seed | extraction | mutation | crossover | regime | portfolio | baseline
    origin: Mapped[str] = mapped_column(String(24))
    source_id: Mapped[int | None] = mapped_column(ForeignKey("options_strategy_sources.id"), nullable=True)
    experiment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    stage: Mapped[str] = mapped_column(String(16), default="RESEARCH")
    stage_history: Mapped[list[Any]] = mapped_column(JSON, default=list)
    role: Mapped[str] = mapped_column(String(12), default="none")  # champion | challenger | none
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    stage_changed_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsStrategyBacktestRow(Base):
    __tablename__ = "options_strategy_backtests"
    __table_args__ = (Index("ix_options_strategy_backtests_version_run", "version_id", "run_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("options_strategy_versions.id", ondelete="CASCADE"))
    run_at: Mapped[datetime] = mapped_column(UTCDateTime())
    purpose: Mapped[str] = mapped_column(String(16))  # train | validate | test | full | baseline
    data_source: Mapped[str] = mapped_column(String(16))  # recorded | model
    # OPTIMISTIC | MIDPOINT | REALISTIC | PESSIMISTIC | STRESS
    execution_model: Mapped[str] = mapped_column(String(16))
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    universe: Mapped[list[Any]] = mapped_column(JSON, default=list)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    label: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsStrategyWalkforwardRow(Base):
    __tablename__ = "options_strategy_walkforwards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey("options_strategy_versions.id", ondelete="CASCADE"), index=True
    )
    run_at: Mapped[datetime] = mapped_column(UTCDateTime())
    data_source: Mapped[str] = mapped_column(String(16))
    windows: Mapped[list[Any]] = mapped_column(JSON, default=list)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)


class OptionsStrategyStressTestRow(Base):
    __tablename__ = "options_strategy_stress_tests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(
        ForeignKey("options_strategy_versions.id", ondelete="CASCADE"), index=True
    )
    run_at: Mapped[datetime] = mapped_column(UTCDateTime())
    kind: Mapped[str] = mapped_column(String(24))  # monte_carlo | tail | regime_transition | critic
    scenarios: Mapped[list[Any]] = mapped_column(JSON, default=list)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)


class OptionsStrategyScoreRow(Base):
    __tablename__ = "options_strategy_scores"
    __table_args__ = (Index("ix_options_strategy_scores_version_at", "version_id", "scored_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("options_strategy_versions.id", ondelete="CASCADE"))
    scored_at: Mapped[datetime] = mapped_column(UTCDateTime())
    dimensions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    overfit_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)


class OptionsStrategyRegimeRow(Base):
    __tablename__ = "options_strategy_regimes"
    __table_args__ = (
        UniqueConstraint("version_id", "regime", "source"),
        Index("ix_options_strategy_regimes_regime", "regime"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("options_strategy_versions.id", ondelete="CASCADE"))
    regime: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(16))  # backtest | shadow | paper
    trades: Mapped[int] = mapped_column(Integer, default=0)
    expectancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


# ----------------------------------------------------------------------------- candidates, theses, positions
class OptionsTradeCandidateRow(Base):
    __tablename__ = "options_trade_candidates"
    __table_args__ = (
        Index("ix_options_trade_candidates_underlying_created", "underlying", "created_at"),
        Index("ix_options_trade_candidates_status", "status"),
        Index("ix_options_trade_candidates_version", "version_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cycle_key: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    underlying: Mapped[str] = mapped_column(String(U))
    version_id: Mapped[int | None] = mapped_column(ForeignKey("options_strategy_versions.id"), nullable=True)
    family: Mapped[str] = mapped_column(String(32))
    structure_key: Mapped[str] = mapped_column(String(200))
    structure: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dte: Mapped[int | None] = mapped_column(Integer, nullable=True)
    iv_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String(12))  # shadow | paper
    # proposed | rejected | no_trade | shadow_opened | submitted | …
    status: Mapped[str] = mapped_column(String(24))
    gate: Mapped[str | None] = mapped_column(String(48), nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_quality: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    audit: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # the market snapshot, versions, checks


class OptionsTradeThesisRow(Base):
    __tablename__ = "options_trade_theses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("options_trade_candidates.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    underlying: Mapped[str] = mapped_column(String(U))
    direction: Mapped[str] = mapped_column(String(16))
    market_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    iv_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    thesis: Mapped[str] = mapped_column(Text)
    body: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # the full structured thesis
    debate: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    explanation: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


class OptionsPositionRow(Base):
    __tablename__ = "options_positions"
    __table_args__ = (
        Index("ix_options_positions_underlying_status", "underlying", "status"),
        Index("ix_options_positions_first_expiration", "first_expiration"),
        Index("ix_options_positions_version", "version_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("options_trade_candidates.id"), nullable=True)
    thesis_id: Mapped[int | None] = mapped_column(ForeignKey("options_trade_theses.id"), nullable=True)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("options_strategy_versions.id"), nullable=True)
    underlying: Mapped[str] = mapped_column(String(U))
    family: Mapped[str] = mapped_column(String(32))
    direction: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(12))  # shadow | paper
    structure: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    quantity: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))  # pending | open | closing | closed
    expiry_state: Mapped[str] = mapped_column(String(20), default="OPEN")
    first_expiration: Mapped[date | None] = mapped_column(Date, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(UTCDateTime())
    entry_value: Mapped[float] = mapped_column(Float)  # dollars paid (negative: credit received)
    entry_mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_underlying: Mapped[float] = mapped_column(Float)
    entry_iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_greeks: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    max_loss: Mapped[float] = mapped_column(Float)
    max_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    marks: Mapped[list[Any]] = mapped_column(JSON, default=list)  # daily marks for attribution
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    exit_client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    closed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    exit_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    attribution: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    critique: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsPositionEventRow(Base):
    __tablename__ = "options_position_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int] = mapped_column(
        ForeignKey("options_positions.id", ondelete="CASCADE"), index=True
    )
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    kind: Mapped[str] = mapped_column(String(24))
    message: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsExecutionLedgerRow(Base):
    __tablename__ = "options_execution_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_positions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("options_trade_candidates.id"), nullable=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    action: Mapped[str] = mapped_column(String(8))  # open | close
    legs: Mapped[list[Any]] = mapped_column(JSON, default=list)
    decision_at: Mapped[datetime] = mapped_column(UTCDateTime())
    quote_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    decision_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage_dollars: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24))


class OptionsAssignmentEventRow(Base):
    __tablename__ = "options_assignment_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_positions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    symbol: Mapped[str] = mapped_column(String(SYM))
    contracts: Mapped[int] = mapped_column(Integer)
    share_delivery: Mapped[int] = mapped_column(Integer)
    cash_flow: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsExerciseEventRow(Base):
    __tablename__ = "options_exercise_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_positions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    symbol: Mapped[str] = mapped_column(String(SYM))
    contracts: Mapped[int] = mapped_column(Integer)
    share_delivery: Mapped[int] = mapped_column(Integer)
    cash_flow: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


# ----------------------------------------------------------------------------- learning
class OptionsCounterfactualRow(Base):
    __tablename__ = "options_counterfactuals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_positions.id", ondelete="CASCADE"), nullable=True, index=True
    )
    candidate_id: Mapped[int | None] = mapped_column(ForeignKey("options_trade_candidates.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    alternative: Mapped[str] = mapped_column(String(48))
    structure: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_on_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    chosen_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    better_than_chosen: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    data_source: Mapped[str] = mapped_column(String(16))  # recorded | model


class OptionsMissedOpportunityRow(Base):
    __tablename__ = "options_missed_opportunities"
    __table_args__ = (Index("ix_options_missed_classification", "classification"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("options_trade_candidates.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    underlying: Mapped[str] = mapped_column(String(U))
    family: Mapped[str] = mapped_column(String(32))
    reject_reason: Mapped[str] = mapped_column(Text)
    gate: Mapped[str | None] = mapped_column(String(48), nullable=True)
    strategy_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade_after: Mapped[date] = mapped_column(Date)
    outcome_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    underlying_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification: Mapped[str | None] = mapped_column(String(32), nullable=True)
    graded_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsLearningEventRow(Base):
    __tablename__ = "options_learning_events"
    __table_args__ = (Index("ix_options_learning_events_kind_at", "kind", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    # trade_graded | prediction | calibration | weight_update | …
    kind: Mapped[str] = mapped_column(String(32))
    # backtest | shadow | paper — never mixed without a label
    evidence: Mapped[str] = mapped_column(String(16))
    subject: Mapped[str] = mapped_column(String(64))
    dims: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    predicted: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsLessonRow(Base):
    __tablename__ = "options_lessons"
    __table_args__ = (Index("ix_options_lessons_memory_status", "memory", "status"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # StrategyMemory | MarketRegimeMemory | … | CounterfactualMemory
    memory: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(32))
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    observation: Mapped[str] = mapped_column(Text)
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    date_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    date_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    applicability: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    expires_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    relevance: Mapped[float] = mapped_column(Float, default=1.0)
    # candidate | validated | refuted | stale
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    source: Mapped[str] = mapped_column(String(16))  # trade | backtest | meta | counterfactual | missed


class OptionsFeatureObservationRow(Base):
    __tablename__ = "options_feature_observations"
    __table_args__ = (Index("ix_options_feature_observations_underlying_day", "underlying", "day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    day: Mapped[date] = mapped_column(Date)
    underlying: Mapped[str] = mapped_column(String(U))
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    outcome: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(16))


class OptionsFeatureImportanceRow(Base):
    __tablename__ = "options_feature_importance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime())
    feature: Mapped[str] = mapped_column(String(96))
    interaction_with: Mapped[str | None] = mapped_column(String(96), nullable=True)
    target: Mapped[str] = mapped_column(String(32))
    importance: Mapped[float | None] = mapped_column(Float, nullable=True)
    oos_importance: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsStrategyWeightRow(Base):
    __tablename__ = "options_strategy_weights"
    __table_args__ = (
        UniqueConstraint("strategy_key", "regime", "structure", "underlying_class", "vol_state"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_key: Mapped[str] = mapped_column(String(64))
    regime: Mapped[str] = mapped_column(String(32))
    structure: Mapped[str] = mapped_column(String(32))
    underlying_class: Mapped[str] = mapped_column(String(32))
    vol_state: Mapped[str] = mapped_column(String(32))
    weight: Mapped[float] = mapped_column(Float)
    mean: Mapped[float] = mapped_column(Float)
    sd: Mapped[float] = mapped_column(Float)
    n: Mapped[int] = mapped_column(Integer)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())


class OptionsStrategyDecayRow(Base):
    __tablename__ = "options_strategy_decay"
    __table_args__ = (Index("ix_options_strategy_decay_version_at", "version_id", "at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("options_strategy_versions.id", ondelete="CASCADE"))
    at: Mapped[datetime] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(12))  # HEALTHY | WATCH | DEGRADING | BROKEN | RETIRED
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    reasons: Mapped[list[Any]] = mapped_column(JSON, default=list)


class OptionsHypothesisRow(Base):
    __tablename__ = "options_hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    # failure | lesson | research | meta | decay | interaction
    source: Mapped[str] = mapped_column(String(24))
    statement: Mapped[str] = mapped_column(Text)
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_strategy_versions.id"), nullable=True
    )
    rationale: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # open | testing | supported | refuted | inconclusive
    status: Mapped[str] = mapped_column(String(16), default="open")


class OptionsExperimentRow(Base):
    __tablename__ = "options_experiments"
    __table_args__ = (Index("ix_options_experiments_status_priority", "status", "priority"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hypothesis_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_hypotheses.id"), nullable=True, index=True
    )
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_strategy_versions.id"), nullable=True
    )
    child_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("options_strategy_versions.id"), nullable=True
    )
    feature_changes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    parameter_changes: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    dataset: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    train_period: Mapped[str | None] = mapped_column(String(32), nullable=True)
    test_period: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # QUEUED | RUNNING | PASSED | FAILED | INCONCLUSIVE | PROMOTED | REJECTED
    status: Mapped[str] = mapped_column(String(16))
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    decision: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class OptionsGenerationRunRow(Base):
    __tablename__ = "options_generation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generation: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created: Mapped[int] = mapped_column(Integer, default=0)
    evaluated: Mapped[int] = mapped_column(Integer, default=0)
    promoted: Mapped[int] = mapped_column(Integer, default=0)
    budget: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class OptionsKnowledgeEdgeRow(Base):
    """The knowledge graph: STRATEGY WORKS_IN REGIME, STRATEGY DERIVED_FROM SOURCE, TRADE PRODUCED LESSON, …"""

    __tablename__ = "options_knowledge_edges"
    __table_args__ = (
        UniqueConstraint("src_type", "src_id", "relation", "dst_type", "dst_id"),
        Index("ix_options_knowledge_edges_dst", "dst_type", "dst_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    src_type: Mapped[str] = mapped_column(String(16))
    src_id: Mapped[str] = mapped_column(String(64))
    relation: Mapped[str] = mapped_column(String(24))
    dst_type: Mapped[str] = mapped_column(String(16))
    dst_id: Mapped[str] = mapped_column(String(64))
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
