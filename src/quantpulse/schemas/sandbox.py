"""Paper-trading sandbox: accounts, orders, agent steps and walk-forward training."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, AwareDatetime, Field, field_validator, model_validator

from quantpulse.domain import screener
from quantpulse.schemas.common import DataStatus, StrictModel, Symbol

PAPER_DISCLAIMER = (
    "Paper trading only: simulated cash, simulated fills (quote ± slippage), no orders are ever sent to a "
    "broker. The agent's learned factor weights describe what worked recently and are not a forecast."
)
AccountMode = Literal["agent", "manual"]
Side = Literal["buy", "sell"]
MAX_WEIGHT_FLOOR = 1.0 / len(screener.WEIGHTS)


class SandboxStrategy(StrictModel):
    """How the agent picks stocks, sizes positions, learns, and how fills are simulated."""

    signal: Literal["factors", "model"] = Field(
        default="factors",
        description="factors = the self-weighting factor rule (learns from its own results); model = the "
        "walk-forward stock model's rankings (retrained monthly on history).",
    )
    universe: list[Symbol] | None = Field(
        default=None, min_length=3, max_length=60, description="Defaults to QP_PICKS_UNIVERSE."
    )
    top_k: int = Field(default=5, ge=1, le=20, description="How many names the agent holds at most.")
    max_position: float = Field(default=0.25, gt=0, le=1, description="Cap per name, as a share of equity.")
    cash_buffer: float = Field(default=0.02, ge=0, le=0.5, description="Share of equity always kept in cash.")
    learning_rate: float = Field(
        default=0.5, ge=0, le=5, description="η in w ← w·exp(η·IC). 0 disables learning."
    )
    prior_shrink: float = Field(
        default=0.05, ge=0, le=1, description="Pull towards the default weights per update."
    )
    weight_floor: float = Field(default=0.02, ge=0, description="Minimum weight any factor keeps.")
    min_trade_value: float = Field(
        default=50.0, ge=0, le=1_000_000, description="Skip smaller rebalancing trades."
    )
    slippage_bps: float = Field(
        default=5.0, ge=0, le=200, description="Fill = quote ± this many basis points."
    )
    commission_per_trade: float = Field(default=0.0, ge=0, le=100)
    commission_bps: float = Field(default=0.0, ge=0, le=100)

    @field_validator("universe")
    @classmethod
    def _dedupe(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        unique = list(dict.fromkeys(value))
        if len(unique) < 3:
            raise ValueError("universe needs at least 3 distinct symbols to rank")
        return unique

    @field_validator("weight_floor")
    @classmethod
    def _floor(cls, value: float) -> float:
        if value >= MAX_WEIGHT_FLOOR:
            raise ValueError(f"weight_floor must be below {MAX_WEIGHT_FLOOR:.4f} (1 / number of factors)")
        return value


def _clean_name(value: str) -> str:
    value = " ".join(value.split())
    if not value:
        raise ValueError("name must not be blank")
    return value


AccountName = Annotated[str, Field(min_length=1, max_length=80), AfterValidator(_clean_name)]


class AccountCreate(StrictModel):
    name: AccountName
    mode: AccountMode = Field(
        default="agent", description="'agent' trades itself; 'manual' only takes orders."
    )
    starting_cash: float = Field(default=100_000.0, gt=0, le=1_000_000_000)
    auto_trade: bool = Field(default=True, description="Let the scheduler run the agent every trading day.")
    allow_synthetic: bool = Field(
        default=False,
        description="Trade on synthetic prices when live data is down. Off by default so the paper record only "
        "ever reflects real market prices.",
    )
    strategy: SandboxStrategy = Field(default_factory=SandboxStrategy)


class AccountUpdate(StrictModel):
    name: AccountName | None = None
    mode: AccountMode | None = None
    auto_trade: bool | None = None
    allow_synthetic: bool | None = None
    strategy: SandboxStrategy | None = Field(default=None, description="Replaces the whole strategy.")


class AccountOut(StrictModel):
    id: int
    name: str
    mode: AccountMode
    starting_cash: float
    cash: float
    auto_trade: bool
    allow_synthetic: bool
    strategy: SandboxStrategy
    universe: list[str] = Field(description="The universe actually screened (strategy or default).")
    factor_weights: dict[str, float] = Field(description="The agent's current (learned) factor weights.")
    prior_weights: dict[str, float]
    ic_ema: dict[str, float] = Field(
        description="Exponential average of each factor's information coefficient."
    )
    periods_learned: int
    last_decision_on: date | None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class PositionOut(StrictModel):
    symbol: str
    quantity: float
    avg_cost: float
    price: float
    market_value: float
    weight: float = Field(description="Share of account equity.")
    unrealized_pnl: float
    unrealized_pnl_pct: float
    data_status: DataStatus


class PerformanceOut(StrictModel):
    equity: float
    cash: float
    invested: float
    total_return: float = Field(description="Equity / starting cash − 1.")
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    trades: int
    benchmark: str
    benchmark_return: float | None = Field(description="Benchmark return since the account's first snapshot.")
    max_drawdown: float | None = Field(description="Worst peak-to-trough fall of recorded equity (negative).")
    snapshots: int


class AccountSummary(StrictModel):
    account: AccountOut
    positions: list[PositionOut]
    performance: PerformanceOut
    data_status: DataStatus
    disclaimer: str = PAPER_DISCLAIMER


class OrderIn(StrictModel):
    """A manual market order. Give exactly one of ``quantity`` (shares) or ``notional`` (dollars)."""

    symbol: Symbol
    side: Side
    quantity: float | None = Field(default=None, gt=0, le=1e9)
    notional: float | None = Field(default=None, gt=0, le=1e10)
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _exactly_one(self) -> OrderIn:
        if (self.quantity is None) == (self.notional is None):
            raise ValueError("give exactly one of quantity or notional")
        return self


class TradeOut(StrictModel):
    id: int
    executed_at: AwareDatetime
    symbol: str
    side: Side
    quantity: float
    price: float
    reference_price: float
    notional: float
    commission: float
    realized_pnl: float
    data_status: DataStatus
    source: Literal["agent", "manual"]
    note: str | None


class EquityPoint(StrictModel):
    recorded_at: AwareDatetime
    equity: float
    cash: float
    benchmark_price: float | None
    data_status: DataStatus


class JournalEntry(StrictModel):
    id: int
    created_at: AwareDatetime
    kind: str
    summary: str
    details: dict[str, Any]


class LessonOut(StrictModel):
    factor: str
    label: str
    ic: float = Field(description="Spearman rank correlation of factor scores with realised returns.")
    observations: int
    weight_before: float
    weight_after: float


class Candidate(StrictModel):
    symbol: str
    composite: float
    rating: int = Field(ge=1, le=10)
    target_weight: float


class StepResult(StrictModel):
    account_id: int
    executed: bool
    skipped_reason: str | None = None
    trading_day: date
    as_of: AwareDatetime
    trades: list[TradeOut] = Field(default_factory=list)
    lessons: list[LessonOut] = Field(default_factory=list)
    weights_before: dict[str, float] = Field(default_factory=dict)
    weights_after: dict[str, float] = Field(default_factory=dict)
    targets: dict[str, float] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict, description="Symbols left out and why.")
    equity: float | None = None
    cash: float | None = None
    data_status: DataStatus


class TrainRequest(StrictModel):
    lookback_days: int = Field(
        default=1095,
        ge=450,
        le=3650,
        description="Calendar days of history to replay (the first ~253 trading "
        "days only warm up the 12-month factors).",
    )
    rebalance_every: int = Field(default=5, ge=1, le=21, description="Trading days between decisions.")
    apply: bool = Field(default=True, description="Adopt the learned weights for live paper trading.")


class BacktestMetrics(StrictModel):
    total_return: float
    annual_return: float | None
    annual_volatility: float | None
    sharpe: float | None
    max_drawdown: float | None


class CurvePoint(StrictModel):
    date: date
    strategy: float
    benchmark: float


class WeightSnapshot(StrictModel):
    date: date
    weights: dict[str, float]


class TrainReport(StrictModel):
    account_id: int
    start: date
    end: date
    trading_days: int
    decisions: int
    rebalance_every: int
    symbols: list[str]
    skipped: dict[str, str] = Field(default_factory=dict)
    benchmark: str
    risk_free_rate: float
    strategy: BacktestMetrics
    benchmark_metrics: BacktestMetrics
    trades: int
    fees_paid: float
    turnover: float = Field(description="Traded notional / average equity over the replay.")
    equity_curve: list[CurvePoint]
    weights_history: list[WeightSnapshot]
    mean_ic: dict[str, float] = Field(
        description="Average information coefficient per factor over the replay."
    )
    prior_weights: dict[str, float]
    learned_weights: dict[str, float]
    applied: bool
    data_status: DataStatus
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = PAPER_DISCLAIMER
