"""Alpaca paper trading: account, positions, orders, strategy cycles, risk and performance.

No schema here ever carries an API key or secret; account numbers are masked.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import DataStatus, StrictModel

PAPER_BANNER = "ALPACA PAPER TRADING — SIMULATED MONEY ONLY"
DRY_RUN_BANNER = "DRY RUN — NO ORDERS WILL BE SUBMITTED"
CLOSE_ALL_PHRASE = "CLOSE ALL"
TradingMode = Literal["dry_run", "paper"]


class KillSwitchOut(StrictModel):
    active: bool
    source: Literal["env", "runtime"] | None = Field(
        description="'env' (QP_TRADING_KILL_SWITCH — only the setting can release it) or 'runtime' (dashboard/API)"
    )
    reason: str | None = None
    changed_at: AwareDatetime | None = None


class MarketClockOut(StrictModel):
    is_open: bool
    next_open: AwareDatetime | None = None
    next_close: AwareDatetime | None = None
    source: Literal["alpaca", "calendar"]


class CycleSummary(StrictModel):
    id: int
    cycle_key: str
    trigger: str
    mode: TradingMode
    status: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    skip_reason: str | None
    orders_submitted: int
    trades_proposed: int


class TradingStatus(StrictModel):
    banner: str = PAPER_BANNER
    paper: bool = Field(True, description="Always true: QuantPulse only trades Alpaca paper accounts")
    endpoint: str = Field(description="The Alpaca API QuantPulse talks to (always the paper endpoint)")
    broker_configured: bool
    trading_enabled: bool = Field(description="QP_ALPACA_TRADING_ENABLED")
    dry_run: bool = Field(description="QP_TRADING_DRY_RUN")
    mode: TradingMode
    mode_banner: str
    can_submit: bool = Field(description="Whether a cycle would send orders right now")
    kill_switch: KillSwitchOut
    scheduler_enabled: bool
    interval_minutes: int
    first_cycle_time: str
    next_cycle_at: AwareDatetime | None
    market: MarketClockOut | None
    last_cycle: CycleSummary | None
    last_reconciled_at: AwareDatetime | None
    api_token_set: bool
    warnings: list[str]


class BrokerAccountOut(StrictModel):
    paper: bool = True
    account_number: str = Field(description="Masked: last four characters only")
    status: str
    currency: str
    equity: float
    last_equity: float
    cash: float
    buying_power: float
    long_market_value: float
    portfolio_value: float
    day_pl: float
    day_pl_pct: float
    total_pl: float | None = Field(description="Equity minus the baseline QuantPulse first recorded")
    total_pl_pct: float | None
    baseline_equity: float | None
    baseline_at: AwareDatetime | None
    exposure_pct: float
    trading_blocked: bool
    pattern_day_trader: bool
    daytrade_count: int


class BrokerPositionOut(StrictModel):
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pl: float
    unrealized_plpc: float
    intraday_pl: float
    cost_basis: float
    target_weight: float | None = Field(description="From the latest strategy cycle")
    signal_score: float | None
    stop_loss_price: float


class BrokerOrderOut(StrictModel):
    client_order_id: str
    alpaca_order_id: str | None
    symbol: str
    side: str
    qty: float | None
    filled_qty: float
    order_type: str
    limit_price: float | None
    filled_avg_price: float | None
    notional: float | None
    status: str
    submitted_at: AwareDatetime | None
    filled_at: AwareDatetime | None
    strategy: str | None
    kind: str | None
    signal_score: float | None
    reason: str | None
    error: str | None
    source: Literal["alpaca", "quantpulse"] = Field(
        description="'alpaca': read from the paper account now; 'quantpulse': local record only"
    )


class RiskCheckOut(StrictModel):
    name: str
    passed: bool
    detail: str


class ProposedTradeOut(StrictModel):
    symbol: str
    side: str
    qty: float
    est_price: float
    notional: float
    kind: str
    reason: str
    current_weight: float
    target_weight: float
    score: float | None
    approved: bool
    risk: str = Field(description="'approved' or the failed checks")
    checks: list[RiskCheckOut]
    order_type: str | None = None
    limit_price: float | None = None
    client_order_id: str | None = None
    status: str = Field(description="dry_run, not_submitted, risk_rejected, or the order's Alpaca status")
    error: str | None = None


class SignalOut(StrictModel):
    symbol: str
    rank: int
    score: float
    components: dict[str, float]
    price: float
    risk_vol: float | None
    adv_dollar: float | None
    trend_ok: bool
    trend_broken: bool
    model_z: float | None
    implied_vol: float | None
    earnings_date: date | None
    entry_blocks: list[str]
    target_weight: float
    held: bool


class TargetOut(StrictModel):
    symbol: str
    weight: float
    score: float
    conviction: float
    risk_vol: float
    capped_by: str | None
    incumbent: bool


class TradingRegimeOut(StrictModel):
    label: str
    description: str
    trend_score: float
    stressed: bool
    exposure_share: float
    entry_penalty: float
    beta_tilt: float
    metrics: dict[str, float | None]
    reasons: list[str]


class CycleOut(StrictModel):
    id: int
    cycle_key: str
    trigger: str
    mode: TradingMode
    status: str
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    skip_reason: str | None
    error: str | None
    equity: float | None
    last_equity: float | None
    cash: float | None
    buying_power: float | None
    long_market_value: float | None
    data_status: DataStatus | None
    regime: TradingRegimeOut | None
    gross_target: float | None
    positions: list[dict[str, Any]]
    signals: list[SignalOut]
    targets: list[TargetOut]
    trades: list[ProposedTradeOut]
    exits: dict[str, str]
    skipped: dict[str, str]
    notes: list[str]


class TradingEventOut(StrictModel):
    id: int
    created_at: AwareDatetime
    cycle_id: int | None
    kind: str
    symbol: str | None
    client_order_id: str | None
    message: str
    details: dict[str, Any]


class RiskSnapshotOut(StrictModel):
    equity: float
    day_pl: float
    day_pl_pct: float
    daily_loss_limit_pct: float
    daily_loss_limit_hit: bool
    daily_loss_action: str
    exposure: float
    exposure_pct: float
    max_exposure_pct: float
    cash: float
    cash_pct: float
    cash_buffer_pct: float
    positions: int
    max_positions: int
    largest_position: str | None
    largest_position_pct: float | None
    max_position_pct: float
    max_order_notional: float
    min_order_notional: float
    position_loss_limit_pct: float
    positions_at_stop: dict[str, float]
    kill_switch: KillSwitchOut
    dry_run: bool
    trading_enabled: bool
    can_submit: bool
    market_open: bool | None


class DailyPL(StrictModel):
    date: date
    equity: float
    pl: float | None
    ret: float | None = Field(alias="return")


class MonthlyPL(StrictModel):
    month: str
    pl: float
    ret: float | None = Field(alias="return")


class TradingPerformanceOut(StrictModel):
    days: int
    first_day: date | None
    last_day: date | None
    start_equity: float | None
    end_equity: float | None
    total_return: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    best_day: float | None
    worst_day: float | None
    round_trips: int
    win_rate: float | None
    avg_winner: float | None
    avg_loser: float | None
    profit_factor: float | None
    realized_pl: float
    turnover: float | None
    avg_exposure: float | None
    daily: list[DailyPL]
    monthly: list[MonthlyPL]
    by_symbol: dict[str, float]
    by_exit: dict[str, float]
    notes: list[str]


class ReconcileOut(StrictModel):
    at: AwareDatetime
    equity: float
    positions: int
    open_orders: int
    orders_checked: int
    orders_updated: int
    orders_added: int
    unknown_resolved: int
    changes: list[str]


class KillSwitchIn(StrictModel):
    active: bool
    reason: str | None = Field(default=None, max_length=300)
    cancel_open_orders: bool = Field(
        default=True, description="Also cancel QuantPulse's working orders when activating"
    )


class CancelAllIn(StrictModel):
    confirm: bool = Field(description="Must be true: cancels every open order on the Alpaca paper account")


class CloseAllIn(StrictModel):
    confirm: str = Field(description=f"Type exactly {CLOSE_ALL_PHRASE!r} to sell every position")


class ActionOut(StrictModel):
    message: str
    mode: TradingMode
    submitted: int = 0
    canceled: int = 0
    trades: list[ProposedTradeOut] = Field(default_factory=list)
