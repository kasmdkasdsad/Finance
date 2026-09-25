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
TEST_ORDER_PHRASE = "SUBMIT ONE PAPER TEST ORDER"
TradingMode = Literal["dry_run", "paper"]
# Where a proposed trade got to. Only "submitted" and later mean Alpaca has the order.
TradeStage = Literal[
    "risk_rejected",  # failed a risk check: never sent
    "risk_approved",  # passed every risk check but was not sent (dry run, kill switch, not paper mode)
    "submitting",  # recorded, request in flight (or interrupted: reconciliation settles it)
    "unknown",  # sent, no answer: looked up by client order id, never resent
    "failed",  # never reached Alpaca (invalid request, or Alpaca never received it)
    "rejected",  # Alpaca refused it
    "submitted",  # Alpaca has it (pending_new)
    "accepted",  # Alpaca accepted it and it is working (new / accepted)
    "partially_filled",
    "filled",
    "canceled",
    "expired",
]


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
    orders_submitted: int = Field(description="Orders Alpaca acknowledged (they have an Alpaca order id)")
    trades_proposed: int
    trades_approved: int = 0
    orders_filled: int = 0
    orders_failed: int = Field(0, description="Rejected by Alpaca, failed, or of unknown outcome")


class SettingSourceOut(StrictModel):
    variable: str
    source: Literal["environment", "env_file", "default"] = Field(
        description="'environment' (process variables override .env), 'env_file' or 'default'"
    )
    value: str = Field(description="Secrets are reported only as 'set' / 'not set'")


class TradingConfigOut(StrictModel):
    env_file: str | None = Field(
        description="The .env file the running API read (none: defaults + environment)"
    )
    env_file_found: bool
    loaded_at: AwareDatetime | None = Field(description="When settings were read; .env edits need a restart")
    restart_required: bool
    drift: list[str] = Field(description="Switches whose .env value changed since the API started")
    sources: list[SettingSourceOut]
    database: str = Field(description="Database file in use (relative paths resolve from the API's folder)")


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
    submit_blockers: list[str] = Field(
        default_factory=list, description="Why orders are not sent (empty when can_submit)"
    )
    kill_switch: KillSwitchOut
    scheduler_enabled: bool
    scheduler_armed: bool = Field(
        True,
        description="Scheduled cycles may send orders (false until paper execution is used once by hand)",
    )
    scheduled_mode: TradingMode = Field("dry_run", description="What the next scheduled cycle would do")
    interval_minutes: int
    first_cycle_time: str
    next_cycle_at: AwareDatetime | None
    market: MarketClockOut | None
    last_cycle: CycleSummary | None
    last_reconciled_at: AwareDatetime | None
    api_token_set: bool
    warnings: list[str]
    config: TradingConfigOut | None = None


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
    stage: TradeStage | None = Field(None, description="Where the trade got to (see TradeStage)")
    alpaca_order_id: str | None = Field(None, description="Set once Alpaca acknowledged the order")
    filled_qty: float | None = None
    filled_avg_price: float | None = None
    submitted_at: AwareDatetime | None = None
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


class DiagnosticCheckOut(StrictModel):
    name: str
    ok: bool | None = Field(description="None: not run (an earlier step failed or it does not apply)")
    detail: str
    ms: float | None = None


class CredentialsOut(StrictModel):
    key_id_set: bool
    secret_set: bool
    key_id_source: str
    secret_source: str
    key_id_looks_like_paper: bool | None = Field(
        description="Alpaca paper key ids start with 'PK' (live ones with 'AK'); the key itself is never shown"
    )


class QuoteDiagnosticOut(StrictModel):
    symbol: str
    provider: str
    feed: str | None
    price: float
    trade_age_seconds: float
    bid: float | None
    ask: float | None
    quote_age_seconds: float | None
    venue_spread_bps: float | None = Field(description="Spread on the primary feed (IEX: one exchange)")
    consolidated_feed: str | None
    consolidated_bid: float | None
    consolidated_ask: float | None
    consolidated_age_seconds: float | None
    consolidated_spread_bps: float | None
    spread_bps: float | None = Field(description="The validated spread the risk engine uses")
    spread_source: str
    max_spread_bps: float
    spread_ok: bool
    previous_close: float | None
    history_close: float | None
    problems: list[str]
    entry_blocks: list[str]


class DiagnosticOrderStatusOut(StrictModel):
    allowed: bool
    blockers: list[str]
    phrase: str = TEST_ORDER_PHRASE
    max_rest_notional: float
    max_fill_notional: float


class TradingDiagnosticsOut(StrictModel):
    at: AwareDatetime
    paper: bool = True
    endpoint: str
    endpoint_verified: bool = Field(
        description="The SDK client was built with paper=True and points at the paper API"
    )
    sdk_version: str | None
    credentials: CredentialsOut
    mode: TradingMode
    can_submit: bool
    submit_blockers: list[str]
    config: TradingConfigOut
    checks: list[DiagnosticCheckOut]
    account: BrokerAccountOut | None
    market: MarketClockOut | None
    positions: list[BrokerPositionOut]
    open_orders: list[BrokerOrderOut]
    quotes: list[QuoteDiagnosticOut]
    test_order: DiagnosticOrderStatusOut
    orders_sent: bool = Field(False, description="Always false: diagnostics never place or cancel orders")


class DiagnosticOrderIn(StrictModel):
    confirm: str = Field(description=f"Type exactly {TEST_ORDER_PHRASE!r}")
    symbol: str = Field("SPY", min_length=1, max_length=10, pattern=r"^[A-Za-z][A-Za-z.\-]{0,9}$")
    mode: Literal["rest_and_cancel", "fill"] = Field(
        "rest_and_cancel",
        description=(
            "'rest_and_cancel': BUY 1 share with a limit ~10% below the bid (cannot fill), then cancel it. "
            "'fill': a market BUY for `notional` dollars (≤ $25) that fills and leaves a tiny position."
        ),
    )
    notional: float = Field(10.0, gt=0, le=25, description="Dollars, 'fill' mode only")


class DiagnosticOrderOut(StrictModel):
    sent: bool = Field(description="True only if the order reached Alpaca (it has an Alpaca order id)")
    mode: Literal["rest_and_cancel", "fill"]
    symbol: str
    side: Literal["buy"] = "buy"
    order_type: str
    qty: float | None
    notional: float | None
    limit_price: float | None
    client_order_id: str | None
    alpaca_order_id: str | None
    status_after_submit: str | None
    final_status: str | None
    statuses_seen: list[str]
    canceled: bool
    filled_qty: float
    filled_avg_price: float | None
    checks: list[RiskCheckOut]
    message: str
    error: str | None = None
