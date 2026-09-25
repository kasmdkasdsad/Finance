"""Automated trading of the Alpaca **paper** account (simulated money only).

One cycle (every ``QP_TRADING_REBALANCE_INTERVAL_MINUTES`` from ``QP_TRADING_TIME`` during the session, or
on demand):

1. **Reconcile** QuantPulse's order records with Alpaca and read the account, positions, open orders and
   Alpaca's market clock (Alpaca is authoritative for all of them).
2. **Load data**: the liquid universe, daily bars, live snapshots, the stock model, VIX, implied volatility
   and earnings dates (:mod:`quantpulse.services.trading_data`).
3. **Regime** (:mod:`quantpulse.domain.trading_regime`) → share of exposure to deploy, extra entry bar.
4. **Signals** (:mod:`quantpulse.domain.trading_signals`) → one opportunity score per symbol.
5. **Portfolio** (:mod:`quantpulse.domain.trading_portfolio`) → targets and proposed trades (exits first).
6. **Risk** (:mod:`quantpulse.services.trading_risk`) → every proposed order is approved or rejected.
7. **Execute** (only when ``QP_ALPACA_TRADING_ENABLED=true`` and ``QP_TRADING_DRY_RUN=false`` and the kill
   switch is off): sells go first; the cycle waits briefly for their fills, re-reads the account and
   re-checks the buys against the new cash (:mod:`quantpulse.services.order_manager`).
8. **Reconcile fills and record** the whole cycle: regime, equity, positions, signals, targets, trades,
   risk decisions, orders — plus an audit event for each step.

In a dry run steps 1-6 and 8 run exactly the same; nothing is sent to Alpaca.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy.exc import IntegrityError

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.jobs import Job, JobPending, JobRegistry
from quantpulse.core.market_calendar import NEW_YORK, is_market_open, is_trading_day, next_open, regular_close
from quantpulse.db import repositories as repo
from quantpulse.db.models import TradingCycleRow, TradingEventRow
from quantpulse.db.session import Database
from quantpulse.domain import trading_performance as perf
from quantpulse.domain import trading_regime as regime_mod
from quantpulse.domain import trading_signals as ts
from quantpulse.domain.trading_portfolio import (
    Candidate,
    Holding,
    PortfolioPlan,
    PositionMemory,
    ProposedTrade,
    StrategyConfig,
    build_plan,
)
from quantpulse.providers.alpaca_trading import (
    AlpacaPaperBroker,
    BrokerAccount,
    BrokerError,
    BrokerNotConfigured,
    BrokerOrder,
    BrokerPosition,
    MarketClock,
)
from quantpulse.schemas.common import DataStatus
from quantpulse.schemas.trading import (
    CLOSE_ALL_PHRASE,
    DRY_RUN_BANNER,
    ActionOut,
    BrokerAccountOut,
    BrokerOrderOut,
    BrokerPositionOut,
    CycleOut,
    CycleSummary,
    KillSwitchOut,
    MarketClockOut,
    ProposedTradeOut,
    ReconcileOut,
    RiskCheckOut,
    RiskSnapshotOut,
    SignalOut,
    TargetOut,
    TradingEventOut,
    TradingPerformanceOut,
    TradingRegimeOut,
    TradingStatus,
)
from quantpulse.services.order_manager import STRATEGY, OrderManager, Submission, client_order_id, is_ours
from quantpulse.services.trading_data import LiveQuote, TradingDataLoader, TradingInputs
from quantpulse.services.trading_risk import (
    OrderIntent,
    QuoteCheck,
    RiskBook,
    RiskDecision,
    RiskLimits,
    losing_positions,
)

logger = logging.getLogger(__name__)

KILL_KEY = "kill_switch"
MEMORY_KEY = "positions"
BASELINE_KEY = "baseline"
DAY_KEY = "daily_loss"
JOB_KEY = "trading-cycle"
RECONCILE_EVERY = timedelta(minutes=5)
SCHEDULER_BACKOFF = timedelta(minutes=10)
TOP_SIGNALS = 40
Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    return None


class TradingCycleRunning(DomainError):
    """A cycle is still running; the API answers 202 with its progress."""

    def __init__(self, job: Job) -> None:
        super().__init__(f"{job.description}: {job.progress:.0%} ({job.stage})")
        self.job = job


def cycle_slots(day: date, first: time, interval_minutes: int, stop_before_close: int) -> list[datetime]:
    """Scheduled cycle start times (New York) on ``day``: from ``first`` every ``interval_minutes`` until
    ``stop_before_close`` minutes before the (possibly early) close."""
    if not is_trading_day(day):
        return []
    opening = datetime.combine(day, time(9, 30), NEW_YORK)
    start = max(datetime.combine(day, first, NEW_YORK), opening)
    last = datetime.combine(day, regular_close(day), NEW_YORK) - timedelta(minutes=stop_before_close)
    out: list[datetime] = []
    t = start
    while t <= last:
        out.append(t)
        t += timedelta(minutes=interval_minutes)
    return out


def _fin(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round_price(p: float) -> float:
    return round(p, 2) if p >= 1 else round(p, 4)


class TradingService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        clock: Clock,
        broker: AlpacaPaperBroker,
        data: TradingDataLoader,
        jobs: JobRegistry,
    ) -> None:
        self._s = settings
        self._db = db
        self._clock = clock
        self.broker = broker
        self._data = data
        self._jobs = jobs
        self.orders = OrderManager(
            db, broker, clock, order_timeout=timedelta(minutes=settings.trading_order_timeout_minutes)
        )
        self.limits = RiskLimits.from_settings(settings)
        self.strategy = StrategyConfig.from_settings(settings)
        self._lock = asyncio.Lock()
        self._last_reconcile: datetime | None = None
        self._retry_at: datetime | None = None
        self._started = False

    # ------------------------------------------------------------------ state
    async def _state(self, key: str) -> dict[str, Any]:
        async with self._db.session() as s:
            return await repo.get_trading_state(s, key) or {}

    async def _put_state(self, key: str, value: Mapping[str, Any]) -> None:
        async with self._db.session() as s:
            await repo.put_trading_state(s, key, value, self._clock.now())

    async def _event(self, kind: str, message: str, **kw: Any) -> None:
        async with self._db.session() as s:
            await repo.add_trading_event(s, kind, message, self._clock.now(), **kw)

    async def kill_switch(self) -> KillSwitchOut:
        if self._s.trading_kill_switch:
            return KillSwitchOut(active=True, source="env", reason="QP_TRADING_KILL_SWITCH=true")
        st = await self._state(KILL_KEY)
        if st.get("active"):
            return KillSwitchOut(
                active=True,
                source="runtime",
                reason=st.get("reason"),
                changed_at=datetime.fromisoformat(st["changed_at"]) if st.get("changed_at") else None,
            )
        changed = st.get("changed_at")
        return KillSwitchOut(
            active=False,
            source=None,
            reason=None,
            changed_at=datetime.fromisoformat(changed) if changed else None,
        )

    def _require_broker(self) -> None:
        if not self.broker.configured():
            raise BrokerNotConfigured(
                "Alpaca paper trading is not configured: set QP_ALPACA_API_KEY_ID and QP_ALPACA_API_SECRET_KEY "
                "(paper keys) in .env and restart the API"
            )

    async def _mode(self, force_dry_run: bool = False) -> tuple[str, KillSwitchOut]:
        kill = await self.kill_switch()
        submit = (
            self._s.trading_can_submit and not kill.active and not force_dry_run and self.broker.configured()
        )
        return ("paper" if submit else "dry_run"), kill

    async def _baseline(self, account: BrokerAccount) -> dict[str, Any]:
        st = await self._state(BASELINE_KEY)
        if not st and account.equity > 0:
            st = {"equity": account.equity, "at": self._clock.now().isoformat()}
            await self._put_state(BASELINE_KEY, st)
            await self._event(
                "baseline_recorded",
                f"P/L baseline recorded: paper equity ${account.equity:,.2f}",
                details={"equity": account.equity},
            )
        return st

    async def _market(self) -> tuple[bool, MarketClockOut]:
        """Alpaca's clock (authoritative, knows holidays); the NYSE calendar if Alpaca cannot be reached."""
        now = self._clock.now()
        if self.broker.configured():
            try:
                c: MarketClock = await self.broker.clock()
                return c.is_open, MarketClockOut(
                    is_open=c.is_open, next_open=c.next_open, next_close=c.next_close, source="alpaca"
                )
            except BrokerError as exc:
                logger.warning("Alpaca clock unavailable, using the calendar: %s", exc)
        is_open = is_market_open(now)
        return is_open, MarketClockOut(is_open=is_open, next_open=next_open(now), source="calendar")

    # ------------------------------------------------------------------ read views
    async def status(self) -> TradingStatus:
        s = self._s
        mode, kill = await self._mode()
        market: MarketClockOut | None = None
        warnings: list[str] = []
        if self.broker.configured():
            _, market = await self._market()
        else:
            warnings.append("Alpaca paper keys are not set (QP_ALPACA_API_KEY_ID / QP_ALPACA_API_SECRET_KEY)")
        if s.alpaca_trading_enabled and not s.trading_dry_run and s.api_token is None:
            warnings.append(
                "Orders are enabled but QP_API_TOKEN is not set: order endpoints only accept requests from this "
                "machine. Set QP_API_TOKEN before exposing the API."
            )
        if not s.enable_live_data:
            warnings.append("QP_ENABLE_LIVE_DATA=false: no live quotes, so every order would be refused")
        async with self._db.session() as sess:
            recent = await repo.trading_cycles(sess, 1)
        last_row = recent[0] if recent else None
        return TradingStatus(
            endpoint=self.broker.base_url,
            broker_configured=self.broker.configured(),
            trading_enabled=s.alpaca_trading_enabled,
            dry_run=s.trading_dry_run,
            mode=mode,
            mode_banner=(
                "ALPACA PAPER EXECUTION ACTIVE — orders are sent to your Alpaca paper account"
                if mode == "paper"
                else DRY_RUN_BANNER
            ),
            can_submit=mode == "paper",
            kill_switch=kill,
            scheduler_enabled=s.trading_scheduler_enabled,
            interval_minutes=s.trading_rebalance_interval_minutes,
            first_cycle_time=s.trading_time,
            next_cycle_at=self._next_slot(),
            market=market,
            last_cycle=self._summary(last_row) if last_row is not None else None,
            last_reconciled_at=self._last_reconcile,
            api_token_set=s.api_token is not None,
            warnings=warnings,
        )

    def _slots(self, day: date) -> list[datetime]:
        s = self._s
        return cycle_slots(
            day,
            time.fromisoformat(s.trading_time),
            s.trading_rebalance_interval_minutes,
            s.trading_stop_minutes_before_close,
        )

    def _next_slot(self) -> datetime | None:
        local = self._clock.now().astimezone(NEW_YORK)
        day = local.date()
        for _ in range(10):
            for slot in self._slots(day):
                if slot > local:
                    return slot
            day += timedelta(days=1)
        return None

    def current_slot(self) -> datetime | None:
        """The scheduled slot whose window contains now (a missed earlier slot is never run late)."""
        local = self._clock.now().astimezone(NEW_YORK)
        slots = self._slots(local.date())
        step = timedelta(minutes=self._s.trading_rebalance_interval_minutes)
        for slot in reversed(slots):
            if slot <= local < slot + step:
                return slot
        return None

    @staticmethod
    def _summary(row: TradingCycleRow) -> CycleSummary:
        return CycleSummary(
            id=row.id,
            cycle_key=row.cycle_key,
            trigger=row.trigger,
            mode=row.mode,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            skip_reason=row.skip_reason,
            orders_submitted=sum(
                1
                for t in row.trades
                if t.get("client_order_id")
                and t.get("status") not in ("dry_run", "risk_rejected", "not_submitted")
            ),
            trades_proposed=len(row.trades),
        )

    async def account(self) -> BrokerAccountOut:
        self._require_broker()
        a = await self.broker.account()
        base = await self._baseline(a)
        b_eq = _fin(base.get("equity"))
        return BrokerAccountOut(
            account_number=a.account_number,
            status=a.status,
            currency=a.currency,
            equity=a.equity,
            last_equity=a.last_equity,
            cash=a.cash,
            buying_power=a.buying_power,
            long_market_value=a.long_market_value,
            portfolio_value=a.portfolio_value,
            day_pl=a.day_pl,
            day_pl_pct=a.day_pl_pct,
            total_pl=a.equity - b_eq if b_eq else None,
            total_pl_pct=a.equity / b_eq - 1 if b_eq else None,
            baseline_equity=b_eq,
            baseline_at=datetime.fromisoformat(base["at"]) if base.get("at") else None,
            exposure_pct=a.long_market_value / a.equity if a.equity > 0 else 0.0,
            trading_blocked=a.blocked,
            pattern_day_trader=a.pattern_day_trader,
            daytrade_count=a.daytrade_count,
        )

    async def _latest_cycle(self) -> TradingCycleRow | None:
        async with self._db.session() as s:
            rows = await repo.trading_cycles(s, 1, statuses=["completed"])
        return rows[0] if rows else None

    async def positions(self) -> list[BrokerPositionOut]:
        self._require_broker()
        a, positions = await asyncio.gather(self.broker.account(), self.broker.positions())
        last = await self._latest_cycle()
        targets = {t["symbol"]: t["weight"] for t in (last.targets if last else [])}
        scores = {x["symbol"]: x["score"] for x in (last.signals if last else [])}
        eq = a.equity if a.equity > 0 else 1.0
        return [
            BrokerPositionOut(
                symbol=p.symbol,
                qty=p.qty,
                avg_entry_price=p.avg_entry_price,
                current_price=p.current_price,
                market_value=p.market_value,
                weight=p.market_value / eq,
                unrealized_pl=p.unrealized_pl,
                unrealized_plpc=p.unrealized_plpc,
                intraday_pl=p.unrealized_intraday_pl,
                cost_basis=p.cost_basis,
                target_weight=targets.get(p.symbol, 0.0 if last else None),
                signal_score=scores.get(p.symbol),
                stop_loss_price=p.avg_entry_price * (1 - self._s.trading_max_position_loss_pct),
            )
            for p in positions
        ]

    async def order_list(self, status: str = "all", limit: int = 100) -> list[BrokerOrderOut]:
        """Alpaca's orders (authoritative), annotated with QuantPulse's reason, strategy and score; plus
        local-only records (never reached Alpaca, or refused)."""
        self._require_broker()
        remote = await self.broker.orders(status, limit=limit)  # type: ignore[arg-type]
        async with self._db.session() as s:
            local_rows = await repo.broker_orders(s, max(limit, 200))
        local = {r.client_order_id: r for r in local_rows}
        out: list[BrokerOrderOut] = []
        for o in remote:
            r = local.get(o.client_order_id)
            out.append(
                BrokerOrderOut(
                    client_order_id=o.client_order_id,
                    alpaca_order_id=o.id,
                    symbol=o.symbol,
                    side=o.side,
                    qty=o.qty,
                    filled_qty=o.filled_qty,
                    order_type=o.order_type,
                    limit_price=o.limit_price,
                    filled_avg_price=o.filled_avg_price,
                    notional=(o.qty or 0) * (o.filled_avg_price or o.limit_price or 0) or None,
                    status=o.status,
                    submitted_at=o.submitted_at,
                    filled_at=o.filled_at,
                    strategy=r.strategy if r else "external",
                    kind=r.kind if r else None,
                    signal_score=r.signal_score if r else None,
                    reason=r.reason if r else None,
                    error=r.error if r else None,
                    source="alpaca",
                )
            )
        seen = {o.client_order_id for o in remote}
        if status in ("all", "closed"):
            for r in local_rows:
                if r.client_order_id in seen or r.alpaca_order_id is not None:
                    continue
                out.append(
                    BrokerOrderOut(
                        client_order_id=r.client_order_id,
                        alpaca_order_id=None,
                        symbol=r.symbol,
                        side=r.side,
                        qty=r.quantity,
                        filled_qty=r.filled_quantity,
                        order_type=r.order_type,
                        limit_price=r.limit_price,
                        filled_avg_price=r.average_fill_price,
                        notional=r.notional,
                        status=r.status,
                        submitted_at=r.submitted_at,
                        filled_at=r.filled_at,
                        strategy=r.strategy,
                        kind=r.kind,
                        signal_score=r.signal_score,
                        reason=r.reason,
                        error=r.error,
                        source="quantpulse",
                    )
                )
        return out[:limit]

    async def risk(self) -> RiskSnapshotOut:
        self._require_broker()
        a, positions = await asyncio.gather(self.broker.account(), self.broker.positions())
        is_open, _ = await self._market()
        mode, kill = await self._mode()
        L = self.limits
        eq = a.equity if a.equity > 0 else 0.0
        largest = max(positions, key=lambda p: p.market_value, default=None)
        return RiskSnapshotOut(
            equity=a.equity,
            day_pl=a.day_pl,
            day_pl_pct=a.day_pl_pct,
            daily_loss_limit_pct=L.max_daily_loss_pct,
            daily_loss_limit_hit=a.last_equity > 0 and a.day_pl_pct <= -L.max_daily_loss_pct,
            daily_loss_action=self._s.trading_daily_loss_action,
            exposure=a.long_market_value,
            exposure_pct=a.long_market_value / eq if eq else 0.0,
            max_exposure_pct=L.max_total_exposure_pct,
            cash=a.cash,
            cash_pct=a.cash / eq if eq else 0.0,
            cash_buffer_pct=L.cash_buffer_pct,
            positions=len(positions),
            max_positions=L.max_positions,
            largest_position=largest.symbol if largest else None,
            largest_position_pct=largest.market_value / eq if largest and eq else None,
            max_position_pct=L.max_position_pct,
            max_order_notional=L.max_order_notional,
            min_order_notional=L.min_order_notional,
            position_loss_limit_pct=L.max_position_loss_pct,
            positions_at_stop=losing_positions({p.symbol: p for p in positions}, L.max_position_loss_pct),
            kill_switch=kill,
            dry_run=self._s.trading_dry_run,
            trading_enabled=self._s.alpaca_trading_enabled,
            can_submit=mode == "paper",
            market_open=is_open,
        )

    async def cycles(self, limit: int = 20) -> list[CycleSummary]:
        async with self._db.session() as s:
            rows = await repo.trading_cycles(s, limit)
        return [self._summary(r) for r in rows]

    async def cycle(self, cycle_id: int) -> CycleOut:
        async with self._db.session() as s:
            row = await repo.get_trading_cycle(s, cycle_id)
        return self._cycle_out(row)

    async def proposed(self) -> CycleOut | None:
        """The latest completed cycle: regime, top opportunities, targets and proposed trades."""
        row = await self._latest_cycle()
        return self._cycle_out(row) if row is not None else None

    async def events(self, limit: int = 200, kinds: Sequence[str] | None = None) -> list[TradingEventOut]:
        async with self._db.session() as s:
            rows = await repo.trading_events(s, limit, kinds=kinds)
        return [_event_out(r) for r in rows]

    async def performance(self) -> TradingPerformanceOut:
        async with self._db.session() as s:
            cycles = await repo.trading_cycles(s, 100_000, statuses=["completed"], oldest_first=True)
            filled = await repo.filled_broker_orders(s)
        points = [
            perf.EquityPoint(
                at=c.started_at,
                day=c.started_at.astimezone(NEW_YORK).date(),
                equity=c.equity,
                long_market_value=c.long_market_value,
            )
            for c in cycles
            if c.equity is not None and c.equity > 0
        ]
        fills = [
            perf.Fill(
                symbol=o.symbol,
                side=o.side,
                qty=o.filled_quantity,
                price=o.average_fill_price or 0.0,
                at=o.filled_at or o.updated_at,
                kind=o.kind,
            )
            for o in filled
            if o.average_fill_price
        ]
        p = perf.summarize(points, fills)
        return TradingPerformanceOut.model_validate(
            {
                "days": p.days,
                "first_day": p.first_day,
                "last_day": p.last_day,
                "start_equity": p.start_equity,
                "end_equity": p.end_equity,
                "total_return": p.total_return,
                "sharpe": p.sharpe,
                "sortino": p.sortino,
                "max_drawdown": p.max_drawdown,
                "best_day": p.best_day,
                "worst_day": p.worst_day,
                "round_trips": p.round_trips,
                "win_rate": p.win_rate,
                "avg_winner": p.avg_winner,
                "avg_loser": p.avg_loser,
                "profit_factor": p.profit_factor,
                "realized_pl": p.realized_pl,
                "turnover": p.turnover,
                "avg_exposure": p.avg_exposure,
                "daily": p.daily,
                "monthly": p.monthly,
                "by_symbol": p.by_symbol,
                "by_exit": p.by_exit,
                "notes": p.notes,
            }
        )

    # ------------------------------------------------------------------ controls
    async def set_kill_switch(
        self, active: bool, reason: str | None, cancel_open_orders: bool
    ) -> KillSwitchOut:
        now = self._clock.now()
        if not active and self._s.trading_kill_switch:
            raise DomainError(
                "the kill switch is set by QP_TRADING_KILL_SWITCH=true: change the setting to release it"
            )
        await self._put_state(KILL_KEY, {"active": active, "reason": reason, "changed_at": now.isoformat()})
        canceled = 0
        if active and cancel_open_orders and self.broker.configured() and self._s.alpaca_trading_enabled:
            open_orders = [o for o in await self.broker.open_orders() if is_ours(o.client_order_id)]
            for o in open_orders:
                try:
                    await self.broker.cancel(o.id)
                    canceled += 1
                except BrokerError as exc:
                    logger.warning("kill switch could not cancel %s: %s", o.client_order_id, exc)
        await self._event(
            "kill_switch_activated" if active else "kill_switch_released",
            (
                f"Kill switch ON: no new orders ({reason or 'no reason given'})"
                if active
                else "Kill switch released"
            )
            + (f"; canceled {canceled} open order(s)" if canceled else ""),
            details={"reason": reason, "canceled": canceled},
        )
        return await self.kill_switch()

    async def cancel_all(self, confirm: bool) -> ActionOut:
        if not confirm:
            raise DomainError("confirm must be true to cancel every open order on the Alpaca paper account")
        self._require_broker()
        if not self._s.alpaca_trading_enabled:
            raise DomainError(
                "QP_ALPACA_TRADING_ENABLED=false: QuantPulse does not change the Alpaca account (enable it first)"
            )
        n = await self.broker.cancel_all()
        await self._event(
            "orders_canceled", f"Cancel all: Alpaca accepted {n} cancellation(s)", details={"count": n}
        )
        await self.orders.reconcile()
        mode, _ = await self._mode()
        return ActionOut(
            message=f"Canceled {n} open order(s) on the Alpaca paper account", mode=mode, canceled=n
        )

    async def close_all(self, confirm: str) -> ActionOut:
        """Sell every position (market orders), after canceling open orders. Needs the exact phrase."""
        if confirm.strip() != CLOSE_ALL_PHRASE:
            raise DomainError(f"type exactly {CLOSE_ALL_PHRASE!r} to close every position")
        self._require_broker()
        async with self._lock:
            now = self._clock.now()
            s = self._s
            execute = s.alpaca_trading_enabled and not s.trading_dry_run
            mode = "paper" if execute else "dry_run"
            canceled = await self.broker.cancel_all() if execute else 0
            if canceled:
                await asyncio.sleep(1.0)
            account, positions, open_orders = await self._snapshot()
            is_open, _ = await self._market()
            kill = await self.kill_switch()
            quotes = await self._position_quotes(positions)
            book = RiskBook(self.limits, account, positions, open_orders, is_open, kill.active, quotes)
            slot = f"x{now.astimezone(NEW_YORK):%Y%m%dT%H%M}"
            trades: list[ProposedTradeOut] = []
            submitted = 0
            for p in positions.values():
                t = ProposedTrade(
                    p.symbol,
                    "sell",
                    p.qty,
                    p.current_price,
                    "flatten",
                    "manual close-all",
                    0.0,
                    0.0,
                    None,
                    closes_position=True,
                )
                out, sub = await self._risk_and_submit(t, book, None, slot, mode, None, flatten=True)
                trades.append(out)
                submitted += int(bool(sub and sub.submitted))
            await self._event(
                "close_all",
                f"Close all ({mode}): {len(trades)} position(s), {submitted} order(s) sent",
                details={"symbols": list(positions), "canceled": canceled},
            )
            msg = (
                f"Sent {submitted} closing order(s) for {len(trades)} position(s)"
                if execute
                else f"DRY RUN: would close {len(trades)} position(s); nothing was sent"
            )
            return ActionOut(message=msg, mode=mode, submitted=submitted, canceled=canceled, trades=trades)

    async def reconcile(self, trigger: str = "manual") -> ReconcileOut:
        self._require_broker()
        if trigger == "startup":
            await self._close_interrupted()
        report = await self.orders.reconcile()
        account, positions = await asyncio.gather(self.broker.account(), self.broker.positions())
        await self._baseline(account)
        await self._forget_closed({p.symbol for p in positions})
        now = self._clock.now()
        self._last_reconcile = now
        await self._event(
            "reconciliation_completed",
            f"Reconciled with Alpaca ({trigger}): {len(positions)} position(s), {report.open_orders} open order(s), "
            f"{report.updated} order update(s), {report.added} added",
            details={"changes": report.changes[:50], "equity": account.equity},
        )
        return ReconcileOut(
            at=now,
            equity=account.equity,
            positions=len(positions),
            open_orders=report.open_orders,
            orders_checked=report.checked,
            orders_updated=report.updated,
            orders_added=report.added,
            unknown_resolved=report.resolved_unknown,
            changes=report.changes,
        )

    async def _close_interrupted(self) -> None:
        """Cycles still marked running when the process starts were cut short by a restart. Their orders
        keep their client ids (never resent); reconciliation settles what actually happened."""
        now = self._clock.now()
        async with self._db.session() as s:
            stuck = await repo.trading_cycles(s, 100, statuses=["running"])
            for row in stuck:
                row.status, row.finished_at = "failed", now
                row.error = "interrupted by a restart; orders it sent are reconciled with Alpaca"
                await repo.add_trading_event(
                    s,
                    "cycle_failed",
                    f"Cycle {row.cycle_key} was interrupted by a restart",
                    now,
                    cycle_id=row.id,
                )

    async def _forget_closed(self, held: set[str]) -> None:
        memory = await self._state(MEMORY_KEY)
        kept = {s: v for s, v in memory.items() if s in held}
        if kept != memory:
            await self._put_state(MEMORY_KEY, kept)

    # ------------------------------------------------------------------ cycles
    async def run(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        wait: float | None = None,
        key: str | None = None,
    ) -> CycleOut:
        """Run one cycle (in the background job registry, so only one runs at a time)."""
        self._require_broker()
        local = self._clock.now().astimezone(NEW_YORK)
        cycle_key = key or f"m{local:%Y%m%dT%H%M}"

        async def work(job: Job) -> CycleOut:
            return await self._cycle(cycle_key, trigger, dry_run, job.reporter(0.0, 1.0))

        job = self._jobs.start("trading", JOB_KEY, f"Trading cycle {cycle_key}", work)
        try:
            return await self._jobs.wait(job, wait)
        except JobPending:
            raise TradingCycleRunning(job) from None

    def cycle_job(self) -> Job | None:
        return self._jobs.latest(JOB_KEY)

    async def run_scheduled(self) -> str:
        """Called every minute by the poller: startup reconciliation, the scheduled cycle for the current
        slot (once), and a reconciliation every few minutes while the market is open."""
        s = self._s
        if not s.trading_scheduler_enabled:
            return "disabled"
        if not self.broker.configured():
            return "not configured (no Alpaca paper keys)"
        now = self._clock.now()
        if self._retry_at is not None and now < self._retry_at:
            return f"backing off until {self._retry_at.astimezone(NEW_YORK):%H:%M}"
        try:
            if not self._started:
                await self.reconcile("startup")
                self._started = True
            slot = self.current_slot()
            if slot is None:
                if is_market_open(now) and self._reconcile_due(now):
                    await self.reconcile("periodic")
                return "no cycle due"
            key = f"{slot:%Y%m%dT%H%M}"
            async with self._db.session() as sess:
                done = await repo.trading_cycle_by_key(sess, key)
            if done is not None:
                if is_market_open(now) and self._reconcile_due(now):
                    await self.reconcile("periodic")
                return f"cycle {key} {done.status}"
            cycle = await self.run(trigger="schedule", key=key)
            self._retry_at = None
            return f"cycle {key} {cycle.status} ({cycle.mode}, {len(cycle.trades)} trade(s))"
        except BrokerError:
            self._retry_at = now + SCHEDULER_BACKOFF
            raise

    def _reconcile_due(self, now: datetime) -> bool:
        return self._last_reconcile is None or now - self._last_reconcile >= RECONCILE_EVERY

    async def _snapshot(self) -> tuple[BrokerAccount, dict[str, BrokerPosition], list[BrokerOrder]]:
        account, positions, open_orders = await asyncio.gather(
            self.broker.account(), self.broker.positions(), self.broker.open_orders()
        )
        return account, {p.symbol: p for p in positions}, open_orders

    async def _cycle(self, key: str, trigger: str, force_dry_run: bool, progress: Progress) -> CycleOut:
        async with self._lock:
            now = self._clock.now()
            mode, kill = await self._mode(force_dry_run)
            try:
                async with self._db.session() as s:
                    existing = await repo.trading_cycle_by_key(s, key)
                    if existing is not None:
                        return self._cycle_out(existing)
                    row = await repo.create_trading_cycle(
                        s,
                        cycle_key=key,
                        trigger=trigger,
                        mode=mode,
                        status="running",
                        started_at=now,
                        regime={},
                        positions=[],
                        signals=[],
                        targets=[],
                        trades=[],
                        plan={},
                        notes=[],
                    )
                    cycle_id = row.id
            except IntegrityError:
                async with self._db.session() as s:
                    found = await repo.trading_cycle_by_key(s, key)
                assert found is not None
                return self._cycle_out(found)
            try:
                record = await self._execute(cycle_id, key, mode, kill, progress)
                status, error = record.pop("status", "completed"), None
            except (BrokerError, DomainError) as exc:
                logger.warning("trading cycle %s failed: %s", key, exc)
                record, status, error = {}, "failed", str(exc)
            except Exception as exc:
                logger.exception("trading cycle %s crashed", key)
                record, status, error = {}, "failed", f"{type(exc).__name__}: {exc}"
            async with self._db.session() as s:
                row = await repo.get_trading_cycle(s, cycle_id)
                for k, v in record.items():
                    setattr(row, k, v)
                row.status, row.error, row.finished_at = status, error, self._clock.now()
                if status == "failed":
                    await repo.add_trading_event(
                        s,
                        "cycle_failed",
                        f"Cycle {key} failed: {error}",
                        self._clock.now(),
                        cycle_id=cycle_id,
                    )
                else:
                    await repo.add_trading_event(
                        s,
                        "cycle_completed" if status == "completed" else "cycle_skipped",
                        f"Cycle {key} ({mode}) {status}: "
                        + (row.skip_reason or f"{len(row.trades)} trade(s) proposed"),
                        self._clock.now(),
                        cycle_id=cycle_id,
                    )
                await s.flush()
                return self._cycle_out(row)

    async def _execute(
        self, cycle_id: int, key: str, mode: str, kill: KillSwitchOut, progress: Progress
    ) -> dict[str, Any]:
        s = self._s
        notes: list[str] = []
        if kill.active:
            notes.append(
                f"Kill switch ON ({kill.reason or kill.source}): every order is refused; this is a dry run"
            )
        elif mode == "dry_run":
            notes.append(DRY_RUN_BANNER)

        # 1. Alpaca is authoritative: reconcile and read the account
        progress(0.01, "reconciling with Alpaca")
        await self.orders.reconcile()
        self._last_reconcile = self._clock.now()
        account, positions, open_orders = await self._snapshot()
        await self._baseline(account)
        is_open, _ = await self._market()
        if mode == "paper":
            canceled = await self.orders.cancel_stale(open_orders)
            if canceled:
                await asyncio.sleep(1.0)
                account, positions, open_orders = await self._snapshot()
                notes.append(f"canceled {len(canceled)} stale unfilled order(s) before re-planning")
        if not is_open:
            notes.append("The market is closed: orders would be refused (risk check market_open)")

        # 2. data
        inputs = await self._data.load(list(positions), progress=lambda f, st: progress(0.05 + 0.5 * f, st))
        notes.extend(inputs.notes)
        if inputs.model_note:
            notes.append(inputs.model_note)
        if inputs.price_status is DataStatus.SYNTHETIC:
            notes.append("Prices are SYNTHETIC (no live data): nothing will be bought or sold")

        # 3. regime
        progress(0.6, "classifying the market regime")
        regime = self._regime(inputs)
        exposure_share = s.trading_regime_exposure.get(regime.label, 1.0)
        entry_penalty = s.trading_regime_entry_penalty.get(regime.label, 0.0)

        # 4. signals (two passes: implied vol and earnings dates are fetched for the leaders only)
        progress(0.65, "scoring opportunities")
        table = self._score(inputs, regime.beta_tilt, {})
        leaders = list(table["score"].sort_values(ascending=False).index[: 2 * s.trading_max_positions])
        leaders = list(dict.fromkeys([*leaders, *[p for p in positions if p in table.index]]))
        progress(0.72, "implied volatility and earnings for the leaders")
        await self._data.enrich(inputs, leaders)
        if inputs.implied_vol:
            table = self._score(inputs, regime.beta_tilt, inputs.implied_vol)
        await self._event(
            "signal_generated",
            f"Scored {len(table)} symbols; leaders: "
            + ", ".join(f"{sym} {table.loc[sym, 'score']:+.2f}" for sym in table["score"].nlargest(5).index),
            cycle_id=cycle_id,
            details={"regime": regime.label},
        )

        # 5. portfolio
        progress(0.8, "building the target portfolio")
        candidates = self._candidates(inputs, table)
        memory = {sym: PositionMemory(**v) for sym, v in (await self._state(MEMORY_KEY)).items()}
        last_traded = await self.orders.last_trades(
            self._clock.now() - timedelta(minutes=self.strategy.cooldown_minutes)
        )
        working = {o.symbol for o in open_orders if o.is_open} | await self.orders.unresolved_symbols()
        holdings = {
            sym: Holding(sym, p.qty, p.avg_entry_price, p.current_price, p.market_value, p.unrealized_plpc)
            for sym, p in positions.items()
            if p.qty > 0
        }
        daily_hit = account.last_equity > 0 and account.day_pl_pct <= -self.limits.max_daily_loss_pct
        if daily_hit:
            await self._daily_loss_event(account)
        if daily_hit and s.trading_daily_loss_action == "flatten":
            plan = PortfolioPlan(
                gross_target=0.0,
                targets={},
                trades=[
                    ProposedTrade(
                        sym,
                        "sell",
                        h.qty,
                        (candidates[sym].price if sym in candidates else h.current_price),
                        "daily_loss_flatten",
                        f"daily loss {account.day_pl_pct:+.2%} hit the −{self.limits.max_daily_loss_pct:.0%} limit "
                        "(QP_TRADING_DAILY_LOSS_ACTION=flatten)",
                        h.market_value / account.equity if account.equity > 0 else 0.0,
                        0.0,
                        None,
                        closes_position=True,
                    )
                    for sym, h in holdings.items()
                    if sym not in working
                ],
                notes=["daily loss limit reached: flattening"],
            )
        else:
            plan = build_plan(
                candidates,
                holdings,
                account.equity,
                self.strategy,
                regime_exposure=exposure_share,
                entry_penalty=entry_penalty,
                risk_off=regime.label == "risk_off",
                memory=memory,
                last_traded=last_traded,
                working=working,
                now=self._clock.now(),
            )
        notes.extend(plan.notes)

        # 6-7. risk and execution
        progress(0.85, "risk checks" + (" and orders" if mode == "paper" else " (dry run)"))
        quotes = self._quote_checks(inputs, table)
        book = RiskBook(self.limits, account, positions, open_orders, is_open, kill.active, quotes, daily_hit)
        trade_rows: list[ProposedTradeOut] = []
        submitted: list[Submission] = []
        sells = [t for t in plan.trades if t.side == "sell"]
        buys = [t for t in plan.trades if t.side == "buy"]
        for t in sells:
            out, sub = await self._risk_and_submit(t, book, inputs.quotes.get(t.symbol), key, mode, cycle_id)
            trade_rows.append(out)
            if sub is not None:
                submitted.append(sub)
                await self._remember(t, sub, candidates, positions)
        if buys:
            if mode == "paper" and any(x.submitted for x in submitted):
                progress(0.9, "waiting for sells to fill")
                await self.orders.wait_for(
                    [x.client_order_id for x in submitted if x.submitted], s.trading_fill_wait_seconds
                )
                account, positions, open_orders = await self._snapshot()
                book = RiskBook(
                    self.limits, account, positions, open_orders, is_open, kill.active, quotes, daily_hit
                )
            elif mode != "paper":
                for row_out, t in zip(trade_rows, sells, strict=True):
                    if row_out.approved:  # a dry run assumes approved sells fill at the estimate
                        book.cash_left += t.notional
                        book.buying_power_left += t.notional
                        book.exposure -= t.notional
                        book.value[t.symbol] = book.value.get(t.symbol, 0.0) - t.notional
                        if t.closes_position:
                            book.held[t.symbol] = 0.0
            for t in buys:
                out, sub = await self._risk_and_submit(
                    t, book, inputs.quotes.get(t.symbol), key, mode, cycle_id
                )
                trade_rows.append(out)
                if sub is not None:
                    submitted.append(sub)
                    await self._remember(t, sub, candidates, positions)

        # 8. fills and the record
        sent = [x.client_order_id for x in submitted if x.submitted]
        if sent:
            progress(0.95, "reconciling fills")
            final = await self.orders.wait_for(sent, s.trading_fill_wait_seconds)
            by_cid = {o.client_order_id: o for o in final.values()}
            trade_rows = [
                r.model_copy(update={"status": by_cid[r.client_order_id].status})
                if r.client_order_id in by_cid
                else r
                for r in trade_rows
            ]
            account, positions, _ = await self._snapshot()
        await self._forget_closed(set(positions) | {x.symbol for x in submitted if x.submitted})
        return {
            "status": "completed",
            "equity": account.equity,
            "last_equity": account.last_equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "long_market_value": account.long_market_value,
            "data_status": inputs.price_status.value,
            "regime": TradingRegimeOut(
                label=regime.label,
                description=regime.description,
                trend_score=regime.trend_score,
                stressed=regime.stressed,
                exposure_share=exposure_share,
                entry_penalty=entry_penalty,
                beta_tilt=regime.beta_tilt,
                metrics=regime.metrics,
                reasons=regime.reasons,
            ).model_dump(mode="json"),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "weight": p.market_value / account.equity if account.equity > 0 else 0.0,
                    "unrealized_plpc": p.unrealized_plpc,
                }
                for p in positions.values()
            ],
            "signals": [
                x.model_dump(mode="json")
                for x in self._signal_rows(table, candidates, plan, inputs, positions)
            ],
            "targets": [
                TargetOut(
                    symbol=t.symbol,
                    weight=t.weight,
                    score=t.score,
                    conviction=t.conviction,
                    risk_vol=t.risk_vol,
                    capped_by=t.capped_by,
                    incumbent=t.incumbent,
                ).model_dump(mode="json")
                for t in sorted(plan.targets.values(), key=lambda t: -t.weight)
            ],
            "trades": [r.model_dump(mode="json") for r in trade_rows],
            "plan": {"gross_target": plan.gross_target, "exits": plan.exits, "skipped": plan.skipped},
            "notes": notes,
        }

    # ------------------------------------------------------------------ cycle helpers
    def _regime(self, inputs: TradingInputs) -> regime_mod.MarketRegime:
        spy = inputs.benchmark
        qqq = inputs.qqq
        if inputs.session_open:
            day = pd.Timestamp(inputs.as_of.astimezone(NEW_YORK).date())
            bq = inputs.quotes.get(self._s.benchmark_symbol)
            if bq is not None:
                spy = pd.concat([spy, pd.Series([bq.price], index=[day])])
            qq = inputs.quotes.get("QQQ")
            if qq is not None and qqq is not None:
                qqq = pd.concat([qqq, pd.Series([qq.price], index=[day])])
        stocks = [c for c in inputs.close.columns if c not in self._s.trading_etfs]
        b50, b200 = regime_mod.breadth(inputs.close[stocks]) if stocks else (None, None)
        return regime_mod.classify(spy, qqq, breadth_50=b50, breadth_200=b200, vix=inputs.vix)

    def _score(
        self, inputs: TradingInputs, beta_tilt: float, implied_vol: Mapping[str, float]
    ) -> pd.DataFrame:
        close, high, low, volume = inputs.close, inputs.high, inputs.low, inputs.volume
        bench = inputs.benchmark
        quoted = [c for c in close.columns if c in inputs.quotes]
        if quoted:  # rank what can actually be traded
            close, high, low, volume = close[quoted], high[quoted], low[quoted], volume[quoted]
        live_row = inputs.session_open and bool(quoted)
        vwap: dict[str, float] = {}
        if live_row:
            day = pd.Timestamp(inputs.as_of.astimezone(NEW_YORK).date())
            bars = {sym: inputs.quotes[sym].bar() for sym in quoted}
            close, high, low, volume = ts.append_live_row(close, high, low, volume, bars, day)
            bq = inputs.quotes.get(self._s.benchmark_symbol)
            bench = pd.concat([bench, pd.Series([bq.price if bq else float("nan")], index=[day])])
            vwap = {sym: q.vwap for sym, q in inputs.quotes.items() if q.vwap}
        raw = ts.raw_signals(
            close,
            high,
            low,
            volume,
            bench,
            live_row=live_row,
            vwap=vwap,
            session_fraction=inputs.session_fraction,
            implied_vol=implied_vol,
        )
        raw = raw[raw["price"].notna()]
        comps = ts.component_scores(
            raw,
            model_z=inputs.model_z,
            fundamentals=inputs.fundamentals([n for n, _ in ts.FUNDAMENTAL_TERMS]),
            beta_tilt=beta_tilt,
        )
        score = ts.opportunity_score(comps, self._s.trading_signal_weights)
        return raw.join(comps).assign(score=score)

    def _candidates(self, inputs: TradingInputs, table: pd.DataFrame) -> dict[str, Candidate]:
        s = self._s
        today = inputs.as_of.astimezone(NEW_YORK).date()
        out: dict[str, Candidate] = {}
        for sym, r in table.iterrows():
            symbol = str(sym)
            q = inputs.quotes.get(symbol)
            if q is None:
                continue  # no live quote: cannot be traded (holdings stay untouched)
            blocks: list[str] = []
            if inputs.price_status is DataStatus.SYNTHETIC:
                blocks.append("synthetic price history: never traded")
            if q.price < s.trading_min_price:
                blocks.append(f"price ${q.price:,.2f} below ${s.trading_min_price:,.2f}")
            adv = _fin(r.get("adv_dollar"))
            if adv is None or adv < s.trading_min_dollar_volume:
                blocks.append("insufficient dollar volume")
            if q.spread_bps is not None and q.spread_bps > s.trading_max_spread_bps:
                blocks.append(f"spread {q.spread_bps:.0f}bp too wide")
            if q.age_seconds > s.trading_max_quote_age_seconds:
                blocks.append(f"quote {q.age_seconds:.0f}s old")
            if symbol in inputs.earnings:
                when, source = inputs.earnings[symbol]
                if 0 <= (when - today).days <= s.trading_earnings_blackout_days:
                    blocks.append(f"earnings on {when} ({source}): no new position before the release")
            if int(r.get("bars") or 0) < ts.MIN_HISTORY:
                blocks.append("not enough price history")
            out[symbol] = Candidate(
                symbol=symbol,
                score=float(r["score"]),
                price=q.price,
                risk_vol=_fin(r.get("risk_vol")) or s.trading_vol_floor,
                adv_dollar=adv,
                trend_ok=bool(r.get("trend_ok")),
                trend_broken=bool(r.get("trend_broken")),
                model_z=inputs.model_z.get(symbol),
                entry_blocks=tuple(blocks),
                components={c: round(float(r[c]), 4) for c in ts.COMPONENTS},
            )
        return out

    def _quote_checks(self, inputs: TradingInputs, table: pd.DataFrame) -> dict[str, QuoteCheck]:
        out: dict[str, QuoteCheck] = {}
        synthetic = inputs.price_status is DataStatus.SYNTHETIC
        for sym, q in inputs.quotes.items():
            adv = _fin(table.loc[sym, "adv_dollar"]) if sym in table.index else None
            out[sym] = QuoteCheck(
                price=q.price,
                status=DataStatus.SYNTHETIC if synthetic else DataStatus.LIVE,
                provider=q.provider,
                age_seconds=q.age_seconds,
                spread_bps=q.spread_bps,
                adv_dollar=adv,
            )
        return out

    async def _position_quotes(self, positions: Mapping[str, BrokerPosition]) -> dict[str, QuoteCheck]:
        """Prices for closing held positions: live quotes where available, else Alpaca's own live mark
        (so a market-data outage can never trap a position)."""
        live = await self._data.live_quotes(list(positions))
        out: dict[str, QuoteCheck] = {}
        for sym, p in positions.items():
            q = live.get(sym)
            if q is not None:
                out[sym] = QuoteCheck(q.price, DataStatus.LIVE, q.provider, q.age_seconds, q.spread_bps, None)
            elif p.current_price > 0:
                out[sym] = QuoteCheck(
                    p.current_price, DataStatus.LIVE, "alpaca position mark", None, None, None
                )
        return out

    def _order_params(self, t: ProposedTrade, q: LiveQuote | None, flatten: bool) -> tuple[str, float | None]:
        kind = self._s.trading_order_type
        whole = float(t.qty).is_integer()
        if flatten or kind == "market" or not whole:
            return "market", None
        if kind == "limit":
            return "limit", _round_price(t.est_price)
        off = self._s.trading_limit_offset_bps / 10_000
        ref = t.est_price
        if q is not None:
            side_px = q.ask if t.side == "buy" else q.bid
            if side_px and side_px > 0 and abs(side_px / t.est_price - 1) < 0.02:
                ref = side_px
        limit = ref * (1 + off) if t.side == "buy" else ref * (1 - off)
        return "marketable_limit", _round_price(limit)

    async def _risk_and_submit(
        self,
        t: ProposedTrade,
        book: RiskBook,
        quote: LiveQuote | None,
        slot: str,
        mode: str,
        cycle_id: int | None,
        *,
        flatten: bool = False,
    ) -> tuple[ProposedTradeOut, Submission | None]:
        intent = OrderIntent(
            symbol=t.symbol,
            side=t.side,
            qty=t.qty,
            est_price=t.est_price,
            kind=t.kind,
            reason=t.reason,
            closes_position=t.closes_position,
            score=t.score,
            intent="flatten" if flatten or t.kind == "daily_loss_flatten" else "strategy",
        )
        decision: RiskDecision = book.evaluate(intent)
        order_type, limit = self._order_params(t, quote, flatten or intent.intent == "flatten")
        cid = client_order_id(slot, t.symbol, t.side)
        async with self._db.session() as s:
            now = self._clock.now()
            await repo.add_trading_event(
                s,
                "trade_proposed",
                f"{t.side.upper()} {t.qty:g} {t.symbol} ({t.kind}): {t.reason}",
                now,
                cycle_id=cycle_id,
                symbol=t.symbol,
                details={"notional": round(t.notional, 2), "score": t.score},
            )
            await repo.add_trading_event(
                s,
                "risk_approved" if decision.approved else "risk_rejected",
                f"{t.symbol} {t.side}: {decision.summary}",
                now,
                cycle_id=cycle_id,
                symbol=t.symbol,
                details={"checks": [c.name for c in decision.failures]},
            )
        base = ProposedTradeOut(
            symbol=t.symbol,
            side=t.side,
            qty=t.qty,
            est_price=t.est_price,
            notional=t.notional,
            kind=t.kind,
            reason=t.reason,
            current_weight=t.current_weight,
            target_weight=t.target_weight,
            score=t.score,
            approved=decision.approved,
            risk=decision.summary,
            checks=[RiskCheckOut(name=c.name, passed=c.passed, detail=c.detail) for c in decision.checks],
            order_type=order_type,
            limit_price=limit,
            client_order_id=cid if decision.approved and mode == "paper" else None,
            status="risk_rejected"
            if not decision.approved
            else ("dry_run" if mode != "paper" else "not_submitted"),
        )
        if not decision.approved or mode != "paper":
            if decision.approved:
                book.commit(intent)
            return base, None
        book.commit(intent)
        sub = await self.orders.submit(
            intent, cid=cid, order_type=order_type, limit_price=limit, cycle_id=cycle_id, strategy=STRATEGY
        )
        return base.model_copy(update={"status": sub.status, "error": sub.error}), sub

    async def _remember(
        self,
        t: ProposedTrade,
        sub: Submission,
        candidates: Mapping[str, Candidate],
        positions: Mapping[str, BrokerPosition],
    ) -> None:
        if not sub.submitted:
            return
        memory = await self._state(MEMORY_KEY)
        entry = dict(memory.get(t.symbol, {}))
        if t.kind == "entry":
            c = candidates.get(t.symbol)
            entry = {
                "entry_score": t.score,
                "entry_model_z": c.model_z if c else None,
                "profit_taken_basis": None,
            }
        elif t.kind == "take_profit" and t.symbol in positions:
            entry["profit_taken_basis"] = positions[t.symbol].avg_entry_price
        elif t.closes_position:
            memory.pop(t.symbol, None)
            await self._put_state(MEMORY_KEY, memory)
            return
        else:
            return
        memory[t.symbol] = entry
        await self._put_state(MEMORY_KEY, memory)

    async def _daily_loss_event(self, account: BrokerAccount) -> None:
        today = self._clock.now().astimezone(NEW_YORK).date().isoformat()
        st = await self._state(DAY_KEY)
        if st.get("date") == today:
            return
        await self._put_state(DAY_KEY, {"date": today, "day_pl_pct": account.day_pl_pct})
        await self._event(
            "daily_loss_limit_reached",
            f"Daily loss {account.day_pl_pct:+.2%} reached the −{self.limits.max_daily_loss_pct:.0%} limit: no new "
            f"positions today (policy: {self._s.trading_daily_loss_action})",
            details={"equity": account.equity, "last_equity": account.last_equity},
        )

    def _signal_rows(
        self,
        table: pd.DataFrame,
        candidates: Mapping[str, Candidate],
        plan: PortfolioPlan,
        inputs: TradingInputs,
        positions: Mapping[str, BrokerPosition],
    ) -> list[SignalOut]:
        ordered = table["score"].sort_values(ascending=False)
        keep = list(ordered.index[:TOP_SIGNALS]) + [p for p in positions if p in table.index]
        rows: list[SignalOut] = []
        for sym in dict.fromkeys(keep):
            r = table.loc[sym]
            c = candidates.get(str(sym))
            earn_date = inputs.earnings.get(str(sym), (None, None))[0]
            rows.append(
                SignalOut(
                    symbol=str(sym),
                    rank=int(list(ordered.index).index(sym)) + 1,
                    score=float(r["score"]),
                    components={comp: round(float(r[comp]), 4) for comp in ts.COMPONENTS},
                    price=float(r["price"]),
                    risk_vol=_fin(r.get("risk_vol")),
                    adv_dollar=_fin(r.get("adv_dollar")),
                    trend_ok=bool(r.get("trend_ok")),
                    trend_broken=bool(r.get("trend_broken")),
                    model_z=inputs.model_z.get(str(sym)),
                    implied_vol=inputs.implied_vol.get(str(sym)),
                    earnings_date=earn_date,
                    entry_blocks=list(c.entry_blocks) if c else ["no live quote"],
                    target_weight=plan.targets[str(sym)].weight if str(sym) in plan.targets else 0.0,
                    held=str(sym) in positions,
                )
            )
        return rows

    def _cycle_out(self, row: TradingCycleRow) -> CycleOut:
        plan = row.plan or {}
        return CycleOut(
            id=row.id,
            cycle_key=row.cycle_key,
            trigger=row.trigger,
            mode=row.mode,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            skip_reason=row.skip_reason,
            error=row.error,
            equity=row.equity,
            last_equity=row.last_equity,
            cash=row.cash,
            buying_power=row.buying_power,
            long_market_value=row.long_market_value,
            data_status=DataStatus(row.data_status) if row.data_status else None,
            regime=TradingRegimeOut.model_validate(row.regime) if row.regime else None,
            gross_target=plan.get("gross_target"),
            positions=list(row.positions or []),
            signals=[SignalOut.model_validate(x) for x in row.signals or []],
            targets=[TargetOut.model_validate(x) for x in row.targets or []],
            trades=[ProposedTradeOut.model_validate(x) for x in row.trades or []],
            exits=dict(plan.get("exits") or {}),
            skipped=dict(plan.get("skipped") or {}),
            notes=list(row.notes or []),
        )


def _event_out(r: TradingEventRow) -> TradingEventOut:
    return TradingEventOut(
        id=r.id,
        created_at=r.created_at,
        cycle_id=r.cycle_id,
        kind=r.kind,
        symbol=r.symbol,
        client_order_id=r.client_order_id,
        message=r.message,
        details=r.details or {},
    )
