"""Paper-trading sandbox: accounts, the self-learning agent's daily step, manual orders and training.

Nothing here talks to a broker. Cash, positions and fills are simulated in the warehouse; prices come
from the same data gateway as the rest of the platform, and by default an account refuses to trade on
synthetic prices so its paper record only ever reflects real quotes.

Daily agent step (at most once per New York trading day unless forced)
    1. load quotes + ~400 days of history for the universe and current holdings;
    2. *learn*: score yesterday's decision against the returns realised since (see
       :mod:`quantpulse.domain.trading_agent`) and update the factor weights;
    3. *decide*: re-rank the universe with the updated weights and pick the target portfolio;
    4. *trade*: rebalance the paper book at the live quote ± slippage, then persist trades, positions,
       the learning state, an equity snapshot and journal entries explaining what happened.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.gateway import Resolved
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day
from quantpulse.db import repositories as repo
from quantpulse.db.models import SandboxAccountRow, SandboxEquityRow, SandboxJournalRow, SandboxTradeRow
from quantpulse.db.session import Database
from quantpulse.domain import screener
from quantpulse.domain import trading_agent as agent
from quantpulse.domain.paper_broker import ExecutionModel, Fill, Holding, PaperBook, floor_quantity
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus, Provenance
from quantpulse.schemas.market import PriceHistory, Quote
from quantpulse.schemas.sandbox import (
    AccountCreate,
    AccountOut,
    AccountSummary,
    AccountUpdate,
    BacktestMetrics,
    Candidate,
    CurvePoint,
    EquityPoint,
    JournalEntry,
    LessonOut,
    OrderIn,
    PerformanceOut,
    PositionOut,
    SandboxStrategy,
    StepResult,
    TradeOut,
    TrainReport,
    TrainRequest,
    WeightSnapshot,
)
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.notifications import SyntheticDataRefused
from quantpulse.services.picks import HISTORY_DAYS, closes_with_quote
from quantpulse.services.portfolio import closes_frame
from quantpulse.services.rates import RatesService

logger = logging.getLogger(__name__)

MIN_RANKABLE = 3  # the cross-sectional z-scores need a few names to mean anything
RETRY_AFTER = timedelta(minutes=15)  # scheduler back-off after a skipped or failed run
TRAIN_WARMUP = 253  # bars before the first training decision: the full factor set (12-1 momentum) exists
MIN_REPLAY_DAYS = 60
MIN_TRAIN_SYMBOLS = 5
MIN_COVERAGE = 0.9  # a symbol must have ≥90% of the benchmark's trading days to join a replay
FETCH_CONCURRENCY = 4


def _config(strategy: SandboxStrategy) -> agent.AgentConfig:
    return agent.AgentConfig(
        top_k=strategy.top_k,
        max_position=strategy.max_position,
        cash_buffer=strategy.cash_buffer,
        learning_rate=strategy.learning_rate,
        prior_shrink=strategy.prior_shrink,
        weight_floor=strategy.weight_floor,
        min_trade_value=strategy.min_trade_value,
    )


def _execution(strategy: SandboxStrategy) -> ExecutionModel:
    return ExecutionModel(
        slippage_bps=strategy.slippage_bps,
        commission_per_trade=strategy.commission_per_trade,
        commission_bps=strategy.commission_bps,
    )


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _round_weights(weights: dict[str, float]) -> dict[str, float]:
    return {f: round(w, 6) for f, w in weights.items()}


def _max_drawdown(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1.0)
    return worst


def _trade_out(row: SandboxTradeRow) -> TradeOut:
    return TradeOut(
        id=row.id,
        executed_at=row.executed_at,
        symbol=row.symbol,
        side=row.side,
        quantity=row.quantity,
        price=row.price,
        reference_price=row.reference_price,
        notional=row.quantity * row.price,
        commission=row.commission,
        realized_pnl=row.realized_pnl,
        data_status=DataStatus(row.data_status),
        source=row.source,
        note=row.note,
    )


def _equity_point(row: SandboxEquityRow) -> EquityPoint:
    return EquityPoint(
        recorded_at=row.recorded_at,
        equity=row.equity,
        cash=row.cash,
        benchmark_price=row.benchmark_price,
        data_status=DataStatus(row.data_status),
    )


def _journal_entry(row: SandboxJournalRow) -> JournalEntry:
    return JournalEntry(
        id=row.id, created_at=row.created_at, kind=row.kind, summary=row.summary, details=row.details
    )


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _lesson_summary(lessons: Sequence[agent.Lesson], since: str) -> str:
    strongest = sorted(lessons, key=lambda lesson: -abs(lesson.ic))[:3]
    parts = [
        f"{screener.LABELS[lesson.factor]} IC {lesson.ic:+.2f} "
        f"({_pct(lesson.weight_before)} → {_pct(lesson.weight_after)})"
        for lesson in strongest
    ]
    return f"Scored the {since} decision against returns since: " + "; ".join(parts)


def _decision_summary(fills: Sequence[Fill], targets: dict[str, float]) -> str:
    buys = sorted({f.symbol for f in fills if f.side == "buy"})
    sells = sorted({f.symbol for f in fills if f.side == "sell"})
    held = ", ".join(sorted(targets)) or "nothing (all cash)"
    actions = []
    if buys:
        actions.append("bought " + ", ".join(buys))
    if sells:
        actions.append("sold " + ", ".join(sells))
    return f"{'; '.join(actions).capitalize() if actions else 'No trades needed'}. Target portfolio: {held}."


class SandboxService:
    def __init__(
        self, settings: Settings, db: Database, clock: Clock, market: MarketService, rates: RatesService
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._rates = rates
        self._locks: dict[int, asyncio.Lock] = {}
        self._retry_at: dict[int, datetime] = {}
        self._skip_logged: dict[int, tuple[date, str]] = {}
        self._marked_on: dict[int, date] = {}
        self.model: ModelService | None = None  # wired by the container; enables the "model" signal

    # ------------------------------------------------------------------ helpers
    def _lock(self, account_id: int) -> asyncio.Lock:
        return self._locks.setdefault(account_id, asyncio.Lock())

    def _local_now(self) -> tuple[datetime, datetime]:
        now = self._clock.now()
        return now, now.astimezone(NEW_YORK)

    def universe(self, strategy: SandboxStrategy) -> list[str]:
        return list(strategy.universe or self._settings.picks_universe)

    def _account_out(self, row: SandboxAccountRow) -> AccountOut:
        strategy = SandboxStrategy.model_validate(row.strategy)
        state = agent.LearningState.from_json(row.state)
        return AccountOut(
            id=row.id,
            name=row.name,
            mode=row.mode,
            starting_cash=row.starting_cash,
            cash=row.cash,
            auto_trade=row.auto_trade,
            allow_synthetic=row.allow_synthetic,
            strategy=strategy,
            universe=self.universe(strategy),
            factor_weights=_round_weights(state.weights),
            prior_weights=_round_weights(agent.normalise(agent.PRIOR)),
            ic_ema={f: round(v, 4) for f, v in state.ic_ema.items()},
            periods_learned=state.periods_learned,
            last_decision_on=date.fromisoformat(state.last_decision_on) if state.last_decision_on else None,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def _usable(self, status: DataStatus, allow_synthetic: bool) -> bool:
        return status is not DataStatus.SYNTHETIC or allow_synthetic

    async def _benchmark_price(self, allow_synthetic: bool) -> tuple[float | None, Resolved[Quote]]:
        resolved = await self._market.quote(self._settings.benchmark_symbol)
        price = resolved.value.price if self._usable(resolved.status, allow_synthetic) else None
        return price, resolved

    async def _load(
        self, symbols: Sequence[str]
    ) -> dict[str, tuple[Resolved[PriceHistory], Resolved[Quote]]]:
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def one(symbol: str) -> tuple[Resolved[PriceHistory], Resolved[Quote]]:
            async with sem:
                return (
                    await self._market.history(symbol, "1d", HISTORY_DAYS),
                    await self._market.quote(symbol),
                )

        results = await asyncio.gather(*(one(s) for s in symbols))
        return dict(zip(symbols, results, strict=True))

    @staticmethod
    def _book(cash: float, positions: Iterable[Any]) -> PaperBook:
        return PaperBook(cash=cash, positions={p.symbol: Holding(p.quantity, p.avg_cost) for p in positions})

    @staticmethod
    def _trade_rows(
        account_id: int,
        fills: Sequence[Fill],
        at: datetime,
        statuses: dict[str, DataStatus],
        source: str,
        note: str | None,
    ) -> list[SandboxTradeRow]:
        return [
            SandboxTradeRow(
                account_id=account_id,
                executed_at=at,
                symbol=f.symbol,
                side=f.side,
                quantity=f.quantity,
                price=f.price,
                reference_price=f.reference_price,
                commission=f.commission,
                realized_pnl=f.realized_pnl,
                data_status=statuses[f.symbol].value,
                source=source,
                note=note,
            )
            for f in fills
        ]

    # ------------------------------------------------------------------ accounts
    async def list_all(self) -> list[AccountOut]:
        async with self._db.session() as s:
            return [self._account_out(r) for r in await repo.list_sandbox_accounts(s)]

    async def account(self, account_id: int) -> AccountOut:
        async with self._db.session() as s:
            return self._account_out(await repo.get_sandbox_account(s, account_id))

    async def create(self, data: AccountCreate) -> AccountOut:
        bench_price, _ = await self._benchmark_price(data.allow_synthetic)
        now = self._clock.now()
        try:
            async with self._db.session() as s:
                if await repo.sandbox_name_exists(s, data.name):
                    raise DomainError(f"a sandbox account named '{data.name}' already exists")
                row = SandboxAccountRow(
                    name=data.name,
                    mode=data.mode,
                    starting_cash=data.starting_cash,
                    cash=data.starting_cash,
                    auto_trade=data.auto_trade,
                    allow_synthetic=data.allow_synthetic,
                    strategy=data.strategy.model_dump(mode="json"),
                    state=agent.LearningState().to_json(),
                    created_at=now,
                    updated_at=now,
                )
                await repo.add_row(s, row)
                s.add(
                    SandboxEquityRow(
                        account_id=row.id,
                        recorded_at=now,
                        equity=data.starting_cash,
                        cash=data.starting_cash,
                        benchmark_price=bench_price,
                        data_status=DataStatus.LIVE.value,  # all cash: nothing needs a price
                    )
                )
                await repo.add_sandbox_journal(
                    s,
                    row.id,
                    "created",
                    f"Opened a {data.mode} paper account with ${data.starting_cash:,.2f} of simulated cash.",
                    {"strategy": data.strategy.model_dump(mode="json")},
                    now,
                )
                return self._account_out(row)
        except IntegrityError as exc:
            raise DomainError(f"a sandbox account named '{data.name}' already exists") from exc

    async def update(self, account_id: int, data: AccountUpdate) -> AccountOut:
        changes = data.model_dump(exclude_unset=True, mode="json")
        async with self._lock(account_id), self._db.session() as s:
            row = await repo.get_sandbox_account(s, account_id)
            if data.name is not None and await repo.sandbox_name_exists(s, data.name, exclude_id=account_id):
                raise DomainError(f"a sandbox account named '{data.name}' already exists")
            for field in ("name", "mode", "auto_trade", "allow_synthetic"):
                value = getattr(data, field)
                if field in changes and value is not None:
                    setattr(row, field, value)
            if "strategy" in changes and data.strategy is not None:
                row.strategy = data.strategy.model_dump(mode="json")
            row.updated_at = self._clock.now()
            if changes:
                await repo.add_sandbox_journal(
                    s,
                    account_id,
                    "update",
                    "Settings changed: " + ", ".join(sorted(changes)),
                    changes,
                    row.updated_at,
                )
            await s.flush()
            return self._account_out(row)

    async def delete(self, account_id: int) -> None:
        async with self._lock(account_id), self._db.session() as s:
            await repo.delete_sandbox_account(s, account_id)
        for cache in (self._retry_at, self._skip_logged, self._marked_on):
            cache.pop(account_id, None)

    async def reset(self, account_id: int, *, keep_learning: bool = False) -> AccountOut:
        """Back to starting cash with no positions or history. ``keep_learning`` keeps the learned weights."""
        async with self._lock(account_id):
            async with self._db.session() as s:
                allow = (await repo.get_sandbox_account(s, account_id)).allow_synthetic
            bench_price, _ = await self._benchmark_price(allow)
            now = self._clock.now()
            async with self._db.session() as s:
                row = await repo.get_sandbox_account(s, account_id)
                old = agent.LearningState.from_json(row.state)
                state = agent.LearningState()
                if keep_learning:
                    state.weights, state.ic_ema, state.periods_learned = (
                        old.weights,
                        old.ic_ema,
                        old.periods_learned,
                    )
                await repo.clear_sandbox_history(s, account_id)
                row.cash = row.starting_cash
                row.state = state.to_json()
                row.updated_at = now
                s.add(
                    SandboxEquityRow(
                        account_id=account_id,
                        recorded_at=now,
                        equity=row.starting_cash,
                        cash=row.starting_cash,
                        benchmark_price=bench_price,
                        data_status=DataStatus.LIVE.value,
                    )
                )
                await repo.add_sandbox_journal(
                    s,
                    account_id,
                    "reset",
                    f"Reset to ${row.starting_cash:,.2f} cash"
                    + (
                        " keeping the learned factor weights."
                        if keep_learning
                        else " and default factor weights."
                    ),
                    {"keep_learning": keep_learning},
                    now,
                )
                await s.flush()
                out = self._account_out(row)
        for cache in (self._retry_at, self._skip_logged, self._marked_on):
            cache.pop(account_id, None)
        return out

    async def summary(self, account_id: int) -> CompositeEnvelope[AccountSummary]:
        async with self._db.session() as s:
            row = await repo.get_sandbox_account(s, account_id)
            positions = await repo.sandbox_positions(s, account_id)
            trades, realized, fees = await repo.sandbox_trade_totals(s, account_id)
            first_bench = await repo.first_sandbox_benchmark(s, account_id)
            history = await repo.sandbox_equity(s, account_id)
        bench = self._settings.benchmark_symbol
        symbols = [p.symbol for p in positions]
        quotes = await self._market.quotes(list(dict.fromkeys([*symbols, bench])))
        sources: dict[str, Provenance] = {f"quote:{s}": r.provenance for s, r in quotes.items()}

        market_values = {p.symbol: p.quantity * quotes[p.symbol].value.price for p in positions}
        invested = sum(market_values.values())
        equity = row.cash + invested
        out_positions = [
            PositionOut(
                symbol=p.symbol,
                quantity=p.quantity,
                avg_cost=p.avg_cost,
                price=quotes[p.symbol].value.price,
                market_value=market_values[p.symbol],
                weight=market_values[p.symbol] / equity if equity > 0 else 0.0,
                unrealized_pnl=(quotes[p.symbol].value.price - p.avg_cost) * p.quantity,
                unrealized_pnl_pct=quotes[p.symbol].value.price / p.avg_cost - 1.0 if p.avg_cost > 0 else 0.0,
                data_status=quotes[p.symbol].status,
            )
            for p in positions
        ]
        bench_q = quotes[bench]
        bench_return = None
        if (
            first_bench is not None
            and first_bench.benchmark_price
            and self._usable(bench_q.status, row.allow_synthetic)
        ):
            bench_return = bench_q.value.price / first_bench.benchmark_price - 1.0
        performance = PerformanceOut(
            equity=equity,
            cash=row.cash,
            invested=invested,
            total_return=equity / row.starting_cash - 1.0,
            realized_pnl=realized,
            unrealized_pnl=sum(p.unrealized_pnl for p in out_positions),
            fees_paid=fees,
            trades=trades,
            benchmark=bench,
            benchmark_return=bench_return,
            max_drawdown=_max_drawdown([*(h.equity for h in history), equity]),
            snapshots=len(history),
        )
        status = (
            DataStatus.worst([p.data_status for p in out_positions]) if out_positions else DataStatus.LIVE
        )
        data = AccountSummary(
            account=self._account_out(row),
            positions=out_positions,
            performance=performance,
            data_status=status,
        )
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, self._clock.now()))

    async def trades(self, account_id: int, limit: int = 200) -> list[TradeOut]:
        async with self._db.session() as s:
            await repo.get_sandbox_account(s, account_id)
            return [_trade_out(r) for r in await repo.sandbox_trades(s, account_id, limit)]

    async def equity(self, account_id: int, limit: int = 2000) -> list[EquityPoint]:
        async with self._db.session() as s:
            await repo.get_sandbox_account(s, account_id)
            return [_equity_point(r) for r in await repo.sandbox_equity(s, account_id, limit)]

    async def journal(self, account_id: int, limit: int = 100) -> list[JournalEntry]:
        async with self._db.session() as s:
            await repo.get_sandbox_account(s, account_id)
            return [_journal_entry(r) for r in await repo.sandbox_journal(s, account_id, limit)]

    # ------------------------------------------------------------------ manual orders
    async def order(self, account_id: int, order: OrderIn) -> TradeOut:
        async with self._lock(account_id):
            quote_r = await self._market.quote(order.symbol)
            async with self._db.session() as s:
                row = await repo.get_sandbox_account(s, account_id)
                if not self._usable(quote_r.status, row.allow_synthetic):
                    raise SyntheticDataRefused(
                        f"no live price for {order.symbol} right now (only synthetic data), so the paper order was "
                        "not filled; enable allow_synthetic on the account to trade on simulated prices"
                    )
                strategy = SandboxStrategy.model_validate(row.strategy)
                model = _execution(strategy)
                book = self._book(row.cash, await repo.sandbox_positions(s, account_id))
                ref = quote_r.value.price
                if order.quantity is not None:
                    qty = order.quantity
                elif order.side == "buy":
                    qty = book.max_affordable(ref, model, cash=order.notional)
                else:
                    qty = floor_quantity((order.notional or 0.0) / model.fill_price("sell", ref))
                if qty <= 0:
                    raise DomainError("order is too small to fill a single micro-share")
                if order.side == "buy":
                    fill = book.buy(order.symbol, qty, ref, model)
                else:
                    fill = book.sell(order.symbol, qty, ref, model)
                now = self._clock.now()
                row.cash = book.cash
                row.updated_at = now
                await repo.replace_sandbox_positions(
                    s, account_id, {sym: (h.quantity, h.avg_cost) for sym, h in book.positions.items()}
                )
                (trade,) = self._trade_rows(
                    account_id, [fill], now, {order.symbol: quote_r.status}, "manual", order.note
                )
                await repo.add_row(s, trade)
                await repo.add_sandbox_journal(
                    s,
                    account_id,
                    "order",
                    f"Manual {fill.side} of {fill.quantity:g} {fill.symbol} at ${fill.price:,.2f}.",
                    {"trade_id": trade.id, "data_status": quote_r.status.value},
                    now,
                )
                return _trade_out(trade)

    # ------------------------------------------------------------------ the agent
    async def step(self, account_id: int, *, force: bool = False) -> StepResult:
        """Let the agent learn from its last decision, re-rank the universe and rebalance the paper book."""
        async with self._lock(account_id):
            return await self._step(account_id, force)

    async def _step(self, account_id: int, force: bool) -> StepResult:
        now, local = self._local_now()
        today = local.date()
        async with self._db.session() as s:
            row = await repo.get_sandbox_account(s, account_id)
            positions = await repo.sandbox_positions(s, account_id)
        if row.mode != "agent":
            raise DomainError(
                "this is a manual account: place orders yourself, or switch its mode to 'agent'"
            )
        state = agent.LearningState.from_json(row.state)
        strategy = SandboxStrategy.model_validate(row.strategy)
        config, model = _config(strategy), _execution(strategy)

        def skipped(reason: str) -> StepResult:
            return StepResult(
                account_id=account_id,
                executed=False,
                skipped_reason=reason,
                trading_day=today,
                as_of=now,
                weights_before=_round_weights(state.weights),
                weights_after=_round_weights(state.weights),
                data_status=DataStatus.LIVE,
            )

        if not force:
            if not is_trading_day(today):
                return skipped(f"{today.isoformat()} is not a NYSE trading day")
            if state.last_decision_on == today.isoformat():
                return skipped(f"already traded on {today.isoformat()} (pass force=true to trade again)")

        universe = self.universe(strategy)
        held = [p.symbol for p in positions]
        loaded = await self._load(list(dict.fromkeys([*universe, *held])))
        bench_price, _ = await self._benchmark_price(row.allow_synthetic)

        prices: dict[str, float] = {}
        statuses: dict[str, DataStatus] = {}
        closes: dict[str, list[float]] = {}
        excluded: dict[str, str] = {}
        for symbol, (hist_r, quote_r) in loaded.items():
            if not self._usable(quote_r.status, row.allow_synthetic):
                excluded[symbol] = "no live quote (synthetic prices are not allowed for this account)"
                continue
            prices[symbol] = quote_r.value.price
            statuses[symbol] = quote_r.status
            if symbol not in universe:
                continue
            if not self._usable(hist_r.status, row.allow_synthetic):
                excluded[symbol] = "no live price history (synthetic prices are not allowed for this account)"
                continue
            closes[symbol] = closes_with_quote(hist_r.value, quote_r.value)
            statuses[symbol] = DataStatus.worst([hist_r.status, quote_r.status])
        factors, unscorable = agent.factor_universe(closes)
        excluded.update(unscorable)
        seen = DataStatus.worst([r.status for pair in loaded.values() for r in pair])

        unpriced = sorted(s for s in held if s not in prices)
        if unpriced:
            return await self._skip(
                row, today, f"no live price for held position(s) {', '.join(unpriced)}", excluded, seen
            )
        if len(factors) < MIN_RANKABLE:
            return await self._skip(
                row,
                today,
                f"only {len(factors)} symbol(s) have usable live data; at least {MIN_RANKABLE} are needed to rank",
                excluded,
                seen,
            )

        weights_before = dict(state.weights)
        lessons: list[agent.Lesson] = []
        since = state.last_decision_on
        if strategy.signal == "model":
            picked = await self._model_targets(row, strategy, config, prices, excluded, today)
            if isinstance(picked, StepResult):
                return picked
            targets, top = picked
        else:
            if since is not None and since < today.isoformat():
                lessons = agent.learn(state, config, prices)
            decision = agent.decide(factors, state.weights, config)
            targets = decision.targets
            top = [(r.symbol, r.composite, r.rating) for r in decision.ranked[:10]]
            state.last_scores = decision.scores
            state.last_prices = {s: prices[s] for s in decision.scores}
        book = self._book(row.cash, positions)
        trade_prices = {s: prices[s] for s in set(targets) | set(held)}
        fills = book.rebalance(
            trade_prices,
            targets,
            model,
            cash_buffer=config.cash_buffer,
            min_trade_value=config.min_trade_value,
        )
        state.last_decision_on = today.isoformat()
        used = sorted(set(factors) | set(held))
        status = DataStatus.worst([statuses[s] for s in used])
        equity = book.equity(prices)

        async with self._db.session() as s:
            row = await repo.get_sandbox_account(s, account_id)
            row.cash = book.cash
            row.state = state.to_json()
            row.updated_at = now
            await repo.replace_sandbox_positions(
                s, account_id, {sym: (h.quantity, h.avg_cost) for sym, h in book.positions.items()}
            )
            trade_rows = self._trade_rows(account_id, fills, now, statuses, "agent", "daily rebalance")
            s.add_all(trade_rows)
            s.add(
                SandboxEquityRow(
                    account_id=account_id,
                    recorded_at=now,
                    equity=equity,
                    cash=book.cash,
                    benchmark_price=bench_price,
                    data_status=status.value,
                )
            )
            if lessons:
                await repo.add_sandbox_journal(
                    s,
                    account_id,
                    "lesson",
                    _lesson_summary(lessons, since or "previous"),
                    {
                        "since": since,
                        "ic": {lesson.factor: round(lesson.ic, 4) for lesson in lessons},
                        "observations": {lesson.factor: lesson.observations for lesson in lessons},
                        "weights_before": _round_weights(weights_before),
                        "weights_after": _round_weights(state.weights),
                    },
                    now,
                )
            await repo.add_sandbox_journal(
                s,
                account_id,
                "decision",
                _decision_summary(fills, targets),
                {
                    "targets": targets,
                    "signal": strategy.signal,
                    "top": [
                        {"symbol": sym, "composite": round(score, 4), "rating": rating}
                        for sym, score, rating in top
                    ],
                    "trades": len(fills),
                    "equity": round(equity, 2),
                    "data_status": status.value,
                    "forced": force,
                    "excluded": excluded,
                },
                now,
            )
            await s.flush()
            trades_out = [_trade_out(t) for t in trade_rows]
        self._retry_at.pop(account_id, None)
        return StepResult(
            account_id=account_id,
            executed=True,
            trading_day=today,
            as_of=now,
            trades=trades_out,
            lessons=[
                LessonOut(
                    factor=lesson.factor,
                    label=screener.LABELS[lesson.factor],
                    ic=lesson.ic,
                    observations=lesson.observations,
                    weight_before=lesson.weight_before,
                    weight_after=lesson.weight_after,
                )
                for lesson in lessons
            ],
            weights_before=_round_weights(weights_before),
            weights_after=_round_weights(state.weights),
            targets=targets,
            candidates=[
                Candidate(
                    symbol=sym, composite=round(score, 4), rating=rating, target_weight=targets.get(sym, 0.0)
                )
                for sym, score, rating in top
            ],
            excluded=excluded,
            equity=equity,
            cash=book.cash,
            data_status=status,
        )

    async def _model_targets(
        self,
        row: SandboxAccountRow,
        strategy: SandboxStrategy,
        config: agent.AgentConfig,
        prices: dict[str, float],
        excluded: dict[str, str],
        today: date,
    ) -> tuple[dict[str, float], list[tuple[str, float, int]]] | StepResult:
        """Targets from the walk-forward stock model: the top ``k`` names it ranks above average."""
        if self.model is None:
            return await self._skip(row, today, "the stock model is not available", excluded, DataStatus.LIVE)
        try:
            live, report = await self.model.live_scores(strategy.universe)
        except DomainError as exc:
            return await self._skip(row, today, f"stock model unavailable: {exc}", excluded, DataStatus.LIVE)
        if not self._usable(report.data_status, row.allow_synthetic):
            return await self._skip(
                row, today, "the stock model only has synthetic prices", excluded, report.data_status
            )
        ranked = sorted((s for s in live if s in prices), key=lambda s: (-live[s].z, s))
        chosen = [s for s in ranked if live[s].z > 0][: config.top_k]
        each = min(config.max_position, 1.0 / config.top_k)
        return dict.fromkeys(chosen, each), [(s, live[s].z, live[s].rating) for s in ranked[:10]]

    async def _skip(
        self, row: SandboxAccountRow, today: date, reason: str, excluded: dict[str, str], status: DataStatus
    ) -> StepResult:
        """Record (once per day and reason) why the agent sat out, and back off the scheduler."""
        now = self._clock.now()
        self._retry_at[row.id] = now + RETRY_AFTER
        if self._skip_logged.get(row.id) != (today, reason):
            self._skip_logged[row.id] = (today, reason)
            async with self._db.session() as s:
                await repo.add_sandbox_journal(
                    s, row.id, "skip", f"Sat out: {reason}.", {"excluded": excluded}, now
                )
        state = agent.LearningState.from_json(row.state)
        return StepResult(
            account_id=row.id,
            executed=False,
            skipped_reason=reason,
            trading_day=today,
            as_of=now,
            weights_before=_round_weights(state.weights),
            weights_after=_round_weights(state.weights),
            excluded=excluded,
            data_status=status,
        )

    async def mark(self, account_id: int) -> EquityPoint | None:
        """Record an equity snapshot at current prices; ``None`` if a holding has no usable price."""
        async with self._lock(account_id):
            async with self._db.session() as s:
                row = await repo.get_sandbox_account(s, account_id)
                positions = await repo.sandbox_positions(s, account_id)
            quotes = await self._market.quotes([p.symbol for p in positions])
            if any(not self._usable(q.status, row.allow_synthetic) for q in quotes.values()):
                return None
            bench_price, _ = await self._benchmark_price(row.allow_synthetic)
            now = self._clock.now()
            statuses = [q.status for q in quotes.values()]
            snapshot = SandboxEquityRow(
                account_id=account_id,
                recorded_at=now,
                equity=row.cash + sum(p.quantity * quotes[p.symbol].value.price for p in positions),
                cash=row.cash,
                benchmark_price=bench_price,
                data_status=(DataStatus.worst(statuses) if statuses else DataStatus.LIVE).value,
            )
            async with self._db.session() as s:
                await repo.add_row(s, snapshot)
            return _equity_point(snapshot)

    # ------------------------------------------------------------------ scheduler
    async def run_scheduled(self) -> str:
        """Called every minute by the poller: trade auto-trading agents once per trading day after
        ``sandbox_trade_time`` and mark every account to market after ``sandbox_mark_time``."""
        s = self._settings
        if not s.sandbox_scheduler_enabled:
            return "disabled"
        now, local = self._local_now()
        today = local.date()
        if not is_trading_day(today):
            return "market closed today"
        trade_at, mark_at = time.fromisoformat(s.sandbox_trade_time), time.fromisoformat(s.sandbox_mark_time)
        async with self._db.session() as session:
            accounts = await repo.list_sandbox_accounts(session)
        report: list[str] = []
        for row in accounts:
            retry = self._retry_at.get(row.id)
            if retry is not None and now < retry:
                continue
            try:
                last = (row.state or {}).get("last_decision_on")
                if (
                    row.mode == "agent"
                    and row.auto_trade
                    and local.time() >= trade_at
                    and last != today.isoformat()
                ):
                    result = await self.step(row.id)
                    report.append(f"#{row.id} {'traded' if result.executed else 'skipped'}")
                if local.time() >= mark_at and not await self._marked_today(row.id, today, mark_at):
                    point = await self.mark(row.id)
                    if point is None:
                        self._retry_at[row.id] = now + RETRY_AFTER
                        report.append(f"#{row.id} mark deferred (no live prices)")
                    else:
                        self._marked_on[row.id] = today
                        report.append(f"#{row.id} marked")
            except Exception as exc:  # one broken account must not stop the others
                self._retry_at[row.id] = now + RETRY_AFTER
                logger.warning("sandbox account %s scheduled run failed: %s", row.id, exc)
                report.append(f"#{row.id} error: {type(exc).__name__}")
        return ", ".join(report) or f"idle ({len(accounts)} account(s))"

    async def _marked_today(self, account_id: int, today: date, mark_at: time) -> bool:
        if self._marked_on.get(account_id) == today:
            return True
        async with self._db.session() as s:
            latest = await repo.latest_sandbox_equity(s, account_id)
        if latest is None:
            return False
        local = latest.recorded_at.astimezone(NEW_YORK)
        if local.date() == today and local.time() >= mark_at:
            self._marked_on[account_id] = today
            return True
        return False

    # ------------------------------------------------------------------ walk-forward training
    async def train(self, account_id: int, req: TrainRequest) -> CompositeEnvelope[TrainReport]:
        """Replay history day by day (no look-ahead) so the agent can learn before risking paper money."""
        async with self._db.session() as s:
            row = await repo.get_sandbox_account(s, account_id)
        strategy = SandboxStrategy.model_validate(row.strategy)
        config, model = _config(strategy), _execution(strategy)
        universe = self.universe(strategy)
        bench = self._settings.benchmark_symbol
        symbols = list(dict.fromkeys([*universe, bench]))
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def history(symbol: str) -> Resolved[PriceHistory]:
            async with sem:
                return await self._market.history(symbol, "1d", req.lookback_days)

        histories, curve_r = await asyncio.gather(
            asyncio.gather(*(history(x) for x in symbols)), self._rates.curve()
        )
        hist = dict(zip(symbols, histories, strict=True))
        warnings: list[str] = []
        skipped: dict[str, str] = {}

        real = [x for x in universe if hist[x].status is not DataStatus.SYNTHETIC]
        bench_real = hist[bench].status is not DataStatus.SYNTHETIC
        if not row.allow_synthetic and bench_real and len(real) >= MIN_TRAIN_SYMBOLS:
            for x in universe:
                if x not in real:
                    skipped[x] = "no live price history"
            chosen = real
        else:
            chosen = list(universe)
        bench_days = len(hist[bench].value.bars)
        for x in chosen:
            n = len(hist[x].value.bars)
            if n < MIN_COVERAGE * bench_days:
                skipped[x] = f"only {n} of {bench_days} trading days of history"
        chosen = [x for x in chosen if x not in skipped]
        if len(chosen) < MIN_RANKABLE:
            raise DomainError(f"only {len(chosen)} symbol(s) have enough history to train on")
        used = list(dict.fromkeys([*chosen, bench]))
        frame = closes_frame({x: hist[x].value for x in used})
        need = TRAIN_WARMUP + req.rebalance_every + MIN_REPLAY_DAYS
        if len(frame) < need:
            raise DomainError(
                f"only {len(frame)} aligned trading days in the last {req.lookback_days} calendar days; need "
                f"{need} ({TRAIN_WARMUP} warm-up + replay) — increase lookback_days"
            )
        data_status = DataStatus.worst([hist[x].status for x in used])
        rf = RatesService.rate_from_curve(curve_r.value, 0.25).bey_rate
        result = await asyncio.to_thread(
            agent.walk_forward,
            frame[chosen],
            frame[bench],
            config,
            model,
            start_cash=row.starting_cash,
            rebalance_every=req.rebalance_every,
            warmup=TRAIN_WARMUP,
        )
        metrics = result.metrics(rf)
        mean_ic: dict[str, float] = {}
        for f in agent.FACTORS:
            values = [ics[f] for _, ics in result.ic_history if f in ics]
            if values:
                mean_ic[f] = round(sum(values) / len(values), 4)
        if data_status is DataStatus.SYNTHETIC:
            warnings.append("the replay used synthetic prices; the results illustrate the mechanics only")
        applied = req.apply and self._usable(data_status, row.allow_synthetic)
        if req.apply and not applied:
            warnings.append(
                "learned weights were NOT applied: they were learned from synthetic prices "
                "(enable allow_synthetic on the account to apply them anyway)"
            )
        learned = _round_weights(result.state.weights)
        start, end = result.dates[0], result.dates[-1]
        fees = sum(f.commission for _, f in result.fills)
        now = self._clock.now()
        async with self._lock(account_id), self._db.session() as s:
            current = await repo.get_sandbox_account(s, account_id)
            if applied:
                state = agent.LearningState.from_json(current.state)
                state.weights = dict(result.state.weights)
                state.ic_ema = dict(result.state.ic_ema)
                state.periods_learned = (
                    result.state.periods_learned
                )  # the weights now reflect this replay only
                current.state = state.to_json()
                current.updated_at = now
            strategy_m = metrics["strategy"]
            bench_m = metrics["benchmark"]
            await repo.add_sandbox_journal(
                s,
                account_id,
                "train",
                f"Walk-forward replay {start} → {end} ({result.decisions} decisions): strategy "
                f"{_pct(strategy_m['total_return'] or 0.0)} vs {bench} {_pct(bench_m['total_return'] or 0.0)}; "
                + ("learned weights applied." if applied else "learned weights not applied."),
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "rebalance_every": req.rebalance_every,
                    "learned_weights": learned,
                    "mean_ic": mean_ic,
                    "applied": applied,
                    "data_status": data_status.value,
                },
                now,
            )

        def bt(m: dict[str, float | None]) -> BacktestMetrics:
            return BacktestMetrics(
                total_return=m["total_return"] or 0.0,
                annual_return=_finite(m["annual_return"]),
                annual_volatility=_finite(m["annual_volatility"]),
                sharpe=_finite(m["sharpe"]),
                max_drawdown=_finite(m["max_drawdown"]),
            )

        report = TrainReport(
            account_id=account_id,
            start=start,
            end=end,
            trading_days=len(result.dates),
            decisions=result.decisions,
            rebalance_every=req.rebalance_every,
            symbols=chosen,
            skipped=skipped,
            benchmark=bench,
            risk_free_rate=rf,
            strategy=bt(metrics["strategy"]),
            benchmark_metrics=bt(metrics["benchmark"]),
            trades=len(result.fills),
            fees_paid=fees,
            turnover=result.turnover,
            equity_curve=[
                CurvePoint(date=d, strategy=round(e, 2), benchmark=round(b, 2))
                for d, e, b in zip(result.dates, result.equity, result.benchmark, strict=True)
            ],
            weights_history=[
                WeightSnapshot(date=d, weights=_round_weights(w)) for d, w in result.weights_history
            ],
            mean_ic=mean_ic,
            prior_weights=_round_weights(agent.normalise(agent.PRIOR)),
            learned_weights=learned,
            applied=applied,
            data_status=data_status,
            warnings=warnings,
        )
        sources = {f"history:{x}": hist[x].provenance for x in used}
        sources["yield_curve"] = curve_r.provenance
        return CompositeEnvelope(data=report, meta=CompositeMeta.from_sources(sources, now))
