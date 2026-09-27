"""The Brain's paper book: a hypothetical portfolio the Brain manages — simulated, never an Alpaca account.

**Ownership.** There are two portfolios and each has exactly one owner:

* the **Alpaca paper account** is owned by the trading strategy (``services/trading.py``): its targets, its
  risk checks, its orders. The Brain only *reads* it, as context;
* the **Brain paper book** is owned by the Brain. The Brain's decisions — buy, increase, reduce, close,
  rebalance, de-risk — are made for this book, previewed by the same deterministic risk engine and limits
  as real orders (``RiskBook`` with ``RiskLimits`` from settings: position and order size, cash reserve,
  daily loss, kill switch, live data, spread), and only those it allows are simulated here.

Nothing here can reach a broker. Whether the Brain should ever manage the Alpaca paper account is a
person's decision (see the README's *Portfolio ownership*); until then this book is how the Brain is
judged as a portfolio manager.

**Fills.** An allowed trade is filled at the price it was proposed at (the live last trade), plus half the
spread the quote validation believed (the consolidated SIP quote when available; otherwise the single
venue's, or ``QP_BRAIN_BOOK_DEFAULT_HALF_SPREAD_BPS`` when no spread could be believed — labelled as an
assumption), plus ``QP_BRAIN_BOOK_SLIPPAGE_BPS``, against the trade's direction; fees are
``QP_BRAIN_BOOK_COST_BPS`` of notional. Sells go first; a buy never uses more than the book's cash (no
margin). Each fill records the proposed price, the fill price, the slippage and the costs.

**Positions** carry their entry, stop (the risk limits' ``max_position_loss_pct`` below the average cost),
the invalidation and thesis they were bought on, the expected return (once the consensus is calibrated),
the horizon and the date to review them. The book is marked to market after every cycle; the last mark of
a day is that day's close.

**Performance** comes only from those marks and fills: return against the benchmark over the same days,
volatility, Sharpe, Sortino, information ratio, beta, maximum and current drawdown, turnover, slippage,
costs, and closed trades' hit rate and holding time — flagged *too short to judge* below 20 sessions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
from sqlalchemy import delete, select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, sessions_after
from quantpulse.db.models import BrainBookEquityRow, BrainBookPositionRow, BrainBookTradeRow, BrainStateRow
from quantpulse.db.session import Database
from quantpulse.providers.alpaca_trading import BrokerAccount, BrokerPosition
from quantpulse.quant.risk import max_drawdown, sharpe_ratio, sortino_ratio
from quantpulse.services.trading_data import QuoteQuality

from .consensus import Consensus, ReliabilityBook
from .context import BrainContext, PortfolioState
from .decisions import Proposal
from .types import MARKET, SELLING

STATE_KEY = "book"
ACCOUNT_LABEL = "BRAIN-BOOK"  # never an Alpaca account number
EXECUTABLE = ("recommended", "dry_run_approved")
MIN_DAYS = 20
ANNUAL = 252


@dataclass
class BookState:
    capital: float
    cash: float
    created_at: datetime
    positions: dict[str, BrainBookPositionRow] = field(default_factory=dict)
    last_equity: float = 0.0  # the book's equity at the previous day's last mark


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    action: str
    qty: float
    proposed_price: float
    fill_price: float
    notional: float
    slippage_bps: float
    cost: float
    price_source: str
    realized_pnl: float | None = None
    holding_days: float | None = None
    decision_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "action": self.action,
            "qty": round(self.qty, 6),
            "proposed_price": round(self.proposed_price, 4),
            "fill_price": round(self.fill_price, 4),
            "notional": round(self.notional, 2),
            "slippage_bps": round(self.slippage_bps, 2),
            "cost": round(self.cost, 2),
            "price_source": self.price_source,
            "realized_pnl": None if self.realized_pnl is None else round(self.realized_pnl, 2),
            "holding_days": self.holding_days,
            "decision_id": self.decision_id,
        }


def fill_price(
    side: str,
    proposed: float,
    quality: QuoteQuality | None,
    *,
    slippage_bps: float,
    default_half_spread_bps: float,
) -> tuple[float, str, float]:
    """(fill price, how it was priced, half-spread in bps) for a marketable simulated order."""
    if quality is not None and quality.spread_bps is not None:
        half, basis = quality.spread_bps / 2, f"half the {quality.spread_source} spread"
    else:
        half, basis = default_half_spread_bps, "an assumed half-spread (no believable bid/ask)"
    sign = 1 if side == "buy" else -1
    price = proposed * (1 + sign * (half + slippage_bps) / 10_000)
    return price, f"proposed price ± {half:.1f}bp ({basis}) ± {slippage_bps:.1f}bp slippage", half


class PaperBook:
    def __init__(self, settings: Settings, db: Database, clock: Clock) -> None:
        self._s = settings
        self._db = db
        self._clock = clock

    # ------------------------------------------------------------------ state
    async def load(self) -> BookState:
        now = self._clock.now()
        today = now.astimezone(NEW_YORK).date()
        async with self._db.session() as s:
            row = await s.get(BrainStateRow, STATE_KEY)
            if row is None:
                capital = float(self._s.brain_book_capital)
                row = BrainStateRow(
                    key=STATE_KEY,
                    value={"capital": capital, "cash": capital, "created_at": now.isoformat()},
                    updated_at=now,
                )
                s.add(row)
            value = dict(row.value)
            positions = {p.symbol: p for p in (await s.scalars(select(BrainBookPositionRow))).all()}
            previous = (
                await s.scalars(
                    select(BrainBookEquityRow)
                    .where(BrainBookEquityRow.day < today)
                    .order_by(BrainBookEquityRow.at.desc())
                    .limit(1)
                )
            ).first()
        capital = float(value["capital"])
        return BookState(
            capital=capital,
            cash=float(value["cash"]),
            created_at=datetime.fromisoformat(value["created_at"]),
            positions=positions,
            last_equity=previous.equity if previous is not None else capital,
        )

    @staticmethod
    def portfolio(state: BookState, price_of: Callable[[str], float | None]) -> PortfolioState:
        """The book as the portfolio the Brain's agents, decisions and the risk engine work on."""
        positions: dict[str, BrokerPosition] = {}
        for sym, p in state.positions.items():
            price = price_of(sym) or p.last_price
            value = p.qty * price
            cost = p.qty * p.avg_cost
            positions[sym] = BrokerPosition(
                symbol=sym,
                qty=p.qty,
                qty_available=p.qty,
                side="long",
                avg_entry_price=p.avg_cost,
                current_price=price,
                market_value=value,
                cost_basis=cost,
                unrealized_pl=value - cost,
                unrealized_plpc=value / cost - 1 if cost > 0 else 0.0,
                unrealized_intraday_pl=0.0,
                lastday_price=p.last_price,
            )
        invested = sum(p.market_value for p in positions.values())
        equity = state.cash + invested
        account = BrokerAccount(
            account_number=ACCOUNT_LABEL,
            status="HYPOTHETICAL",
            currency="USD",
            equity=equity,
            last_equity=state.last_equity,
            cash=state.cash,
            buying_power=max(state.cash, 0.0),  # no margin in the book
            long_market_value=invested,
            short_market_value=0.0,
            portfolio_value=equity,
            trading_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
            pattern_day_trader=False,
            daytrade_count=0,
            multiplier=1.0,
        )
        return PortfolioState(available=True, account=account, positions=positions, open_orders=[])

    # ------------------------------------------------------------------ simulation
    async def execute(
        self,
        ctx: BrainContext,
        proposals: Sequence[Proposal],
        *,
        cycle_id: int,
        decision_ids: dict[str, int],
        expected: dict[str, float | None],
    ) -> list[Fill]:
        """Simulate every trade the risk engine allowed (sells first). Returns the fills."""
        todo = [p for p in proposals if p.is_trade and p.status in EXECUTABLE and p.est_price]
        if not todo:
            return []
        now = self._clock.now()
        fills: list[Fill] = []
        async with self._db.session() as s:
            row = await s.get(BrainStateRow, STATE_KEY)
            if row is None:  # load() creates it; a cycle always loads first
                return []
            value = dict(row.value)
            cash = float(value["cash"])
            positions = {p.symbol: p for p in (await s.scalars(select(BrainBookPositionRow))).all()}
            for p in sorted(todo, key=lambda p: 0 if p.action in SELLING else 1):
                side = "sell" if p.action in SELLING else "buy"
                price, source, _ = fill_price(
                    side,
                    float(p.est_price or 0),
                    ctx.quality.get(p.subject),
                    slippage_bps=self._s.brain_book_slippage_bps,
                    default_half_spread_bps=self._s.brain_book_default_half_spread_bps,
                )
                qty = float(p.quantity or 0)
                pos = positions.get(p.subject)
                if side == "sell":
                    if pos is None or pos.qty <= 0:
                        continue
                    qty = min(qty, pos.qty)
                    notional = qty * price
                    cost = notional * self._s.brain_book_cost_bps / 10_000
                    realized = (price - pos.avg_cost) * qty - cost
                    held = (now - pos.opened_at).total_seconds() / 86_400
                    cash += notional - cost
                    pos.qty -= qty
                    if pos.qty <= 1e-9:
                        await s.delete(pos)
                        positions.pop(p.subject)
                    else:
                        pos.updated_at = now
                    fill = self._fill(
                        p, side, qty, price, cost, source, decision_ids, realized, round(held, 2)
                    )
                else:
                    rate = self._s.brain_book_cost_bps / 10_000
                    affordable = max(cash, 0.0) / (price * (1 + rate))  # never beyond the book's cash
                    qty = min(qty, float(math.floor(affordable)) if qty.is_integer() else affordable)
                    if qty <= 0:
                        continue
                    notional = qty * price
                    cost = notional * rate
                    cash -= notional + cost
                    stop = price * (1 - ctx.limits.max_position_loss_pct)
                    horizon = int((ctx.working.facts.get("horizons") or {}).get(p.subject, 5))
                    thesis = "; ".join(p.reasons)[:500]
                    invalidation = self._invalidation(ctx, p.subject)
                    if pos is None:
                        pos = BrainBookPositionRow(
                            symbol=p.subject,
                            qty=qty,
                            avg_cost=(notional + cost) / qty,
                            last_price=price,
                            opened_at=now,
                            updated_at=now,
                            stop_price=round(stop, 4),
                            invalidation=invalidation,
                            thesis=thesis,
                            expected_return=expected.get(p.subject),
                            horizon_days=horizon,
                            review_after=sessions_after(now.astimezone(NEW_YORK).date(), max(horizon, 1)),
                            entry_decision_id=decision_ids.get(p.subject),
                        )
                        s.add(pos)
                        positions[p.subject] = pos
                    else:
                        total = pos.qty + qty
                        pos.avg_cost = (pos.qty * pos.avg_cost + notional + cost) / total
                        pos.qty = total
                        pos.stop_price = round(pos.avg_cost * (1 - ctx.limits.max_position_loss_pct), 4)
                        pos.updated_at = now
                    fill = self._fill(p, side, qty, price, cost, source, decision_ids)
                fills.append(fill)
                s.add(
                    BrainBookTradeRow(
                        cycle_id=cycle_id,
                        decision_id=fill.decision_id,
                        executed_at=now,
                        symbol=fill.symbol,
                        side=fill.side,
                        action=fill.action,
                        qty=fill.qty,
                        proposed_price=fill.proposed_price,
                        fill_price=fill.fill_price,
                        notional=fill.notional,
                        slippage_bps=fill.slippage_bps,
                        cost=fill.cost,
                        price_source=fill.price_source[:64],
                        realized_pnl=fill.realized_pnl,
                        holding_days=fill.holding_days,
                        reason="; ".join(p.reasons)[:500],
                    )
                )
            row.value, row.updated_at = {**value, "cash": cash}, now
        return fills

    @staticmethod
    def _fill(
        p: Proposal,
        side: str,
        qty: float,
        price: float,
        cost: float,
        source: str,
        decision_ids: dict[str, int],
        realized: float | None = None,
        held: float | None = None,
    ) -> Fill:
        proposed = float(p.est_price or 0)
        worse = (price - proposed) if side == "buy" else (proposed - price)  # positive = paid more / got less
        return Fill(
            symbol=p.subject,
            side=side,
            action=p.action.value,
            qty=qty,
            proposed_price=proposed,
            fill_price=price,
            notional=qty * price,
            slippage_bps=worse / proposed * 10_000 if proposed else 0.0,
            cost=cost,
            price_source=source,
            realized_pnl=realized,
            holding_days=held,
            decision_id=decision_ids.get(p.subject),
        )

    @staticmethod
    def _invalidation(ctx: BrainContext, symbol: str) -> str | None:
        levels = [
            o.invalidation for o in ctx.working.opinions.get(symbol, []) if o.invalidation and o.directional
        ]
        return "; ".join(levels[:3])[:500] if levels else None

    async def mark(self, ctx: BrainContext, cycle_id: int | None) -> dict[str, Any]:
        """Mark the book to market after a cycle and record the point on its equity curve."""
        now = self._clock.now()
        state = await self.load()
        invested = 0.0
        async with self._db.session() as s:
            for sym in list(state.positions):
                p = await s.get(BrainBookPositionRow, state.positions[sym].id)
                if p is None:
                    continue
                price = ctx.price(sym) or p.last_price
                p.last_price = price
                invested += p.qty * price
            equity = state.cash + invested
            s.add(
                BrainBookEquityRow(
                    at=now,
                    day=now.astimezone(NEW_YORK).date(),
                    cycle_id=cycle_id,
                    equity=equity,
                    cash=state.cash,
                    invested=invested,
                    positions=len(state.positions),
                    benchmark_price=ctx.price(ctx.benchmark_symbol),
                )
            )
        return {"equity": round(equity, 2), "cash": round(state.cash, 2), "invested": round(invested, 2)}

    # ------------------------------------------------------------------ reporting
    async def view(self, trades_limit: int = 100) -> dict[str, Any]:
        state = await self.load()
        async with self._db.session() as s:
            marks = (await s.scalars(select(BrainBookEquityRow).order_by(BrainBookEquityRow.at))).all()
            trades = (await s.scalars(select(BrainBookTradeRow).order_by(BrainBookTradeRow.id.desc()))).all()
        pf = self.portfolio(state, lambda _s: None)
        positions = [
            {
                "symbol": sym,
                "qty": round(p.qty, 6),
                "avg_cost": round(p.avg_cost, 4),
                "last_price": round(p.last_price, 4),
                "market_value": round(pf.positions[sym].market_value, 2),
                "unrealized_pl": round(pf.positions[sym].unrealized_pl, 2),
                "unrealized_pct": round(pf.positions[sym].unrealized_plpc, 4),
                "weight": round(pf.weight(sym), 4),
                "stop_price": p.stop_price,
                "invalidation": p.invalidation,
                "thesis": p.thesis,
                "expected_return": p.expected_return,
                "horizon_days": p.horizon_days,
                "review_after": p.review_after.isoformat() if p.review_after else None,
                "opened_at": p.opened_at.isoformat(),
            }
            for sym, p in state.positions.items()
        ]
        daily = _daily(marks)
        return {
            "owner": "the Brain (hypothetical: simulated fills, never sent to a broker)",
            "capital": state.capital,
            "cash": round(state.cash, 2),
            "equity": round(pf.equity, 2),
            "created_at": state.created_at.isoformat(),
            "positions": positions,
            "trades": [_trade(t) for t in trades[:trades_limit]],
            "equity_curve": [
                {"day": d.isoformat(), "equity": round(e, 2), "benchmark": b} for d, e, b in daily
            ],
            "performance": metrics(daily, list(trades), state.capital),
            "assumptions": {
                "slippage_bps": self._s.brain_book_slippage_bps,
                "cost_bps": self._s.brain_book_cost_bps,
                "default_half_spread_bps": self._s.brain_book_default_half_spread_bps,
            },
        }

    async def reset(self) -> None:
        """Start the book again from ``QP_BRAIN_BOOK_CAPITAL`` (its history is deleted)."""
        async with self._db.session() as s:
            await s.execute(delete(BrainBookPositionRow))
            await s.execute(delete(BrainBookTradeRow))
            await s.execute(delete(BrainBookEquityRow))
            row = await s.get(BrainStateRow, STATE_KEY)
            if row is not None:
                await s.delete(row)


def _trade(t: BrainBookTradeRow) -> dict[str, Any]:
    return {
        "executed_at": t.executed_at.isoformat(),
        "cycle_id": t.cycle_id,
        "decision_id": t.decision_id,
        "symbol": t.symbol,
        "side": t.side,
        "action": t.action,
        "qty": round(t.qty, 6),
        "proposed_price": round(t.proposed_price, 4),
        "fill_price": round(t.fill_price, 4),
        "notional": round(t.notional, 2),
        "slippage_bps": round(t.slippage_bps, 2),
        "cost": round(t.cost, 2),
        "price_source": t.price_source,
        "realized_pnl": None if t.realized_pnl is None else round(t.realized_pnl, 2),
        "holding_days": t.holding_days,
        "reason": t.reason,
    }


def _daily(marks: Sequence[BrainBookEquityRow]) -> list[tuple[date, float, float | None]]:
    """The last mark of each day: (day, equity, benchmark price)."""
    out: dict[date, tuple[float, float | None]] = {}
    for m in marks:
        out[m.day] = (m.equity, m.benchmark_price)
    return [(d, e, b) for d, (e, b) in sorted(out.items())]


def metrics(
    daily: Sequence[tuple[date, float, float | None]], trades: Sequence[BrainBookTradeRow], capital: float
) -> dict[str, Any]:
    """Performance from the book's own marks and fills (see the module docstring)."""
    out: dict[str, Any] = {"sessions": len(daily), "too_short_to_judge": len(daily) < MIN_DAYS}
    traded = sum(t.notional for t in trades)
    out["trades"] = len(trades)
    out["costs"] = round(sum(t.cost for t in trades), 2)
    out["slippage"] = round(sum(t.notional * t.slippage_bps / 10_000 for t in trades), 2)
    out["slippage_bps"] = (
        round(sum(t.notional * t.slippage_bps for t in trades) / traded, 2) if traded else None
    )
    closed = [t for t in trades if t.realized_pnl is not None]
    out["closed_trades"] = len(closed)
    out["closed_hit_rate"] = (
        round(sum(1 for t in closed if (t.realized_pnl or 0) > 0) / len(closed), 3) if closed else None
    )
    out["realized_pnl"] = round(sum(t.realized_pnl or 0.0 for t in closed), 2)
    held = [t.holding_days for t in closed if t.holding_days is not None]
    out["avg_holding_days"] = round(sum(held) / len(held), 2) if held else None
    if not daily:
        return out
    equity = np.array([e for _, e, _ in daily], dtype=float)
    out["total_return"] = round(float(equity[-1] / capital - 1), 5)
    peak = np.maximum.accumulate(np.concatenate([[capital], equity]))
    out["current_drawdown"] = round(float(equity[-1] / peak[-1] - 1), 5)
    out["turnover"] = round(traded / float(np.mean(np.concatenate([[capital], equity]))), 4)
    bench = [b for _, _, b in daily]
    if bench[0] and bench[-1]:
        out["benchmark_return"] = round(float(bench[-1] / bench[0] - 1), 5)
        out["excess_return"] = round(out["total_return"] - out["benchmark_return"], 5)
    if len(daily) < 2:
        return out
    r = np.diff(np.concatenate([[capital], equity])) / np.concatenate([[capital], equity])[:-1]
    out["volatility"] = round(float(r.std(ddof=1)) * math.sqrt(ANNUAL), 5) if r.size > 1 else None
    out["sharpe"] = _r(sharpe_ratio(r))
    out["sortino"] = _r(sortino_ratio(r))
    out["max_drawdown"] = round(float(max_drawdown(r)), 5)
    out["turnover_annual"] = round(out["turnover"] * ANNUAL / len(daily), 3)
    if all(b for b in bench):
        b = np.diff(np.array(bench, dtype=float)) / np.array(bench[:-1], dtype=float)
        mine = r[1:] if r.size == b.size + 1 else r[-b.size :]
        if b.size > 2 and np.var(b, ddof=1) > 0:
            ex = mine - b
            te = float(ex.std(ddof=1)) * math.sqrt(ANNUAL)
            out["information_ratio"] = round(float(ex.mean()) * ANNUAL / te, 4) if te > 0 else None
            out["beta"] = round(float(np.cov(mine, b, ddof=1)[0, 1] / np.var(b, ddof=1)), 4)
    return out


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def expected_by_subject(
    consensus: dict[str, Consensus], reliability: ReliabilityBook, version: str
) -> dict[str, float | None]:
    """The consensus's calibrated expected return per subject (``None`` until calibrated)."""
    return {
        subject: reliability.expected_return("consensus", version, c.confidence, 1 if c.score > 0 else -1)
        for subject, c in consensus.items()
        if subject != MARKET and c.actionable_view
    }
