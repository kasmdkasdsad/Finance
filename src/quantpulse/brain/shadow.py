"""The strategy the Brain replaced, kept running as a **shadow** — for the comparison, never for trading.

While the Brain owns the Alpaca paper account the strategy sends nothing. To compare the two honestly the
supervisor runs the strategy's own plan (:meth:`~quantpulse.services.trading.TradingService.shadow_plan`:
the same data, regime, signals and portfolio construction) against a hypothetical portfolio of its own,
starting from the account's equity on the day the shadow started, all in cash:

* every proposed trade goes through the same :class:`~quantpulse.services.trading_risk.RiskBook` limits
  (live data, quote age, spread, size, exposure, cash reserve, daily loss) as a real order — sells first,
  then buys against the cash after them;
* approved trades are filled like the Brain's paper book: the quote plus half the believed spread plus
  ``QP_BRAIN_BOOK_SLIPPAGE_BPS``, less ``QP_BRAIN_BOOK_COST_BPS`` in fees;
* it runs on the strategy's own schedule (``QP_TRADING_REBALANCE_INTERVAL_MINUTES``) while the market is open,
  and is marked at each close (recorded with the day in ``brain_sessions``).

Its fills are modelled, not real, while the Brain's are Alpaca's: the evaluation says so.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.db.session import Database
from quantpulse.domain.trading_portfolio import Holding, PositionMemory
from quantpulse.providers.alpaca_trading import BrokerAccount, BrokerPosition
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_risk import OrderIntent, RiskBook, RiskLimits

from .book import fill_price
from .store import BrainStore

logger = logging.getLogger(__name__)
STATE_KEY = "strategy_shadow"
MAX_FILLS_KEPT = 200


def _positions(state: dict[str, Any]) -> dict[str, BrokerPosition]:
    out: dict[str, BrokerPosition] = {}
    for sym, p in (state.get("positions") or {}).items():
        price, qty, avg = float(p["last_price"]), float(p["qty"]), float(p["avg"])
        out[sym] = BrokerPosition(
            symbol=sym, qty=qty, qty_available=qty, side="long", avg_entry_price=avg, current_price=price,
            market_value=qty * price, cost_basis=qty * avg, unrealized_pl=qty * (price - avg),
            unrealized_plpc=price / avg - 1 if avg > 0 else 0.0, unrealized_intraday_pl=0.0, lastday_price=price,
        )  # fmt: skip
    return out


def _account(state: dict[str, Any], positions: dict[str, BrokerPosition]) -> BrokerAccount:
    invested = sum(p.market_value for p in positions.values())
    cash = float(state["cash"])
    equity = cash + invested
    return BrokerAccount(
        account_number="STRATEGY-SHADOW", status="HYPOTHETICAL", currency="USD", equity=equity,
        last_equity=float(state.get("last_equity") or equity), cash=cash, buying_power=max(cash, 0.0),
        long_market_value=invested, short_market_value=0.0, portfolio_value=equity, trading_blocked=False,
        account_blocked=False, trade_suspended_by_user=False, pattern_day_trader=False, daytrade_count=0,
        multiplier=1.0,
    )  # fmt: skip


def equity_of(state: dict[str, Any]) -> float:
    return float(state["cash"]) + sum(
        float(p["qty"]) * float(p["last_price"]) for p in (state.get("positions") or {}).values()
    )


class StrategyShadow:
    def __init__(self, settings: Settings, clock: Clock, db: Database, trading: TradingService) -> None:
        self._s = settings
        self._clock = clock
        self._store = BrainStore(db)
        self._trading = trading

    async def state(self) -> dict[str, Any] | None:
        return await self._store.get_state(STATE_KEY)

    async def _start(self) -> dict[str, Any]:
        account = await self._trading.broker.account()
        now = self._clock.now()
        state = {
            "started_at": now.isoformat(),
            "capital": account.equity,
            "cash": account.equity,
            "last_equity": account.equity,
            "positions": {},
            "memory": {},
            "last_traded": {},
            "trades": 0,
            "turnover": 0.0,
            "fills": [],
        }
        await self._store.set_state(STATE_KEY, state, now)
        return state

    async def step(self) -> dict[str, Any]:
        """One strategy cycle against the shadow portfolio (only while the market is open)."""
        s = self._s
        now = self._clock.now()
        state = await self.state() or await self._start()
        positions = _positions(state)
        holdings = {
            sym: Holding(sym, p.qty, p.avg_entry_price, p.current_price, p.market_value, p.unrealized_plpc)
            for sym, p in positions.items()
        }
        cooldown = timedelta(minutes=s.trading_cooldown_minutes)
        last = {
            sym: (datetime.fromisoformat(at), side)
            for sym, (at, side) in (state.get("last_traded") or {}).items()
            if now - datetime.fromisoformat(at) < cooldown
        }
        memory = {sym: PositionMemory(**m) for sym, m in (state.get("memory") or {}).items()}
        shadow = await self._trading.shadow_plan(holdings, equity_of(state), memory=memory, last_traded=last)
        if not shadow.market_open:
            return {"skipped": "the market is closed"}
        for sym, p in state["positions"].items():  # mark at this cycle's quotes
            q = shadow.quotes.get(sym)
            if q is not None:
                p["last_price"] = q.price
        limits = RiskLimits.from_settings(s)
        fills: list[dict[str, Any]] = []
        sells = [t for t in shadow.plan.trades if t.side == "sell"]
        buys = [t for t in shadow.plan.trades if t.side == "buy"]
        for batch in (sells, buys):
            positions = _positions(state)
            book = RiskBook(limits, _account(state, positions), positions, [], True, False, shadow.checks)
            for t in batch:
                q = shadow.quotes.get(t.symbol)
                price = q.price if q is not None else t.est_price
                intent = OrderIntent(
                    t.symbol, t.side, t.qty, price, t.kind, t.reason, t.closes_position, t.score
                )
                decision = book.evaluate(intent)
                if not decision.approved:
                    continue
                book.commit(intent)
                fills.append(
                    self._fill(
                        state, t.symbol, t.side, t.qty, price, t.kind, shadow.quality.get(t.symbol), now
                    )
                )
                if t.side == "buy" and t.kind == "entry":
                    c = shadow.candidates.get(t.symbol)
                    state["memory"][t.symbol] = {"entry_score": t.score, "entry_model_z": c.model_z if c else None,
                                                 "profit_taken_basis": None}  # fmt: skip
                elif t.closes_position:
                    state["memory"].pop(t.symbol, None)
        state["fills"] = [*state.get("fills", []), *fills][-MAX_FILLS_KEPT:]
        await self._store.set_state(STATE_KEY, state, now)
        return {"trades": len(fills), "equity": round(equity_of(state), 2), "regime": shadow.regime}

    def _fill(
        self,
        state: dict[str, Any],
        sym: str,
        side: str,
        qty: float,
        price: float,
        kind: str,
        quality: Any,
        now: datetime,
    ) -> dict[str, Any]:
        s = self._s
        px, how, _ = fill_price(side, price, quality, slippage_bps=s.brain_book_slippage_bps,
                                default_half_spread_bps=s.brain_book_default_half_spread_bps)  # fmt: skip
        cost = qty * px * s.brain_book_cost_bps / 10_000
        pos = state["positions"].get(sym)
        realized = None
        if side == "sell" and pos is not None:
            qty = min(qty, float(pos["qty"]))
            state["cash"] = float(state["cash"]) + qty * px - cost
            realized = (px - float(pos["avg"])) * qty - cost
            left = float(pos["qty"]) - qty
            if left <= 1e-9:
                state["positions"].pop(sym)
            else:
                pos["qty"] = left
        elif side == "buy":
            affordable = max(float(state["cash"]), 0.0) / (px * (1 + s.brain_book_cost_bps / 10_000))
            qty = min(qty, math.floor(affordable) if float(qty).is_integer() else affordable)
            if qty <= 0:
                return {"symbol": sym, "side": side, "qty": 0, "skipped": "not enough shadow cash"}
            cost = qty * px * s.brain_book_cost_bps / 10_000
            state["cash"] = float(state["cash"]) - qty * px - cost
            if pos is None:
                state["positions"][sym] = {"qty": qty, "avg": (qty * px + cost) / qty, "last_price": px,
                                           "opened_at": now.isoformat()}  # fmt: skip
            else:
                total = float(pos["qty"]) + qty
                pos["avg"] = (float(pos["qty"]) * float(pos["avg"]) + qty * px + cost) / total
                pos["qty"], pos["last_price"] = total, px
        state["trades"] = int(state.get("trades", 0)) + 1
        state["turnover"] = float(state.get("turnover", 0.0)) + qty * px
        state.setdefault("last_traded", {})[sym] = [now.isoformat(), side]
        return {"at": now.isoformat(), "symbol": sym, "side": side, "qty": qty, "price": round(px, 4),
                "kind": kind, "cost": round(cost, 2), "priced": how,
                "realized_pnl": None if realized is None else round(realized, 2)}  # fmt: skip

    async def mark(self, closes: dict[str, float]) -> dict[str, Any] | None:
        """The shadow at the close (last prices, or the day's closes when known); the day's equity is kept."""
        state = await self.state()
        if state is None:
            return None
        for sym, p in state["positions"].items():
            if sym in closes:
                p["last_price"] = closes[sym]
        equity = equity_of(state)
        out = {
            "equity": round(equity, 2),
            "cash": round(float(state["cash"]), 2),
            "positions": len(state["positions"]),
            "trades": int(state.get("trades", 0)),
            "turnover": round(float(state.get("turnover", 0.0)), 2),
            "day_return": round(equity / float(state["last_equity"]) - 1, 6)
            if state.get("last_equity")
            else None,
        }
        state["last_equity"] = equity
        await self._store.set_state(STATE_KEY, state, self._clock.now())
        return out
