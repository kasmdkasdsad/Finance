"""The trading day around the Brain's cycles, when it owns the Alpaca paper account.

* **Pre-market** (:meth:`SessionKeeper.premarket`, from 08:30 New York): verify the account (the SDK client
  points at the paper API, Alpaca's view of the account), reconcile orders and positions, read the calendar
  (Alpaca's clock, an early close), check that market data is available (the vendors' feeds and a live
  benchmark quote), and list what changed overnight (positions and equity against the last close) and any
  order still open before the bell. The result is kept with the day; a failed check is reported, and every
  Brain execution re-verifies and reconciles before it sends anything anyway.
* **During the session** the supervisor reconciles every few minutes and runs the cycles (order and fill
  changes arrive as events from the trading service's audit trail).
* **After the close** (:meth:`SessionKeeper.close`): reconcile, then record the day — equity, the day's
  return and the benchmark's, exposure, positions, the Brain's orders sent and filled and their notional,
  how many cycles ran and how many had new positions halted, and why. These rows (``brain_sessions``) are
  what the 60-session evaluation is computed from.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any

from sqlalchemy import select

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, regular_close
from quantpulse.db.models import BrainCycleRow, BrainSessionRow, BrokerOrderRow
from quantpulse.db.session import Database
from quantpulse.services.order_manager import BRAIN, STRATEGY
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_data import TradingDataLoader

from .evaluation import MarketPrices

logger = logging.getLogger(__name__)


def day_start(day: date) -> datetime:
    return datetime.combine(day, time(0, 0), NEW_YORK)


class SessionKeeper:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        db: Database,
        trading: TradingService,
        data: TradingDataLoader,
        prices: MarketPrices | None,
        feeds: Any = None,  # () -> list of vendor feed states (the market service's feed_status)
        shadow: Any = None,  # the strategy shadow (marked at the close)
    ) -> None:
        self._s = settings
        self._clock = clock
        self._db = db
        self._trading = trading
        self._data = data
        self._prices = prices
        self._feeds = feeds
        self._shadow = shadow

    async def _row(self, day: date) -> BrainSessionRow:
        async with self._db.session() as s:
            row = (await s.scalars(select(BrainSessionRow).where(BrainSessionRow.day == day))).first()
            if row is None:
                row = BrainSessionRow(day=day, owner=self._trading.owner, halts={}, premarket={}, close={},
                                      orders_sent=0, orders_filled=0, traded_notional=0.0, cycles=0,
                                      data_blocked_cycles=0, updated_at=self._clock.now())  # fmt: skip
                s.add(row)
                await s.flush()
            s.expunge(row)
        return row

    async def _previous(self, day: date) -> BrainSessionRow | None:
        async with self._db.session() as s:
            return (
                await s.scalars(
                    select(BrainSessionRow)
                    .where(BrainSessionRow.day < day)
                    .order_by(BrainSessionRow.day.desc())
                )
            ).first()

    async def _save(self, day: date, **fields: Any) -> None:
        await self._row(day)
        async with self._db.session() as s:
            row = (await s.scalars(select(BrainSessionRow).where(BrainSessionRow.day == day))).one()
            for k, v in fields.items():
                setattr(row, k, v)
            row.owner = self._trading.owner
            row.updated_at = self._clock.now()

    # ------------------------------------------------------------------ pre-market
    async def premarket(self) -> dict[str, Any]:
        now = self._clock.now()
        day = now.astimezone(NEW_YORK).date()
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool | None, detail: str, **extra: Any) -> None:
            checks.append({"name": name, "ok": ok, "detail": detail, **extra})

        trading = self._trading
        try:
            add("paper_endpoint", True, trading.verify_paper())
        except Exception as exc:  # fail closed: reported, and every execution re-verifies before sending
            add("paper_endpoint", False, f"{type(exc).__name__}: {exc}")
        account: Any = None
        try:
            account = await trading.broker.account()
            add(
                "account",
                not account.blocked,
                f"{account.status}; equity ${account.equity:,.2f}, cash ${account.cash:,.2f}, buying power "
                f"${account.buying_power:,.2f}" + ("; BLOCKED by Alpaca" if account.blocked else ""),
            )
        except Exception as exc:
            add("account", False, f"{type(exc).__name__}: {exc}")
        positions: dict[str, dict[str, float]] = {}
        try:
            rec = await trading.reconcile("brain pre-market")
            add(
                "reconciliation",
                True,
                f"{rec.positions} position(s), {rec.open_orders} open order(s), {rec.orders_updated} order "
                f"update(s), {rec.orders_added} found on Alpaca",
            )
            for p in await trading.broker.positions():
                positions[p.symbol] = {"qty": p.qty, "price": p.current_price, "value": p.market_value}
            open_orders = await trading.broker.open_orders()
            if open_orders:
                add(
                    "open_orders",
                    None,
                    f"{len(open_orders)} order(s) open before the bell: "
                    + ", ".join(
                        f"{o.side} {o.qty:g} {o.symbol} ({o.client_order_id})" for o in open_orders[:10]
                    ),
                )
        except Exception as exc:
            add("reconciliation", False, f"{type(exc).__name__}: {exc}")
        try:
            clock = await trading.broker.clock()
            early = regular_close(day) < time(16, 0) if is_trading_day(day) else False
            add(
                "calendar",
                True,
                ("a trading day" if is_trading_day(day) else "not a trading day")
                + (f", early close at {regular_close(day):%H:%M}" if early else "")
                + (f"; next open {clock.next_open.astimezone(NEW_YORK):%a %H:%M}" if clock.next_open else ""),
                trading_day=is_trading_day(day),
                early_close=early,
            )
        except Exception as exc:
            add("calendar", False, f"{type(exc).__name__}: {exc}")
        feeds = list(self._feeds()) if self._feeds is not None else []
        refused = [f for f in feeds if f.get("stock_feed_error")]
        bench = self._s.benchmark_symbol
        try:
            quotes = await self._data.live_quotes([bench], consolidated=False)
            q = quotes.get(bench)
            add(
                "market_data",
                q is not None and not refused,
                (f"{bench} quote from {q.provider} ({q.feed or 'unknown feed'}), last trade {q.age_seconds:,.0f}s old"
                 if q is not None else f"no live quote for {bench}")
                + ("; refused feeds: " + ", ".join(str(f.get("stock_feed")) for f in refused) if refused else ""),
            )  # fmt: skip
        except Exception as exc:
            add("market_data", False, f"{type(exc).__name__}: {exc}")
        prev = await self._previous(day)
        changes: list[str] = []
        if prev is not None and prev.close.get("positions") is not None:
            before = prev.close["positions"]
            for sym in sorted(set(before) | set(positions)):
                a, b = (before.get(sym) or {}).get("qty", 0.0), (positions.get(sym) or {}).get("qty", 0.0)
                if abs(a - b) > 1e-9:
                    changes.append(f"{sym} {a:g} → {b:g}")
            if account is not None and prev.equity_close:
                changes.append(f"equity {account.equity / prev.equity_close - 1:+.2%} since the last close")
        add("overnight", None, "; ".join(changes) or "no change since the last recorded close")
        report = {
            "at": now.isoformat(),
            "owner": trading.owner,
            "ok": all(c["ok"] is not False for c in checks),
            "checks": checks,
            "positions": positions,
        }
        await self._save(day, premarket=report)
        return {"ok": report["ok"], "failed": [c["name"] for c in checks if c["ok"] is False]}

    # ------------------------------------------------------------------ after the close
    async def close(self) -> dict[str, Any]:
        now = self._clock.now()
        day = now.astimezone(NEW_YORK).date()
        if not is_trading_day(day):
            return {"skipped": "not a trading day"}
        trading = self._trading
        rec = await trading.reconcile("brain close")
        account = await trading.broker.account()
        positions = await trading.broker.positions()
        start = day_start(day)
        who = BRAIN if trading.owner == "brain" else STRATEGY
        async with self._db.session() as s:
            orders = (
                await s.scalars(
                    select(BrokerOrderRow).where(
                        BrokerOrderRow.created_at >= start, BrokerOrderRow.strategy == who
                    )
                )
            ).all()
            cycles = (
                await s.scalars(
                    select(BrainCycleRow).where(
                        BrainCycleRow.started_at >= start, BrainCycleRow.status == "completed"
                    )
                )
            ).all()
        in_session = [c for c in cycles if (c.market or {}).get("open")]
        halts: dict[str, int] = {}
        for c in in_session:
            for code in (c.summary or {}).get("entry_halts") or []:
                halts[code] = halts.get(code, 0) + 1
        bench_close = bench_ret = None
        note = None
        shadow_state = await self._shadow.state() if self._shadow is not None else None
        held = list((shadow_state or {}).get("positions") or {})
        all_closes = (
            await self._prices.closes([self._s.benchmark_symbol, *held], day - timedelta(days=10))
            if self._prices is not None
            else {}
        )
        shadow = (
            await self._shadow.mark({s: c[day] for s, c in all_closes.items() if day in c})
            if shadow_state is not None
            else None
        )
        if self._prices is not None:
            closes = all_closes.get(self._s.benchmark_symbol) or {}
            days = sorted(d for d in closes if d <= day)
            if days and days[-1] == day and len(days) >= 2:
                bench_close = closes[day]
                bench_ret = round(bench_close / closes[days[-2]] - 1, 6)
            else:
                note = "the benchmark's close for today is not available yet: its return is left empty"
        fields = {
            "equity_open": account.last_equity,
            "equity_close": account.equity,
            "day_return": round(account.equity / account.last_equity - 1, 6)
            if account.last_equity > 0
            else None,
            "benchmark_close": bench_close,
            "benchmark_return": bench_ret,
            "exposure": round(account.long_market_value / account.equity, 5) if account.equity > 0 else None,
            "positions": len([p for p in positions if p.qty > 0]),
            "orders_sent": sum(1 for o in orders if o.alpaca_order_id),
            "orders_filled": sum(1 for o in orders if o.filled_quantity > 0),
            "traded_notional": round(
                sum(o.filled_quantity * (o.average_fill_price or 0.0) for o in orders), 2
            ),
            "cycles": len(in_session),
            "data_blocked_cycles": halts.get("data_quality", 0),
            "halts": halts,
            "close": {
                "at": now.isoformat(),
                "cash": account.cash,
                "reconciliation": {"open_orders": rec.open_orders, "updated": rec.orders_updated},
                "positions": {
                    p.symbol: {"qty": p.qty, "price": p.current_price, "value": p.market_value}
                    for p in positions
                },
                "note": note,
                "strategy_shadow": shadow,
            },
        }
        await self._save(day, **fields)
        return {k: fields[k] for k in ("day_return", "benchmark_return", "orders_sent", "cycles")}

    async def sessions(self, limit: int = 60) -> list[dict[str, Any]]:
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(BrainSessionRow).order_by(BrainSessionRow.day.desc()).limit(limit))
            ).all()
        return [session_view(r) for r in rows]


def session_view(r: BrainSessionRow) -> dict[str, Any]:
    return {
        "day": r.day.isoformat(),
        "owner": r.owner,
        "equity_open": r.equity_open,
        "equity_close": r.equity_close,
        "day_return": r.day_return,
        "benchmark_return": r.benchmark_return,
        "excess_return": round(r.day_return - r.benchmark_return, 6)
        if r.day_return is not None and r.benchmark_return is not None
        else None,
        "exposure": r.exposure,
        "positions": r.positions,
        "orders_sent": r.orders_sent,
        "orders_filled": r.orders_filled,
        "traded_notional": r.traded_notional,
        "cycles": r.cycles,
        "data_blocked_cycles": r.data_blocked_cycles,
        "halts": r.halts,
        "premarket": r.premarket,
        "close": r.close,
    }
