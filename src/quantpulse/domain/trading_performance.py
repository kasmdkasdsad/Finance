"""Paper-trading performance from recorded data only: equity snapshots read from Alpaca at each cycle and
the fills of QuantPulse's orders. Nothing is estimated or back-filled; a statistic without enough data is
``None`` with a note saying why.

* daily equity = the last recorded equity of each day; daily return = change against the previous day;
* Sharpe = mean / standard deviation of daily returns × √252 (no risk-free rate); Sortino uses the
  downside deviation; maximum drawdown is measured on the daily equity curve;
* round trips pair each sell with the earliest unmatched buys of the same symbol (FIFO); win rate,
  average winner/loser and profit factor use those trips' realised P/L;
* turnover = traded notional ÷ average equity over the period; exposure = long market value ÷ equity,
  averaged over cycles;
* attribution = realised P/L by symbol and by the kind of trade that closed it (stop-loss, take-profit,
  signal exit, …).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

MIN_DAYS_FOR_RATIOS = 5


@dataclass(frozen=True, slots=True)
class EquityPoint:
    at: datetime
    day: date
    equity: float
    long_market_value: float | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    at: datetime
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class RoundTrip:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    opened: datetime
    closed: datetime
    exit_kind: str | None

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def ret(self) -> float:
        return self.exit_price / self.entry_price - 1 if self.entry_price > 0 else 0.0


@dataclass
class Performance:
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
    daily: list[dict[str, object]] = field(default_factory=list)
    monthly: list[dict[str, object]] = field(default_factory=list)
    by_symbol: dict[str, float] = field(default_factory=dict)
    by_exit: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def daily_equity(points: Sequence[EquityPoint]) -> list[EquityPoint]:
    last: dict[date, EquityPoint] = {}
    for p in sorted(points, key=lambda p: p.at):
        last[p.day] = p
    return [last[d] for d in sorted(last)]


def round_trips(fills: Sequence[Fill]) -> list[RoundTrip]:
    lots: dict[str, deque[list[float | datetime]]] = defaultdict(deque)
    trips: list[RoundTrip] = []
    for f in sorted(fills, key=lambda f: f.at):
        if f.side == "buy":
            lots[f.symbol].append([f.qty, f.price, f.at])
            continue
        remaining = f.qty
        book = lots[f.symbol]
        while remaining > 1e-9 and book:
            lot = book[0]
            qty = min(remaining, float(lot[0]))  # type: ignore[arg-type]
            trips.append(
                RoundTrip(f.symbol, qty, float(lot[1]), f.price, lot[2], f.at, f.kind)  # type: ignore[arg-type]
            )
            lot[0] = float(lot[0]) - qty  # type: ignore[arg-type]
            remaining -= qty
            if float(lot[0]) <= 1e-9:  # type: ignore[arg-type]
                book.popleft()
    return trips


def summarize(points: Sequence[EquityPoint], fills: Sequence[Fill]) -> Performance:
    notes: list[str] = []
    days = daily_equity(points)
    equity = np.array([p.equity for p in days], dtype=float)
    rets = equity[1:] / equity[:-1] - 1 if len(equity) > 1 else np.array([])
    sharpe = sortino = None
    if len(rets) >= MIN_DAYS_FOR_RATIOS:
        sd = float(rets.std(ddof=1))
        sharpe = float(rets.mean() / sd * math.sqrt(252)) if sd > 0 else None
        downside = rets[rets < 0]
        dd = float(math.sqrt((downside**2).mean())) if len(downside) else 0.0
        sortino = float(rets.mean() / dd * math.sqrt(252)) if dd > 0 else None
    else:
        notes.append(
            f"Sharpe and Sortino need at least {MIN_DAYS_FOR_RATIOS} daily returns; {len(rets)} recorded so far."
        )
    max_dd = None
    if len(equity) >= 2:
        peaks = np.maximum.accumulate(equity)
        max_dd = float((equity / peaks - 1).min())
    daily: list[dict[str, object]] = [
        {
            "date": p.day,
            "equity": p.equity,
            "pl": (p.equity - days[i - 1].equity) if i else None,
            "return": (p.equity / days[i - 1].equity - 1) if i and days[i - 1].equity > 0 else None,
        }
        for i, p in enumerate(days)
    ]
    monthly: list[dict[str, object]] = []
    by_month: dict[str, list[EquityPoint]] = defaultdict(list)
    for p in days:
        by_month[p.day.strftime("%Y-%m")].append(p)
    previous: float | None = None
    for month in sorted(by_month):
        start = previous if previous is not None else by_month[month][0].equity
        end = by_month[month][-1].equity
        monthly.append({"month": month, "pl": end - start, "return": end / start - 1 if start > 0 else None})
        previous = end

    trips = round_trips(fills)
    winners = [t.pnl for t in trips if t.pnl > 0]
    losers = [t.pnl for t in trips if t.pnl < 0]
    by_symbol: dict[str, float] = defaultdict(float)
    by_exit: dict[str, float] = defaultdict(float)
    for t in trips:
        by_symbol[t.symbol] += t.pnl
        by_exit[t.exit_kind or "unknown"] += t.pnl
    if not trips:
        notes.append("No completed round trips yet: win rate and average winner/loser need closed positions.")
    traded = sum(f.qty * f.price for f in fills)
    avg_equity = float(np.mean([p.equity for p in points])) if points else 0.0
    exposures = [
        p.long_market_value / p.equity for p in points if p.long_market_value is not None and p.equity > 0
    ]
    return Performance(
        days=len(days),
        first_day=days[0].day if days else None,
        last_day=days[-1].day if days else None,
        start_equity=float(equity[0]) if len(equity) else None,
        end_equity=float(equity[-1]) if len(equity) else None,
        total_return=float(equity[-1] / equity[0] - 1) if len(equity) >= 2 and equity[0] > 0 else None,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        best_day=float(rets.max()) if len(rets) else None,
        worst_day=float(rets.min()) if len(rets) else None,
        round_trips=len(trips),
        win_rate=len(winners) / len(trips) if trips else None,
        avg_winner=float(np.mean(winners)) if winners else None,
        avg_loser=float(np.mean(losers)) if losers else None,
        profit_factor=(sum(winners) / -sum(losers)) if winners and losers else None,
        realized_pl=float(sum(t.pnl for t in trips)),
        turnover=traded / avg_equity if avg_equity > 0 and fills else None,
        avg_exposure=float(np.mean(exposures)) if exposures else None,
        daily=daily,
        monthly=monthly,
        by_symbol=dict(sorted(by_symbol.items(), key=lambda kv: kv[1])),
        by_exit=dict(by_exit),
        notes=notes,
    )
