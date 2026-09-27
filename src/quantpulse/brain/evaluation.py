"""Outcomes: grade predictions whose horizon has passed, against real closing prices only.

A prediction is due once its due date is a *completed* session. Its realised return runs from the entry
price recorded when it was made to the close on the due date; symbols are graded on that return minus the
benchmark's over the same span, ``@market`` views on the benchmark's own return. A hit means the realised
(relative) return had the predicted sign. Prices come from :class:`PriceSource` — the market service's daily
history in production (synthetic prices are refused, exactly like the prediction ledger), a fixed table in
tests. A prediction whose due-date close never arrives is voided after ``VOID_AFTER_DAYS``; one whose close is
simply late stays open.

Each graded prediction also gets an outcome breakdown in its context, so accuracy is never confused with
luck, risk or timing:

* ``benchmark_return`` — what the benchmark did over the same span;
* ``relative_z`` — the relative return in units of the call's own risk (its volatility at entry over the
  horizon): the *risk-adjusted* outcome; ``noise`` when it is within half a standard deviation;
* ``timing`` — the favourable relative return between the entry reference and the next close (a decision
  could first be acted on at that close, so this part is an execution effect, not capturable skill) and
  ``after_next_close`` — the part that remained;
* ``data_ok`` — whether it was made on usable data (fresh, live, or a closed market's last close).

When a consensus prediction is graded, the decision made on it in the same cycle gets its outcome too.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from sqlalchemy import select

from quantpulse.core.clock import Clock
from quantpulse.core.market_calendar import NEW_YORK, regular_close
from quantpulse.db.models import BrainDecisionRow, BrainPredictionRow
from quantpulse.db.session import Database
from quantpulse.schemas.common import DataStatus
from quantpulse.services.market import MarketService
from quantpulse.services.predictions import last_completed_session

from .types import MARKET

VOID_AFTER_DAYS = 10
NOISE_Z = 0.5
HORIZONS = (1, 5, 10, 21)  # sessions at which every call is also graded
USABLE_DATA = frozenset({"fresh", "live", "market_closed"})
CONCURRENCY = 4


class PriceSource(Protocol):
    async def closes(self, symbols: Sequence[str], since: date) -> dict[str, dict[date, float]]:
        """Daily closes by New York date for each symbol that has *real* prices (others are left out)."""
        ...


class MarketPrices:
    """Daily closes from the market service (warehouse first, then vendors); never synthetic."""

    def __init__(self, market: MarketService, clock: Clock) -> None:
        self._market = market
        self._clock = clock

    async def closes(self, symbols: Sequence[str], since: date) -> dict[str, dict[date, float]]:
        lookback = max(30, (self._clock.now().astimezone(NEW_YORK).date() - since).days + 10)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(symbol: str) -> tuple[str, dict[date, float] | None]:
            async with sem:
                try:
                    res = await self._market.history(symbol, "1d", lookback)
                except Exception:  # an unknown or delisted symbol only leaves its predictions open
                    return symbol, None
            if res.status is DataStatus.SYNTHETIC:
                return symbol, None
            return symbol, {b.timestamp.astimezone(NEW_YORK).date(): b.close for b in res.value.bars}

        got = await asyncio.gather(*(one(s) for s in symbols))
        return {s: c for s, c in got if c}


@dataclass
class EvaluationResult:
    evaluated: int = 0
    voided: int = 0
    pending: int = 0
    decisions: int = 0
    prediction_ids: tuple[int, ...] = ()


def grade(direction: int, entry: float, close: float, bench_entry: float | None, bench_close: float | None,
          absolute: bool) -> tuple[float, float, bool] | None:  # fmt: skip
    """(return, relative return, hit) — ``None`` when the benchmark leg is missing for a relative claim."""
    if entry <= 0:
        return None
    ret = close / entry - 1
    if absolute:
        rel = ret
    elif bench_entry and bench_close and bench_entry > 0:
        rel = ret - (bench_close / bench_entry - 1)
    else:
        return None
    return ret, rel, direction * rel > 0


def next_close(series: dict[date, float], made_at: datetime) -> tuple[date, float] | None:
    """The first close at which a decision made at ``made_at`` could be acted on."""
    local = made_at.astimezone(NEW_YORK)
    start = local.date() if local.time() < regular_close(local.date()) else local.date() + timedelta(days=1)
    days = sorted(d for d in series if d >= start)
    return (days[0], series[days[0]]) if days else None


def breakdown(
    row: BrainPredictionRow,
    relative: float,
    series: dict[date, float],
    bench: dict[date, float],
    bench_close: float | None,
) -> dict[str, object]:
    """How the outcome divides into benchmark, risk, noise, timing and data quality (see the module)."""
    ctx = row.context or {}
    out: dict[str, object] = {}
    absolute = row.benchmark == "absolute"
    if bench_close and row.entry_benchmark:
        out["benchmark_return"] = round(bench_close / row.entry_benchmark - 1, 6)
    vol = ctx.get("vol")
    if isinstance(vol, int | float) and vol > 0:
        sigma = vol * math.sqrt(max(row.horizon_days, 1) / 252)
        z = relative / sigma
        out["relative_z"] = round(z, 4)
        out["noise"] = abs(z) < NOISE_Z
    nxt = next_close(series, row.made_at)
    if nxt is not None and row.entry_price:
        day, close = nxt
        leg = close / row.entry_price - 1
        if not absolute:
            b = bench.get(day)
            leg = leg - (b / row.entry_benchmark - 1) if b and row.entry_benchmark else float("nan")
        if leg == leg:
            out["timing"] = round(row.direction * leg, 6)
            out["after_next_close"] = round(row.direction * relative - row.direction * leg, 6)
    state = ctx.get("data_state")
    out["data_ok"] = state is None or state in USABLE_DATA
    out["by_horizon"] = by_horizon(row, series, bench)
    return out


def by_horizon(
    row: BrainPredictionRow, series: dict[date, float], bench: dict[date, float]
) -> dict[str, float]:
    """The relative return at each standard horizon up to the call's own (in sessions after it was made),
    so the learning system can see at which horizon a signal actually works."""
    if not row.entry_price:
        return {}
    start = row.made_at.astimezone(NEW_YORK).date()
    days = sorted(d for d in series if d > start and d <= row.due_date)
    out: dict[str, float] = {}
    for h in HORIZONS:
        if h > max(row.horizon_days, 1) or len(days) < h:
            continue
        d = days[h - 1]
        rel = series[d] / row.entry_price - 1
        if row.benchmark != "absolute":
            b = bench.get(d)
            if not b or not row.entry_benchmark:
                continue
            rel -= b / row.entry_benchmark - 1
        out[str(h)] = round(rel, 6)
    return out


async def evaluate_due(db: Database, prices: PriceSource, clock: Clock, benchmark: str) -> EvaluationResult:
    now = clock.now()
    cutoff = last_completed_session(now)
    today = now.astimezone(NEW_YORK).date()
    async with db.session() as s:
        due = list(
            (
                await s.scalars(
                    select(BrainPredictionRow).where(
                        BrainPredictionRow.status == "open", BrainPredictionRow.due_date <= cutoff
                    )
                )
            ).all()
        )
    result = EvaluationResult()
    if not due:
        return result
    since = min(p.made_at.astimezone(NEW_YORK).date() for p in due) - timedelta(days=5)
    symbols = sorted({benchmark} | {p.subject for p in due if p.subject != MARKET})
    closes = await prices.closes(symbols, since)
    bench = closes.get(benchmark, {})
    done: list[int] = []
    async with db.session() as s:
        for stale in due:
            row = await s.get(BrainPredictionRow, stale.id)
            if row is None or row.status != "open" or row.entry_price is None:
                continue
            series = bench if row.subject == MARKET else closes.get(row.subject, {})
            close = series.get(row.due_date)
            bench_close = bench.get(row.due_date)
            graded = (
                grade(
                    row.direction,
                    row.entry_price,
                    close,
                    row.entry_benchmark,
                    bench_close,
                    row.benchmark == "absolute",
                )
                if close is not None
                else None
            )
            if graded is None:
                if (today - row.due_date).days > VOID_AFTER_DAYS:
                    row.status, row.evaluated_at = "void", now
                    row.context = {**row.context, "void_reason": "no real close on the due date"}
                    result.voided += 1
                else:
                    result.pending += 1
                continue
            row.realized_return, row.realized_relative, row.hit = graded
            row.context = {**row.context, "outcome": breakdown(row, graded[1], series, bench, bench_close)}
            row.status, row.evaluated_at = "evaluated", now
            result.evaluated += 1
            done.append(row.id)
    result.prediction_ids = tuple(done)
    result.decisions = await _decision_outcomes(db, done, now)
    return result


async def _decision_outcomes(db: Database, prediction_ids: Sequence[int], now: datetime) -> int:
    """Copy each graded consensus prediction's outcome onto the decision made from it (same cycle, same
    subject)."""
    if not prediction_ids:
        return 0
    n = 0
    async with db.session() as s:
        preds = (
            await s.scalars(
                select(BrainPredictionRow).where(
                    BrainPredictionRow.id.in_(prediction_ids), BrainPredictionRow.source_type == "consensus"
                )
            )
        ).all()
        for p in preds:
            decisions = (
                await s.scalars(
                    select(BrainDecisionRow).where(
                        BrainDecisionRow.cycle_id == p.cycle_id,
                        BrainDecisionRow.subject == p.subject,
                        BrainDecisionRow.evaluated_at.is_(None),
                    )
                )
            ).all()
            for d in decisions:
                d.outcome = {
                    "prediction_id": p.id,
                    "horizon_days": p.horizon_days,
                    "due_date": p.due_date.isoformat(),
                    "return": p.realized_return,
                    "relative": p.realized_relative,
                    "benchmark": p.benchmark,
                    "consensus_direction": p.direction,
                    "consensus_hit": p.hit,
                }
                d.evaluated_at = now
                n += 1
    return n
