"""Events: what happened, recorded once, and what the brain should do about it.

An :class:`EventBus` persists every event (``brain_events``), keeps a short in-memory history, drops
repeats of the same (type, subject) inside a cooldown, and calls the handlers subscribed to its type.
Handlers never run agents themselves: the supervisor subscribes and turns events into *wake-ups* — a
focused cycle on the symbols involved, a portfolio review, a learning pass — that it schedules within its
own rate limits (so a burst of events cannot trigger a burst of cycles).

Where events come from — existing data, nothing invented:

* **brain cycles** (:func:`from_cycle`): regime changes, quotes that went stale, large price moves, volume
  spikes, detected opportunities, earnings approaching for a holding, portfolio and position changes since
  the last cycle, agent completions and failures;
* **the trading service's own audit trail** (:class:`TradingEventBridge`, read only): orders submitted,
  filled or canceled, and risk limits triggered (risk rejections, the daily loss limit, the kill switch);
* **learning passes**: predictions matured and decision outcomes available;
* **news** (:class:`NewsSource`): an interface only — QuantPulse has no news provider yet, so nothing is
  emitted until one is configured.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import select

from quantpulse.db.models import BrainEventRow, TradingEventRow
from quantpulse.db.session import Database

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    MARKET_DATA_UPDATED = "MarketDataUpdated"
    QUOTE_BECAME_STALE = "QuoteBecameStale"
    PRICE_MOVE_DETECTED = "PriceMoveDetected"
    VOLUME_SPIKE_DETECTED = "VolumeSpikeDetected"
    NEWS_EVENT_DETECTED = "NewsEventDetected"
    EARNINGS_APPROACHING = "EarningsApproaching"
    MARKET_REGIME_CHANGED = "MarketRegimeChanged"
    PORTFOLIO_CHANGED = "PortfolioChanged"
    POSITION_CHANGED = "PositionChanged"
    ORDER_SUBMITTED = "OrderSubmitted"
    ORDER_FILLED = "OrderFilled"
    ORDER_CANCELED = "OrderCanceled"
    RISK_LIMIT_TRIGGERED = "RiskLimitTriggered"
    OPPORTUNITY_DETECTED = "OpportunityDetected"
    AGENT_COMPLETED = "AgentCompleted"
    AGENT_FAILED = "AgentFailed"
    PREDICTION_MATURED = "PredictionMatured"
    TRADE_OUTCOME_AVAILABLE = "TradeOutcomeAvailable"


# repeats of the same (type, subject) inside these windows are dropped (they carry no new information)
COOLDOWN = {
    EventType.QUOTE_BECAME_STALE: timedelta(minutes=30),
    EventType.PRICE_MOVE_DETECTED: timedelta(minutes=60),
    EventType.VOLUME_SPIKE_DETECTED: timedelta(minutes=60),
    EventType.EARNINGS_APPROACHING: timedelta(hours=20),
    EventType.OPPORTUNITY_DETECTED: timedelta(hours=4),
    EventType.MARKET_DATA_UPDATED: timedelta(minutes=5),
}


@dataclass
class Event:
    type: EventType
    subject: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    at: datetime | None = None
    cycle_id: int | None = None
    source: str = "brain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "subject": self.subject,
            "payload": self.payload,
            "at": self.at.isoformat() if self.at else None,
            "cycle_id": self.cycle_id,
            "source": self.source,
        }


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self, db: Database, clock: Any, history: int = 500) -> None:
        self._db = db
        self._clock = clock
        self._handlers: dict[EventType, list[Handler]] = {}
        self._last: dict[tuple[EventType, str | None], datetime] = {}
        self.recent: deque[Event] = deque(maxlen=history)
        self.dropped = 0

    def subscribe(self, types: Sequence[EventType], handler: Handler) -> None:
        for t in types:
            self._handlers.setdefault(t, []).append(handler)

    async def publish(self, events: Sequence[Event]) -> list[Event]:
        """Persist and dispatch ``events``; returns the ones not dropped as repeats."""
        now = self._clock.now()
        kept: list[Event] = []
        for e in events:
            e.at = e.at or now
            key = (e.type, e.subject)
            window = COOLDOWN.get(e.type)
            last = self._last.get(key)
            if window is not None and last is not None and e.at - last < window:
                self.dropped += 1
                continue
            self._last[key] = e.at
            kept.append(e)
        if not kept:
            return kept
        async with self._db.session() as s:
            for e in kept:
                s.add(
                    BrainEventRow(
                        type=e.type.value,
                        subject=e.subject,
                        payload={**e.payload, "source": e.source},
                        cycle_id=e.cycle_id,
                        created_at=e.at,
                    )
                )
        for e in kept:
            self.recent.append(e)
            for handler in self._handlers.get(e.type, []):
                try:
                    await handler(e)
                except Exception:  # one handler never breaks the bus or the publisher
                    logger.exception("event handler failed for %s", e.type.value)
        return kept

    async def history(
        self, *, type: str | None = None, subject: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        stmt = select(BrainEventRow).order_by(BrainEventRow.id.desc()).limit(limit)
        if type:
            stmt = stmt.where(BrainEventRow.type == type)
        if subject:
            stmt = stmt.where(BrainEventRow.subject == subject)
        async with self._db.session() as s:
            rows = (await s.scalars(stmt)).all()
        return [
            {
                "id": r.id,
                "type": r.type,
                "subject": r.subject,
                "payload": r.payload,
                "cycle_id": r.cycle_id,
                "created_at": r.created_at,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------------------------- sources
def from_cycle(
    cycle_id: int,
    *,
    regime: str | None,
    previous_regime: str | None,
    states: dict[str, str],
    held: Sequence[str],
    focus: Sequence[str],
    moves: dict[str, float],
    opportunities: Sequence[Any],
    event_risk: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    previous_portfolio: dict[str, Any] | None,
    runs: Sequence[Any],
) -> list[Event]:
    """Events implied by one completed cycle (compared with the last one where needed)."""
    out: list[Event] = []

    def add(t: EventType, subject: str | None, **payload: Any) -> None:
        out.append(Event(t, subject, payload, cycle_id=cycle_id))

    if regime and previous_regime and regime != previous_regime:
        add(EventType.MARKET_REGIME_CHANGED, "@market", before=previous_regime, after=regime)
    for sym in dict.fromkeys([*held, *focus]):
        if states.get(sym) in ("stale", "provider_error"):
            add(EventType.QUOTE_BECAME_STALE, sym, state=states[sym])
        move = moves.get(sym)
        if move is not None and abs(move) >= 3:
            add(EventType.PRICE_MOVE_DETECTED, sym, move_sigma=round(move, 2), held=sym in held)
    for o in opportunities:
        if o.kind == "abnormal_volume":
            add(EventType.VOLUME_SPIKE_DETECTED, o.lead, headline=o.headline, **o.evidence)
        if o.lead is not None and o.status not in ("not_analysed", "rejected_data"):
            add(EventType.OPPORTUNITY_DETECTED, o.lead, kind=o.kind, headline=o.headline, status=o.status)
    for sym in held:
        days = (event_risk.get(sym) or {}).get("days_to_earnings")
        if days is not None and 0 <= days <= 5:
            add(
                EventType.EARNINGS_APPROACHING,
                sym,
                days=days,
                typical_move=(event_risk[sym] or {}).get("typical_move"),
            )
    if previous_portfolio is not None and portfolio.get("available") and previous_portfolio.get("available"):
        now_pos, before_pos = portfolio.get("positions") or {}, previous_portfolio.get("positions") or {}
        changed = False
        for sym in sorted(set(now_pos) | set(before_pos)):
            q_now = (now_pos.get(sym) or {}).get("qty", 0.0)
            q_before = (before_pos.get(sym) or {}).get("qty", 0.0)
            if abs(q_now - q_before) > 1e-9:
                changed = True
                add(EventType.POSITION_CHANGED, sym, qty_before=q_before, qty_after=q_now)
        eq_now, eq_before = portfolio.get("equity") or 0.0, previous_portfolio.get("equity") or 0.0
        if changed or (eq_before and abs(eq_now / eq_before - 1) >= 0.01):
            add(EventType.PORTFOLIO_CHANGED, "@portfolio", equity_before=eq_before, equity_after=eq_now)
    for r in runs:
        if r.status == "ok":
            add(
                EventType.AGENT_COMPLETED,
                r.agent_id,
                opinions=len(r.opinions),
                duration_ms=round(r.duration_ms, 1),
            )
        else:
            add(EventType.AGENT_FAILED, r.agent_id, status=r.status, error=r.error)
    return out


TRADING_KINDS = {
    "order_submitted": EventType.ORDER_SUBMITTED,
    "order_filled": EventType.ORDER_FILLED,
    "order_partially_filled": EventType.ORDER_FILLED,
    "order_canceled": EventType.ORDER_CANCELED,
    "order_expired": EventType.ORDER_CANCELED,
    "risk_rejected": EventType.RISK_LIMIT_TRIGGERED,
    "daily_loss_limit_reached": EventType.RISK_LIMIT_TRIGGERED,
    "kill_switch_activated": EventType.RISK_LIMIT_TRIGGERED,
    "brain_kill_switch_activated": EventType.RISK_LIMIT_TRIGGERED,
}


class TradingEventBridge:
    """Reads the trading service's audit trail (``trading_events``) — never writes to it — and turns new
    order and risk entries into brain events."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self.last_id: int | None = None

    async def poll(self) -> list[Event]:
        async with self._db.session() as s:
            if self.last_id is None:  # start from now: history before the brain was watching is not news
                self.last_id = int(
                    await s.scalar(select(TradingEventRow.id).order_by(TradingEventRow.id.desc())) or 0
                )
                return []
            rows = (
                await s.scalars(
                    select(TradingEventRow)
                    .where(TradingEventRow.id > self.last_id)
                    .order_by(TradingEventRow.id)
                )
            ).all()
        out: list[Event] = []
        for r in rows:
            self.last_id = r.id
            t = TRADING_KINDS.get(r.kind)
            if t is None:
                continue
            out.append(
                Event(
                    t,
                    r.symbol or ("@portfolio" if t is EventType.RISK_LIMIT_TRIGGERED else None),
                    {"kind": r.kind, "message": r.message[:300], "client_order_id": r.client_order_id},
                    at=r.created_at,
                    source="trading",
                )
            )
        return out


class NewsSource(Protocol):
    """A provider of news headlines for symbols. QuantPulse has none yet: nothing is emitted until one is
    configured (no invented news)."""

    def configured(self) -> bool: ...

    async def headlines(self, symbols: Sequence[str], since: datetime) -> list[dict[str, Any]]: ...


class NoNews:
    def configured(self) -> bool:
        return False

    async def headlines(self, symbols: Sequence[str], since: datetime) -> list[dict[str, Any]]:
        return []


async def news_events(source: NewsSource, symbols: Sequence[str], since: datetime) -> list[Event]:
    if not source.configured():
        return []
    return [
        Event(
            EventType.NEWS_EVENT_DETECTED,
            h.get("symbol"),
            {k: v for k, v in h.items() if k != "symbol"},
            source="news",
        )
        for h in await source.headlines(symbols, since)
    ]
