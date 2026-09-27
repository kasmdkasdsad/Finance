"""The brain's shared picture of the world for one cycle (situational awareness), plus working memory.

A :class:`BrainContext` is built once per cycle by :mod:`quantpulse.brain.perception` from QuantPulse's
existing services (the trading data loader, the Alpaca paper account read through :class:`BrokerView`,
the NYSE calendar). Agents only *read* it. :class:`WorkingMemory` is where agents post facts during the
cycle (the regime label, data states, …) so later agents can build on earlier ones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any

import pandas as pd

from quantpulse.core.market_calendar import NEW_YORK, Session, is_trading_day, session_at
from quantpulse.domain.trading_regime import MarketRegime
from quantpulse.providers.alpaca_trading import (
    AlpacaPaperBroker,
    BrokerAccount,
    BrokerOrder,
    BrokerPosition,
    MarketClock,
)
from quantpulse.schemas.common import DataStatus
from quantpulse.services.model import ModelSnapshot
from quantpulse.services.trading_data import LiveQuote, QuoteQuality
from quantpulse.services.trading_risk import RiskLimits

from .opportunities import Opportunity
from .types import BrainMode, BrainSession, DataState, Opinion

if TYPE_CHECKING:
    from .data_health import QuoteDiagnosis
    from .llm import ModelRouter


class BrokerView:
    """Read-only view of the Alpaca paper account for the brain: account, positions, open orders and the
    market clock. It has no way to place, change or cancel an order."""

    def __init__(self, broker: AlpacaPaperBroker) -> None:
        self._broker = broker

    def configured(self) -> bool:
        return self._broker.configured()

    async def account(self) -> BrokerAccount:
        return await self._broker.account()

    async def positions(self) -> list[BrokerPosition]:
        return await self._broker.positions()

    async def open_orders(self) -> list[BrokerOrder]:
        return await self._broker.open_orders()

    async def clock(self) -> MarketClock:
        return await self._broker.clock()


def brain_session(moment: datetime) -> BrainSession:
    local = moment.astimezone(NEW_YORK)
    if local.weekday() >= 5:
        return BrainSession.WEEKEND
    if not is_trading_day(local.date()):
        return BrainSession.HOLIDAY
    s = session_at(moment)
    if s is Session.REGULAR:
        return BrainSession.OPEN
    if s is Session.PRE or local.time() < time(9, 30):
        return BrainSession.PRE_MARKET
    return BrainSession.AFTER_HOURS


@dataclass
class PortfolioState:
    """The Alpaca paper account as read at the start of the cycle (Alpaca is authoritative)."""

    available: bool
    error: str | None = None
    account: BrokerAccount | None = None
    positions: dict[str, BrokerPosition] = field(default_factory=dict)
    open_orders: list[BrokerOrder] = field(default_factory=list)

    @property
    def equity(self) -> float:
        return self.account.equity if self.account else 0.0

    def weight(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.market_value / self.equity if p is not None and self.equity > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        a = self.account
        return {
            "available": self.available,
            "error": self.error,
            "equity": a.equity if a else None,
            "cash": a.cash if a else None,
            "buying_power": a.buying_power if a else None,
            "long_market_value": a.long_market_value if a else None,
            "day_pl_pct": a.day_pl_pct if a else None,
            "blocked": a.blocked if a else None,
            "positions": {
                s: {
                    "qty": p.qty,
                    "market_value": round(p.market_value, 2),
                    "weight": round(self.weight(s), 4),
                }
                for s, p in self.positions.items()
            },
            "open_orders": len(self.open_orders),
        }


@dataclass
class WorkingMemory:
    """What the brain is thinking about during one cycle: facts agents post for each other, every opinion
    so far (by subject), open questions for research, and notes."""

    facts: dict[str, Any] = field(default_factory=dict)
    opinions: dict[str, list[Opinion]] = field(default_factory=lambda: defaultdict(list))
    questions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def post(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def add(self, opinion: Opinion) -> None:
        self.opinions[opinion.subject].append(opinion)

    def ask(self, question: str, subject: str, raised_by: str, **detail: Any) -> None:
        self.questions.append({"question": question, "subject": subject, "raised_by": raised_by, **detail})


@dataclass
class BrainContext:
    as_of: datetime
    session: BrainSession
    market_open: bool
    clock_source: str  # "alpaca" (authoritative) or "calendar"
    mode: BrainMode
    universe: list[str]
    close: pd.DataFrame  # completed sessions × symbols
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series
    benchmark_symbol: str
    qqq: pd.Series | None
    price_status: DataStatus
    quotes: dict[str, LiveQuote]
    quality: dict[str, QuoteQuality]
    missing_quotes: dict[str, str]
    indicators: pd.DataFrame  # symbol × indicator (today's partial session included while the market is open)
    market_stats: dict[str, float | None]
    regime: MarketRegime | None
    vix: float | None
    implied_vol: dict[str, float]
    earnings: dict[str, tuple[date, str]]
    model_z: dict[str, float]
    fundamentals: pd.DataFrame | None
    sectors: dict[str, str]
    portfolio: PortfolioState
    data_states: dict[str, DataState]
    limits: RiskLimits
    kill_switch: bool
    data_health: dict[str, QuoteDiagnosis] = field(default_factory=dict)  # precise quote status per symbol
    feed: dict[str, Any] = field(default_factory=dict)  # the cycle's data report (feeds, clock skew, causes)
    trading_blockers: list[str] = field(default_factory=list)  # trading controls read now (never acted on)
    model: ModelSnapshot | None = None  # the stock model's live scores and raw features (fundamentals…)
    options: dict[str, dict[str, Any]] = field(default_factory=dict)  # option-chain metrics per symbol
    events: dict[str, dict[str, Any]] = field(default_factory=dict)  # earnings calendar and reactions
    opportunities: list[Opportunity] = field(default_factory=list)  # detected this cycle, strongest first
    strategy_signals: dict[str, Any] = field(default_factory=dict)  # promoted lab strategies' rankings
    focus: list[str] = field(default_factory=list)  # symbols this cycle studies closely
    focus_reasons: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    provider_errors: dict[str, str] = field(default_factory=dict)
    working: WorkingMemory = field(default_factory=WorkingMemory)
    llm: ModelRouter | None = None  # language models for the model-backed agents (never for calculations)

    @property
    def held(self) -> list[str]:
        return [s for s, p in self.portfolio.positions.items() if p.qty > 0]

    def ind(self, symbol: str, column: str) -> float | None:
        """One indicator value (``None`` if missing or not finite)."""
        if symbol not in self.indicators.index or column not in self.indicators.columns:
            return None
        v = self.indicators.at[symbol, column]
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f and abs(f) != float("inf") else None

    def feature(self, symbol: str, column: str) -> float | None:
        """One of the stock model's raw features (fundamentals, earnings reaction, sector momentum…)."""
        f = self.model.features if self.model is not None else None
        if f is None or f.empty or symbol not in f.index or column not in f.columns:
            return None
        try:
            v = float(f.at[symbol, column])
        except (TypeError, ValueError):
            return None
        return v if v == v and abs(v) != float("inf") else None

    def state(self, symbol: str) -> DataState:
        return self.data_states.get(symbol, DataState.UNAVAILABLE)

    def price(self, symbol: str) -> float | None:
        q = self.quotes.get(symbol)
        if q is not None:
            return q.price
        p = self.portfolio.positions.get(symbol)
        if p is not None and p.current_price > 0:
            return p.current_price
        if symbol in self.close.columns:
            s = self.close[symbol].dropna()
            return float(s.iloc[-1]) if len(s) else None
        return None
