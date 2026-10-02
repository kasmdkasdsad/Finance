"""The options market-data interface the Brain reads — and what it knows about the data's quality.

:class:`OptionsMarketDataProvider` is the only way the Brain sees option data: contracts, chains
(snapshots with quotes, trades and Greeks), latest quotes and trades, historical bars and a live stream.
Adapters (Alpaca first, :mod:`quantpulse.providers.alpaca_options`) hide each vendor's format; nothing
outside an adapter knows it.

Every chain carries its provenance, exposed to the Brain as ``OPTIONS_DATA_FEED`` (``opra``, ``indicative``,
``recorded``, ``model``), ``OPTIONS_DATA_SOURCE`` (the provider), ``OPTIONS_DATA_AGE`` (seconds since the
oldest quote the decision would use) and ``OPTIONS_DATA_QUALITY`` (a verdict with its reasons). A chain
from a delayed or model source is research data and is labelled so; it is never an execution quote.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from quantpulse.options.contracts import OptionContract
from quantpulse.options.quotes import OptionQuote, QuoteRules, validate


@dataclass(frozen=True, slots=True)
class ContractInfo:
    contract: OptionContract
    tradable: bool
    status: str  # active | inactive
    open_interest: float | None = None
    open_interest_date: date | None = None
    close_price: float | None = None


@dataclass(frozen=True, slots=True)
class OptionBar:
    symbol: str
    at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class OptionTrade:
    symbol: str
    at: datetime
    price: float
    size: float
    exchange: str | None = None


@dataclass
class ChainSnapshot:
    underlying: str
    underlying_price: float
    underlying_at: datetime | None
    fetched_at: datetime
    feed: str
    source: str
    quotes: list[OptionQuote] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def quality(self, now: datetime, rules: QuoteRules | None = None) -> dict[str, Any]:
        """OPTIONS_DATA_QUALITY: how much of the chain is usable for research and for execution, why not,
        and the data's age and feed."""
        verdicts = [validate(q, now, rules) for q in self.quotes]
        research = sum(v.usable_for_research for v in verdicts)
        execution = sum(v.usable_for_execution for v in verdicts)
        reasons: dict[str, int] = {}
        for v in verdicts:
            for b in v.blockers:
                key = b.split(" (")[0]
                reasons[key] = reasons.get(key, 0) + 1
        ages = [v.age_seconds for v in verdicts if v.age_seconds is not None]
        grade = (
            "execution" if execution else "research" if research else "unusable"
        ) if self.quotes else "empty"  # fmt: skip
        return {
            "OPTIONS_DATA_FEED": self.feed,
            "OPTIONS_DATA_SOURCE": self.source,
            "OPTIONS_DATA_AGE": round(max(ages), 1) if ages else None,
            "OPTIONS_DATA_QUALITY": grade,
            "contracts": len(self.quotes),
            "usable_for_research": research,
            "usable_for_execution": execution,
            "blockers": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:10]),
            "underlying_age": None
            if self.underlying_at is None
            else round((now - self.underlying_at).total_seconds(), 1),
            "notes": list(self.notes),
        }

    def expirations(self) -> list[date]:
        return sorted({q.contract.expiration for q in self.quotes})

    def for_expiration(self, expiration: date) -> list[OptionQuote]:
        return [q for q in self.quotes if q.contract.expiration == expiration]

    def find(self, symbol: str) -> OptionQuote | None:
        return next((q for q in self.quotes if q.symbol == symbol), None)


@runtime_checkable
class OptionsMarketDataProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    @property
    def feed(self) -> str: ...

    async def contracts(
        self,
        underlying: str,
        *,
        expiration_from: date | None = None,
        expiration_to: date | None = None,
        strike_from: float | None = None,
        strike_to: float | None = None,
    ) -> list[ContractInfo]: ...

    async def chain(
        self, underlying: str, *, expiration_from: date | None = None, expiration_to: date | None = None
    ) -> ChainSnapshot: ...

    async def latest_quotes(self, symbols: Sequence[str]) -> dict[str, OptionQuote]: ...

    async def latest_trades(self, symbols: Sequence[str]) -> dict[str, OptionTrade]: ...

    async def bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, list[OptionBar]]: ...

    def stream(self, symbols: Sequence[str]) -> AsyncIterator[OptionQuote | OptionTrade]: ...
