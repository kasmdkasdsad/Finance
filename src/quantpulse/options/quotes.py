"""Option quotes and whether they may be used — for research, or (much stricter) for execution.

For execution a quote is refused when it is stale, when the underlying's quote is stale, when the market is
crossed or locked, when the bid is zero or negative, when the ask is missing or not above the bid, when the
spread is absurd, when a Greek the decision needs is missing or impossible, when the contract has expired
or stopped trading, and when the market is closed. Nothing is ever substituted: a delayed research quote is
never passed off as an execution quote, and every verdict names the feed it came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from quantpulse.core.market_calendar import is_market_open
from quantpulse.options.contracts import OptionContract

Feed = Literal["opra", "indicative", "recorded", "model", "unknown"]
# Firm consolidated quotes (OPRA) are execution grade. Alpaca's free "indicative" feed is derived from them and
# is not firm: usable for paper limit orders when fresh, always labelled. Recorded (historical) and model-priced
# data are research only and can never reach an order.
EXECUTION_FEEDS: tuple[Feed, ...] = ("opra", "indicative")


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None  # per calendar day, per share
    vega: float | None = None  # per volatility point (1%), per share
    rho: float | None = None  # per 1% rate, per share
    source: Literal["vendor", "model", "none"] = "none"

    def missing(self, required: tuple[str, ...]) -> list[str]:
        return [g for g in required if getattr(self, g) is None]

    def impossible(self, kind: str) -> list[str]:
        """Values no option can have (a vendor glitch, a mis-parsed field)."""
        bad: list[str] = []
        if self.delta is not None:
            lo, hi = (0.0, 1.0) if kind == "call" else (-1.0, 0.0)
            if not (math.isfinite(self.delta) and lo - 1e-6 <= self.delta <= hi + 1e-6):
                bad.append(f"delta {self.delta}")
        if self.gamma is not None and not (math.isfinite(self.gamma) and self.gamma >= -1e-9):
            bad.append(f"gamma {self.gamma}")
        if self.vega is not None and not (math.isfinite(self.vega) and self.vega >= -1e-9):
            bad.append(f"vega {self.vega}")
        if self.theta is not None and not math.isfinite(self.theta):
            bad.append(f"theta {self.theta}")
        return bad


@dataclass(frozen=True, slots=True)
class OptionQuote:
    contract: OptionContract
    bid: float | None
    ask: float | None
    quote_at: datetime | None
    feed: Feed
    source: str  # the provider (alpaca, recorded, model, …)
    bid_size: float | None = None
    ask_size: float | None = None
    last: float | None = None
    last_at: datetime | None = None
    volume: float | None = None
    open_interest: float | None = None
    iv: float | None = None
    greeks: Greeks = field(default_factory=Greeks)
    underlying_price: float | None = None
    underlying_at: datetime | None = None

    @property
    def symbol(self) -> str:
        return self.contract.symbol

    @property
    def two_sided(self) -> bool:
        return self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > self.bid

    @property
    def mid(self) -> float | None:
        return 0.5 * (self.bid + self.ask) if self.two_sided else None  # type: ignore[operator]

    @property
    def spread(self) -> float | None:
        return self.ask - self.bid if self.two_sided else None  # type: ignore[operator]

    @property
    def spread_pct(self) -> float | None:
        """Spread as a share of the mid (0.10 = 10%)."""
        mid, spread = self.mid, self.spread
        return spread / mid if mid and spread is not None else None

    def age(self, now: datetime) -> float | None:
        return (now - self.quote_at).total_seconds() if self.quote_at is not None else None

    def underlying_age(self, now: datetime) -> float | None:
        return (now - self.underlying_at).total_seconds() if self.underlying_at is not None else None


@dataclass(frozen=True, slots=True)
class QuoteRules:
    """Execution limits (research uses only the structural checks). Defaults come from the settings."""

    max_quote_age_seconds: float = 60.0
    max_underlying_age_seconds: float = 60.0
    max_spread_pct: float = 0.15
    max_spread_dollars: float = 0.50
    min_bid: float = 0.05
    required_greeks: tuple[str, ...] = ("delta",)
    execution_feeds: tuple[str, ...] = EXECUTION_FEEDS
    clock_tolerance_seconds: float = 5.0  # a quote stamped this far in the future is still believed


@dataclass(frozen=True, slots=True)
class QuoteVerdict:
    symbol: str
    usable_for_research: bool
    usable_for_execution: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    feed: str
    age_seconds: float | None
    spread_pct: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "research": self.usable_for_research,
            "execution": self.usable_for_execution,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "feed": self.feed,
            "age_seconds": None if self.age_seconds is None else round(self.age_seconds, 1),
            "spread_pct": None if self.spread_pct is None else round(self.spread_pct, 4),
        }


def validate(q: OptionQuote, now: datetime, rules: QuoteRules | None = None) -> QuoteVerdict:
    """Every reason not to use this quote. Structural problems stop research use too; the rest only
    execution. The market being closed stops execution but not research."""
    r = rules or QuoteRules()
    structural: list[str] = []
    execution: list[str] = []
    warnings: list[str] = []
    c = q.contract
    if c.expired(now):
        structural.append("the contract has expired")
    if q.bid is not None and q.ask is not None and q.bid > 0 and q.ask > 0 and q.bid > q.ask:
        structural.append(f"crossed market (bid {q.bid} > ask {q.ask})")
    elif q.bid is not None and q.ask is not None and q.bid > 0 and q.bid == q.ask:
        execution.append(f"locked market (bid = ask = {q.bid})")
    if q.ask is None or not math.isfinite(q.ask) or q.ask <= 0:
        structural.append("no valid ask")
    if q.bid is None or not math.isfinite(q.bid) or q.bid <= 0:
        execution.append("zero or missing bid (no buyer: nothing could be sold back)")
    elif q.bid < r.min_bid:
        execution.append(f"bid ${q.bid:.2f} below ${r.min_bid:.2f}")
    bad = q.greeks.impossible(c.kind)
    if bad:
        structural.append("impossible Greeks: " + ", ".join(bad))
    if q.iv is not None and not (math.isfinite(q.iv) and 0 < q.iv < 5):
        structural.append(f"impossible implied volatility {q.iv}")
    if q.mid is not None and q.underlying_price:
        intrinsic = c.intrinsic(q.underlying_price)
        if (
            q.ask is not None and q.ask < intrinsic - 0.05
        ):  # an ask below intrinsic value: arbitrage or bad data
            structural.append(f"ask {q.ask} below intrinsic value {intrinsic:.2f}")
    missing = q.greeks.missing(r.required_greeks)
    if missing:
        execution.append("missing Greeks: " + ", ".join(missing))
    age = q.age(now)
    if age is None:
        execution.append("the quote has no timestamp (its age cannot be known)")
    elif age < -r.clock_tolerance_seconds:
        execution.append(f"the quote is stamped {-age:.0f}s in the future (clock skew)")
    elif age > r.max_quote_age_seconds:
        execution.append(f"stale quote ({age:.0f}s old, limit {r.max_quote_age_seconds:.0f}s)")
    u_age = q.underlying_age(now)
    if q.underlying_price is None or u_age is None:
        execution.append("no timestamped underlying price")
    elif u_age > r.max_underlying_age_seconds:
        execution.append(f"stale underlying ({u_age:.0f}s old, limit {r.max_underlying_age_seconds:.0f}s)")
    spct = q.spread_pct
    if spct is not None:
        if spct > r.max_spread_pct:
            execution.append(f"spread {spct:.0%} of the mid (limit {r.max_spread_pct:.0%})")
        if q.spread is not None and q.spread > r.max_spread_dollars and spct > r.max_spread_pct / 2:
            execution.append(f"spread ${q.spread:.2f} wide (limit ${r.max_spread_dollars:.2f})")
    if q.feed not in r.execution_feeds:
        execution.append(f"{q.feed} data is research only, never an execution quote")
    elif q.feed == "indicative":
        warnings.append("indicative feed: derived from OPRA, not firm (paper limit orders only)")
    if not is_market_open(now):
        execution.append("the market is closed")
    if q.open_interest is not None and q.open_interest == 0:
        warnings.append("no open interest")
    return QuoteVerdict(
        symbol=q.symbol,
        usable_for_research=not structural,
        usable_for_execution=not structural and not execution,
        blockers=tuple(structural + execution),
        warnings=tuple(warnings),
        feed=q.feed,
        age_seconds=age,
        spread_pct=spct,
    )
