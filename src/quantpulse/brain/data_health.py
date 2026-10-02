"""What market data the Brain actually has — symbol by symbol, and for the cycle as a whole.

A quote is not simply "live" or "stale". :func:`diagnose` gives each symbol one precise
:class:`QuoteStatus` and the reasons behind it:

* **fresh / live** — a real-time print within ``QP_BRAIN_FRESH_QUOTE_SECONDS`` / the trading limit
  ``QP_TRADING_MAX_QUOTE_AGE_SECONDS``;
* **stale** — the market is open but the last print on this feed is older than the limit;
* **no_trade_today** — the feed has not printed the symbol since the open: its price is the previous
  session's;
* **delayed** — the feed itself is delayed (the 15-minute SIP feed): never an executable price;
* **missing** — no quote came back; **subscription_unavailable** — the vendor refused the feed for this
  subscription; **provider_error** — the request failed;
* **invalid_timestamp** — stamped in the future: its age cannot be known (bad data or a wrong clock);
* **market_closed** / **holiday** — outside the regular session: last-session values, not stale data;
* **synthetic** — simulated prices: never data.

Each diagnosis also says what the price and the spread were measured on — IEX alone (one exchange with a
few percent of US volume), the consolidated SIP quote in real time, or the 15-minute delayed SIP quote —
and every quote problem the existing validation found. :func:`feed_report` explains the cycle's data in
plain words, most important cause first, including the difference between this computer's clock and
Alpaca's (every quote age is off by that much).

Nothing here loosens a limit. The coarse :class:`~quantpulse.brain.types.DataState` each status maps to
is what decides whether an action can be executable, exactly as before; this module makes the reason
visible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from quantpulse.core.market_calendar import NEW_YORK, REGULAR_OPEN, is_trading_day
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_data import (
    FEED_LABELS,
    FUTURE_TOLERANCE_SECONDS,
    LiveQuote,
    QuoteQuality,
)

from .types import DataState

CLOCK_SKEW_WARN_SECONDS = 2.0  # worth saying
CLOCK_SKEW_VETO_SECONDS = 30.0  # quote ages are wrong by more than this: nothing executable


class QuoteStatus(StrEnum):
    FRESH = "fresh"
    LIVE = "live"
    STALE = "stale"
    NO_TRADE_TODAY = "no_trade_today"
    DELAYED = "delayed"
    MISSING = "missing"
    SUBSCRIPTION = "subscription_unavailable"
    PROVIDER_ERROR = "provider_error"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MARKET_CLOSED = "market_closed"
    HOLIDAY = "holiday"
    SYNTHETIC = "synthetic"


COARSE: dict[QuoteStatus, DataState] = {
    QuoteStatus.FRESH: DataState.FRESH,
    QuoteStatus.LIVE: DataState.LIVE,
    QuoteStatus.STALE: DataState.STALE,
    QuoteStatus.NO_TRADE_TODAY: DataState.STALE,
    QuoteStatus.DELAYED: DataState.STALE,
    QuoteStatus.MISSING: DataState.UNAVAILABLE,
    QuoteStatus.SUBSCRIPTION: DataState.UNAVAILABLE,
    QuoteStatus.SYNTHETIC: DataState.UNAVAILABLE,
    QuoteStatus.PROVIDER_ERROR: DataState.PROVIDER_ERROR,
    QuoteStatus.INVALID_TIMESTAMP: DataState.INVALID,
    QuoteStatus.MARKET_CLOSED: DataState.MARKET_CLOSED,
    QuoteStatus.HOLIDAY: DataState.MARKET_CLOSED,
}

COVERAGE = {
    "iex": "IEX only (one exchange, a few percent of US volume)",
    "sip": "SIP (all US exchanges, real time)",
    "delayed_sip": "SIP (all US exchanges, 15 minutes delayed)",
}


def closed_reason(moment: datetime) -> str | None:
    """Why the regular session is not open at ``moment`` on the calendar (``None`` if it should be)."""
    local = moment.astimezone(NEW_YORK)
    if local.weekday() >= 5:
        return "weekend"
    if not is_trading_day(local.date()):
        return "holiday"
    if local.time() < REGULAR_OPEN:
        return "pre-market"
    return None  # a trading day after the open: the broker's clock decides (after hours, early close)


@dataclass
class QuoteDiagnosis:
    symbol: str
    status: QuoteStatus
    feed: str | None = None
    coverage: str = "none"
    trade_age_s: float | None = None
    quote_age_s: float | None = None
    price_age_s: float | None = None  # what the live-data check measures (the freshest reliable observation)
    price_source: str | None = None
    consolidated: str | None = None  # what the spread could be checked against beyond the primary feed
    spread_bps: float | None = None
    spread_source: str = "unavailable"
    reasons: list[str] = field(default_factory=list)

    @property
    def state(self) -> DataState:
        return COARSE[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "status": self.status.value,
            "state": self.state.value,
            "feed": self.feed,
            "coverage": self.coverage,
            "trade_age_s": None if self.trade_age_s is None else round(self.trade_age_s, 1),
            "price_age_s": None if self.price_age_s is None else round(self.price_age_s, 1),
            "price_source": self.price_source,
            "quote_age_s": None if self.quote_age_s is None else round(self.quote_age_s, 1),
            "consolidated": self.consolidated,
            "spread_bps": None if self.spread_bps is None else round(self.spread_bps, 1),
            "spread_source": self.spread_source,
            "reasons": list(self.reasons),
        }


def diagnose(
    symbol: str,
    *,
    quote: LiveQuote | None,
    quality: QuoteQuality | None,
    missing_reason: str | None,
    market_open: bool,
    closed: str | None,
    price_status: DataStatus,
    fresh_seconds: float,
    max_age_seconds: float,
    refused_feed: str | None = None,
) -> QuoteDiagnosis:
    """The precise state of one symbol's market data. ``closed`` says why the market is shut (weekend,
    holiday, pre-market, after hours); ``refused_feed`` names the primary feed if the vendor refused it."""
    d = QuoteDiagnosis(symbol, QuoteStatus.MISSING)
    if quote is not None:
        d.feed = quote.feed
        d.coverage = COVERAGE.get(quote.feed or "", quote.provider)
        d.trade_age_s = quote.trade_age_seconds
        d.quote_age_s = quote.quote_age_seconds
        d.price_age_s = quote.age_seconds
        d.price_source = quote.price_source
        if quote.nbbo_feed:
            age = quote.nbbo_age_seconds
            d.consolidated = FEED_LABELS.get(quote.nbbo_feed, quote.nbbo_feed) + (
                f", {age:,.0f}s old" if age is not None else ""
            )
    if quality is not None:
        d.spread_bps, d.spread_source = quality.spread_bps, quality.spread_source
        d.reasons.extend(quality.problems)
        d.reasons.extend(quality.entry_blocks)

    if price_status is DataStatus.SYNTHETIC:
        d.status = QuoteStatus.SYNTHETIC
        d.reasons.insert(0, "only synthetic prices are available: never data")
        return d
    if not market_open:
        d.status = QuoteStatus.HOLIDAY if closed == "holiday" else QuoteStatus.MARKET_CLOSED
        when = (
            f"; last print {(quote.trade_time or quote.timestamp).astimezone(NEW_YORK):%a %H:%M} New York"
            if quote
            else ""
        )
        d.reasons.insert(0, f"market closed ({closed or 'outside the regular session'}){when}")
        return d
    if quote is None:
        reason = missing_reason or "no quote returned"
        low = reason.lower()
        if refused_feed:
            d.status = QuoteStatus.SUBSCRIPTION
            reason = f"the vendor refused the {refused_feed} feed for this subscription ({reason})"
        elif "fail" in low or "error" in low:
            d.status = QuoteStatus.PROVIDER_ERROR
        d.reasons.insert(0, reason)
        return d

    ahead = quote.ahead_seconds(quote.trade_time or quote.timestamp)
    if ahead > FUTURE_TOLERANCE_SECONDS:
        d.status = QuoteStatus.INVALID_TIMESTAMP
        d.trade_age_s = None
        d.reasons.insert(0, f"last print stamped {ahead:,.0f}s in the future: its age cannot be known")
        return d
    if quote.feed == "delayed_sip":
        d.status = QuoteStatus.DELAYED
        d.reasons.insert(0, "the price feed is 15 minutes delayed: never an executable price")
        return d
    local = quote.timestamp.astimezone(NEW_YORK)
    today = (quote.as_of or quote.timestamp).astimezone(NEW_YORK).date()
    if quote.price_basis == "trade" and (local.date() < today or local.time() < REGULAR_OPEN):
        d.status = QuoteStatus.NO_TRADE_TODAY
        d.reasons.insert(
            0,
            f"no print on {FEED_LABELS.get(quote.feed or '', quote.provider)} since the open; "
            f"the price is from {local:%a %H:%M} New York",
        )
        return d
    age = quote.age_seconds
    if quote.quote_age_seconds is not None and quality is not None and quality.usable_bid_ask:
        age = max(age, quote.quote_age_seconds)  # a believed bid/ask older than the print counts too
    if age > max_age_seconds:
        d.status = QuoteStatus.STALE
        d.reasons.insert(0, f"{quote.price_source} {age:,.0f}s old (limit {max_age_seconds:,.0f}s)")
    else:
        d.status = QuoteStatus.FRESH if age <= fresh_seconds else QuoteStatus.LIVE
        if quote.price_basis == "quote" and d.trade_age_s is not None:
            d.reasons.append(
                f"priced from {quote.venue}'s live bid/ask: no {quote.venue} print for {d.trade_age_s:,.0f}s"
            )
    if quote.feed == "iex" and d.status is not QuoteStatus.FRESH and not quote.nbbo_feed:
        d.reasons.append("IEX alone: a quiet IEX book says little about the whole market")
    return d


def feed_report(
    diagnoses: dict[str, QuoteDiagnosis],
    *,
    market_open: bool,
    closed: str | None,
    feeds: list[dict[str, Any]],
    skew_seconds: float | None,
    max_age_seconds: float,
) -> dict[str, Any]:
    """The cycle's market data in one place: counts by status, feeds, subscription refusals, clock skew,
    and the causes behind anything unusable, most important first."""
    counts = Counter(d.status.value for d in diagnoses.values())
    by_feed = Counter(d.feed or "none" for d in diagnoses.values())
    n = len(diagnoses)
    usable = counts[QuoteStatus.FRESH.value] + counts[QuoteStatus.LIVE.value]
    causes: list[str] = []

    if skew_seconds is not None and abs(skew_seconds) > CLOCK_SKEW_WARN_SECONDS:
        direction = "ahead of" if skew_seconds > 0 else "behind"
        causes.append(
            f"This computer's clock is {abs(skew_seconds):.1f}s {direction} Alpaca's: every quote age is off by "
            "that much (synchronise the system clock; the limits are not changed to compensate)."
        )
    if not market_open:
        causes.append(
            f"The market is closed ({closed or 'outside the regular session'}): quotes are last-session values. "
            "This is not stale data, and nothing is executable until the open."
        )
    for f in feeds:
        err = f.get("stock_feed_error")
        if err:
            causes.append(
                f"Alpaca refused the {f.get('stock_feed')} stock feed (HTTP {err.get('status')}) at "
                f"{str(err.get('at'))[:19]}: the subscription does not include it."
            )
        refused = f.get("refused_feeds") or {}
        if "sip" in refused:
            causes.append(
                "Real-time SIP is not in this subscription (refused at "
                f"{str(refused['sip'])[:19]}): spreads are checked on the 15-minute delayed SIP quote or on "
                "IEX alone."
            )
        if f.get("history_feed_refused"):
            causes.append(
                f"Alpaca refused all-exchange (SIP) price history (HTTP {f['history_feed_refused']}): daily bars "
                "are IEX's own, whose volume is a few percent of the market's, so liquidity limits read it low."
            )
    if market_open:
        invalid = counts[QuoteStatus.INVALID_TIMESTAMP.value]
        if invalid:
            causes.append(
                f"{invalid} quote(s) are stamped in the future: their age cannot be known (bad vendor "
                "timestamps or a wrong system clock)."
            )
        iex = [d for d in diagnoses.values() if d.feed == "iex"]
        quiet = [
            d for d in iex if d.status in (QuoteStatus.STALE, QuoteStatus.NO_TRADE_TODAY)
        ]  # the usual cause on the free plan
        if quiet:
            causes.append(
                f"{len(quiet)} of {len(iex)} IEX-priced symbols have no IEX print within {max_age_seconds:,.0f}s "
                "(or none since the open). IEX is one exchange with a few percent of US volume, so its last "
                "trade can be minutes old while the stock trades elsewhere. Those prices are treated as stale "
                "by design: real-time SIP data (QP_ALPACA_STOCK_FEED=sip with a subscription that includes "
                "it) is the fix, not a longer quote-age limit."
            )
        other = [
            d
            for d in diagnoses.values()
            if d.feed != "iex" and d.status in (QuoteStatus.STALE, QuoteStatus.NO_TRADE_TODAY)
        ]
        if other:
            sources = ", ".join(sorted({d.coverage for d in other}))
            causes.append(
                f"{len(other)} symbol(s) priced on {sources} have no print within {max_age_seconds:,.0f}s "
                "(or none since the open): stale, so not executable."
            )
        missing = counts[QuoteStatus.MISSING.value] + counts[QuoteStatus.PROVIDER_ERROR.value]
        if missing:
            causes.append(f"{missing} symbol(s) had no quote at all (missing or a failed request).")
        delayed = counts[QuoteStatus.DELAYED.value]
        if delayed:
            causes.append(f"{delayed} symbol(s) are priced from a 15-minute delayed feed: never executable.")
    if counts[QuoteStatus.SYNTHETIC.value]:
        causes.append("Prices are synthetic (no live data source): nothing is executable.")
    healthy = not causes
    if healthy:
        causes.append(f"Market data is healthy: {usable} of {n} symbols have fresh or live quotes.")
    return {
        "healthy": healthy,
        "symbols": n,
        "usable": usable,
        "counts": dict(counts),
        "feeds_seen": dict(by_feed),
        "vendors": feeds,
        "clock_skew_s": None if skew_seconds is None else round(skew_seconds, 2),
        "market_open": market_open,
        "closed": closed,
        "headline": causes[0],
        "causes": causes,
    }
