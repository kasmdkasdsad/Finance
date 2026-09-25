"""Alpaca Market Data provider (key id + secret): stock snapshots, bars and option snapshots.

The free plan serves the IEX feed for equities and the ``indicative`` feed for options; set
``QP_ALPACA_STOCK_FEED=sip`` / ``QP_ALPACA_OPTIONS_FEED=opra`` with a paid subscription.
Alpaca option snapshots carry quotes, trades and IV but not open interest.

IEX is a single exchange: its bid/ask is IEX's own book, not the national best bid/offer, and can be far
wider than the market (or one-sided) for names IEX trades thinly. :meth:`Alpaca.consolidated_quotes`
reads the consolidated (SIP) quote instead — real time with a subscription that allows it, else the
15-minute-delayed SIP feed — so spreads can be measured across all exchanges.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from pydantic import Field

from quantpulse.core.errors import ProviderHTTPError, ProviderNotConfigured
from quantpulse.core.http import HttpClient
from quantpulse.providers.base import (
    WireModel,
    finite_or_none,
    occ_parse,
    parse_wire,
    positive_or_none,
    require,
)
from quantpulse.schemas.market import Bar, Interval, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract

logger = logging.getLogger(__name__)

NAME = "alpaca"
# Consolidated (all-exchange) quote feeds, best first; a feed the subscription refuses is skipped for a while.
CONSOLIDATED_FEEDS = ("sip", "delayed_sip")
FEED_REFUSAL_TTL = timedelta(hours=1)
LATEST_QUOTE_BATCH = 200
TIMEFRAME: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
    "1wk": "1Week",
    "1mo": "1Month",
}
MAX_PAGES = 10
BATCH_SYMBOLS = 50  # symbols per multi-symbol bars request (a page holds at most 10,000 bars)
BATCH_MAX_PAGES = 200


class _Trade(WireModel):
    t: datetime | None = None
    p: float | None = None


class _Quote(WireModel):
    t: datetime | None = None
    bp: float | None = None
    ap: float | None = None
    bs: float | None = None
    as_: float | None = Field(default=None, alias="as")  # "as" is a Python keyword


class _Bar(WireModel):
    t: datetime
    o: float | None = None
    h: float | None = None
    l: float | None = None  # noqa: E741
    c: float | None = None
    v: float | None = None
    vw: float | None = None


class _StockSnapshot(WireModel):
    latestTrade: _Trade | None = None
    latestQuote: _Quote | None = None
    dailyBar: _Bar | None = None
    prevDailyBar: _Bar | None = None
    minuteBar: _Bar | None = None


class _BarsResponse(WireModel):
    bars: list[_Bar] | None = None
    next_page_token: str | None = None


class _MultiBarsResponse(WireModel):
    bars: dict[str, list[_Bar] | None] | None = None
    next_page_token: str | None = None


class _OptionSnapshot(WireModel):
    latestQuote: _Quote | None = None
    latestTrade: _Trade | None = None
    impliedVolatility: float | None = None
    dailyBar: _Bar | None = None


class _LatestQuotesResponse(WireModel):
    quotes: dict[str, _Quote | None] | None = None


@dataclass(frozen=True, slots=True)
class ConsolidatedQuote:
    """A bid/ask across all US exchanges (SIP), possibly 15 minutes delayed (``feed='delayed_sip'``)."""

    symbol: str
    bid: float | None
    ask: float | None
    timestamp: datetime | None
    feed: str


class _OptionSnapshotsResponse(WireModel):
    snapshots: dict[str, _OptionSnapshot] | None = None
    next_page_token: str | None = None


def vendor_symbol(symbol: str) -> str:
    """Alpaca writes class shares with a dot (BRK.B) where Yahoo/SEC use a dash (BRK-B)."""
    return symbol.replace("-", ".")


class Alpaca:
    name = NAME

    def __init__(
        self,
        http: HttpClient,
        key_id: str | None,
        secret: str | None,
        data_url: str = "https://data.alpaca.markets",
        stock_feed: str = "iex",
        options_feed: str = "indicative",
    ) -> None:
        self._http = http
        self._key_id = key_id
        self._secret = secret
        self._base = data_url.rstrip("/")
        self._stock_feed = stock_feed
        self._options_feed = options_feed
        self._refused_feeds: dict[str, datetime] = {}  # consolidated feed -> when the subscription refused it

    def configured(self) -> bool:
        return bool(self._key_id and self._secret)

    def _headers(self) -> dict[str, str]:
        if not self.configured():
            raise ProviderNotConfigured(NAME, "QP_ALPACA_API_KEY_ID / QP_ALPACA_API_SECRET_KEY not set")
        return {"APCA-API-KEY-ID": self._key_id or "", "APCA-API-SECRET-KEY": self._secret or ""}

    async def quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        ours = {vendor_symbol(s): s for s in symbols}
        payload = await self._http.get_json(
            NAME,
            f"{self._base}/v2/stocks/snapshots",
            params={"symbols": ",".join(ours), "feed": self._stock_feed},
            headers=self._headers(),
        )
        if not isinstance(payload, dict):
            payload = {}
        out: dict[str, Quote] = {}
        for sym, raw in payload.items():
            if raw is None:
                continue
            symbol = ours.get(sym.upper(), sym.upper())
            snap = parse_wire(NAME, _StockSnapshot, raw)
            quote = _snapshot_to_quote(symbol, snap, self._stock_feed)
            if quote is not None:
                out[symbol] = quote
        require(bool(out), NAME, "no snapshots returned")
        return out

    @property
    def stock_feed(self) -> str:
        return self._stock_feed

    async def consolidated_quotes(self, symbols: Sequence[str]) -> dict[str, ConsolidatedQuote]:
        """Latest consolidated (SIP) bid/ask for ``symbols``: the real-time SIP feed when the subscription
        allows it, otherwise the 15-minute-delayed SIP feed; empty if neither is permitted."""
        if not symbols or not self.configured():
            return {}
        now = datetime.now(UTC)
        for feed in CONSOLIDATED_FEEDS:
            refused = self._refused_feeds.get(feed)
            if refused is not None and now - refused < FEED_REFUSAL_TTL:
                continue
            try:
                out = await self._latest_quotes(symbols, feed)
            except ProviderHTTPError as exc:
                if exc.status_code in (400, 401, 403, 422):  # not in this subscription (or unknown feed)
                    logger.info(
                        "Alpaca %s quotes unavailable (HTTP %s): trying the next feed", feed, exc.status_code
                    )
                    self._refused_feeds[feed] = now
                    continue
                raise
            self._refused_feeds.pop(feed, None)
            return out
        return {}

    async def _latest_quotes(self, symbols: Sequence[str], feed: str) -> dict[str, ConsolidatedQuote]:
        out: dict[str, ConsolidatedQuote] = {}
        for i in range(0, len(symbols), LATEST_QUOTE_BATCH):
            chunk = list(symbols[i : i + LATEST_QUOTE_BATCH])
            ours = {vendor_symbol(s): s for s in chunk}
            payload = await self._http.get_json(
                NAME,
                f"{self._base}/v2/stocks/quotes/latest",
                params={"symbols": ",".join(ours), "feed": feed},
                headers=self._headers(),
            )
            page = parse_wire(NAME, _LatestQuotesResponse, payload)
            for vendor, q in (page.quotes or {}).items():
                if q is None:
                    continue
                symbol = ours.get(vendor.upper(), vendor.upper())
                out[symbol] = ConsolidatedQuote(
                    symbol, positive_or_none(q.bp), positive_or_none(q.ap), q.t, feed
                )
        return out

    async def quote(self, symbol: str) -> Quote:
        quotes = await self.quotes([symbol])
        require(symbol in quotes, NAME, f"no snapshot for {symbol}")
        return quotes[symbol]

    async def history(self, symbol: str, interval: Interval, start: datetime, end: datetime) -> PriceHistory:
        params: dict[str, object] = {
            "timeframe": TIMEFRAME[interval],
            "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "adjustment": "all",
            "feed": self._stock_feed,
            "sort": "asc",
        }
        bars: list[Bar] = []
        for _ in range(MAX_PAGES):
            payload = await self._http.get_json(
                NAME,
                f"{self._base}/v2/stocks/{vendor_symbol(symbol)}/bars",
                params=params,
                headers=self._headers(),
            )
            page = parse_wire(NAME, _BarsResponse, payload)
            for b in page.bars or []:
                bar = Bar.sanitized(b.t, b.o, b.h, b.l, b.c, b.v)
                if bar is not None:
                    bars.append(bar)
            if not page.next_page_token:
                break
            params["page_token"] = page.next_page_token
        require(bool(bars), NAME, f"no bars for {symbol}")
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)

    async def histories(
        self, symbols: Sequence[str], interval: Interval, start: datetime, end: datetime
    ) -> dict[str, PriceHistory]:
        """Bars for many symbols through the multi-symbol endpoint (one paginated request per batch of
        ``BATCH_SYMBOLS``). Symbols the feed does not know are simply absent from the result."""
        out: dict[str, PriceHistory] = {}
        for i in range(0, len(symbols), BATCH_SYMBOLS):
            chunk = list(symbols[i : i + BATCH_SYMBOLS])
            ours = {vendor_symbol(s): s for s in chunk}
            params: dict[str, object] = {
                "symbols": ",".join(ours),
                "timeframe": TIMEFRAME[interval],
                "start": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "end": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "limit": 10000,
                "adjustment": "all",
                "feed": self._stock_feed,
                "sort": "asc",
            }
            collected: dict[str, list[Bar]] = {}
            for _ in range(BATCH_MAX_PAGES):
                payload = await self._http.get_json(
                    NAME, f"{self._base}/v2/stocks/bars", params=params, headers=self._headers()
                )
                page = parse_wire(NAME, _MultiBarsResponse, payload)
                for vendor, rows in (page.bars or {}).items():
                    symbol = ours.get(vendor.upper(), vendor.upper())
                    bucket = collected.setdefault(symbol, [])
                    for b in rows or []:
                        bar = Bar.sanitized(b.t, b.o, b.h, b.l, b.c, b.v)
                        if bar is not None:
                            bucket.append(bar)
                if not page.next_page_token:
                    break
                params["page_token"] = page.next_page_token
            for symbol, bars in collected.items():
                if bars:
                    out[symbol] = PriceHistory(symbol=symbol, interval=interval, bars=bars)
        return out

    async def option_chain(
        self, symbol: str, expirations: Sequence[date] | None, max_expirations: int = 8
    ) -> OptionChain:
        today = datetime.now(UTC).date()
        params: dict[str, object] = {
            "feed": self._options_feed,
            "limit": 1000,
            "expiration_date_gte": today.isoformat(),
        }
        snapshots: dict[str, _OptionSnapshot] = {}
        for _ in range(MAX_PAGES):
            payload = await self._http.get_json(
                NAME,
                f"{self._base}/v1beta1/options/snapshots/{vendor_symbol(symbol)}",
                params=params,
                headers=self._headers(),
            )
            page = parse_wire(NAME, _OptionSnapshotsResponse, payload)
            snapshots.update(page.snapshots or {})
            if not page.next_page_token:
                break
            params["page_token"] = page.next_page_token

        parsed: list[tuple[str, date, str, float, _OptionSnapshot]] = []
        for occ, snap in snapshots.items():
            try:
                _root, exp, kind, strike = occ_parse(occ)
            except ValueError:
                continue
            parsed.append((occ, exp, kind, strike, snap))
        all_exp = sorted({p[1] for p in parsed})
        require(bool(all_exp), NAME, f"no option snapshots for {symbol}")
        wanted = {e for e in (expirations or all_exp[:max_expirations]) if e in all_exp}
        require(bool(wanted), NAME, "requested expirations are not listed")
        contracts = [
            OptionContract(
                contract_symbol=occ,
                kind=kind,
                strike=strike,
                expiration=exp,
                bid=positive_or_none((snap.latestQuote or _Quote()).bp),
                ask=positive_or_none((snap.latestQuote or _Quote()).ap),
                last=positive_or_none((snap.latestTrade or _Trade()).p),
                volume=finite_or_none(snap.dailyBar.v) if snap.dailyBar else None,
                implied_volatility=positive_or_none(snap.impliedVolatility),
            )
            for occ, exp, kind, strike, snap in parsed
            if exp in wanted
        ]
        underlying = await self.quote(symbol)
        return OptionChain(
            underlying=symbol,
            underlying_price=underlying.price,
            as_of=underlying.timestamp,
            expirations=all_exp,
            contracts=sorted(contracts, key=lambda c: (c.expiration, c.strike, c.kind)),
        )


def _snapshot_to_quote(symbol: str, snap: _StockSnapshot, feed: str | None = None) -> Quote | None:
    trade = snap.latestTrade or _Trade()
    daily = snap.dailyBar
    price = positive_or_none(trade.p) or (positive_or_none(daily.c) if daily else None)
    if price is None:
        return None
    stamp = trade.t or (daily.t if daily else None) or datetime.now(UTC)
    q = snap.latestQuote or _Quote()
    return Quote(
        symbol=symbol,
        price=price,
        previous_close=positive_or_none(snap.prevDailyBar.c) if snap.prevDailyBar else None,
        bid=positive_or_none(q.bp),
        ask=positive_or_none(q.ap),
        bid_size=positive_or_none(q.bs),
        ask_size=positive_or_none(q.as_),
        day_open=positive_or_none(daily.o) if daily else None,
        day_high=positive_or_none(daily.h) if daily else None,
        day_low=positive_or_none(daily.l) if daily else None,
        volume=finite_or_none(daily.v) if daily else None,
        vwap=positive_or_none(daily.vw) if daily else None,
        timestamp=stamp,
        quote_timestamp=q.t,
        feed=feed,
    )
