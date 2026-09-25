"""Polygon.io provider (API key): real-time snapshots, aggregates and options-chain snapshots.

Endpoint availability depends on the Polygon plan. A 403 ``NOT_AUTHORIZED`` (e.g. snapshots on the free
tier) is surfaced as a provider error so the gateway fails over instead of mislabelling stale data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from urllib.parse import parse_qsl, urlsplit

from quantpulse.core.errors import ProviderNoData, ProviderNotConfigured
from quantpulse.core.http import HttpClient
from quantpulse.providers.base import (
    WireModel,
    epoch_to_datetime,
    finite_or_none,
    parse_wire,
    positive_or_none,
    require,
)
from quantpulse.schemas.market import Bar, Interval, PriceHistory, Quote
from quantpulse.schemas.options import OptionChain, OptionContract

NAME = "polygon"
TIMESPAN: dict[str, tuple[int, str]] = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "1h": (1, "hour"),
    "1d": (1, "day"),
    "1wk": (1, "week"),
    "1mo": (1, "month"),
}
MAX_PAGES = 10


class _Agg(WireModel):
    t: int
    o: float | None = None
    h: float | None = None
    l: float | None = None  # noqa: E741 - vendor field name
    c: float | None = None
    v: float | None = None


class _AggResponse(WireModel):
    status: str | None = None
    results: list[_Agg] | None = None
    next_url: str | None = None


class _DayBar(WireModel):
    o: float | None = None
    h: float | None = None
    l: float | None = None  # noqa: E741
    c: float | None = None
    v: float | None = None


class _LastTrade(WireModel):
    p: float | None = None
    t: int | None = None


class _LastQuote(WireModel):
    p: float | None = None  # bid
    P: float | None = None  # ask
    s: float | None = None
    S: float | None = None


class _TickerSnapshot(WireModel):
    ticker: str
    updated: int | None = None
    day: _DayBar | None = None
    prevDay: _DayBar | None = None
    min: _DayBar | None = None
    lastTrade: _LastTrade | None = None
    lastQuote: _LastQuote | None = None


class _SnapshotResponse(WireModel):
    status: str | None = None
    ticker: _TickerSnapshot | None = None


class _OptDetails(WireModel):
    contract_type: str
    expiration_date: date
    strike_price: float
    ticker: str


class _OptQuote(WireModel):
    bid: float | None = None
    ask: float | None = None


class _OptDay(WireModel):
    close: float | None = None
    volume: float | None = None


class _OptLastTrade(WireModel):
    price: float | None = None


class _Underlying(WireModel):
    price: float | None = None
    last_updated: int | None = None


class _OptSnapshot(WireModel):
    details: _OptDetails
    last_quote: _OptQuote | None = None
    last_trade: _OptLastTrade | None = None
    day: _OptDay | None = None
    open_interest: float | None = None
    implied_volatility: float | None = None
    underlying_asset: _Underlying | None = None


class _OptChainResponse(WireModel):
    status: str | None = None
    results: list[_OptSnapshot] | None = None
    next_url: str | None = None


class _ContractRef(WireModel):
    expiration_date: date


class _ContractsResponse(WireModel):
    results: list[_ContractRef] | None = None


def vendor_symbol(symbol: str) -> str:
    """Polygon writes class shares with a dot (BRK.B) where Yahoo/SEC use a dash (BRK-B)."""
    return symbol.replace("-", ".")


class Polygon:
    name = NAME

    def __init__(
        self, http: HttpClient, api_key: str | None, base_url: str = "https://api.polygon.io"
    ) -> None:
        self._http = http
        self._key = api_key
        self._base = base_url.rstrip("/")

    def configured(self) -> bool:
        return bool(self._key)

    def _params(self, extra: Mapping[str, object] | None = None) -> dict[str, object]:
        if not self._key:
            raise ProviderNotConfigured(NAME, "QP_POLYGON_API_KEY not set")
        return {**(extra or {}), "apiKey": self._key}

    async def _paged(
        self, url: str, params: dict[str, object], model: type[_AggResponse] | type[_OptChainResponse]
    ):
        payload = await self._http.get_json(NAME, url, params=self._params(params))
        page = parse_wire(NAME, model, payload)
        pages = [page]
        while page.next_url and len(pages) < MAX_PAGES:
            parts = urlsplit(page.next_url)
            next_params = dict(parse_qsl(parts.query))
            payload = await self._http.get_json(
                NAME, f"{parts.scheme}://{parts.netloc}{parts.path}", params=self._params(next_params)
            )
            page = parse_wire(NAME, model, payload)
            pages.append(page)
        return pages

    async def quote(self, symbol: str) -> Quote:
        payload = await self._http.get_json(
            NAME,
            f"{self._base}/v2/snapshot/locale/us/markets/stocks/tickers/{vendor_symbol(symbol)}",
            params=self._params(),
        )
        snap = parse_wire(NAME, _SnapshotResponse, payload).ticker
        if snap is None:
            raise ProviderNoData(NAME, f"no snapshot for {symbol}")
        last = snap.lastTrade or _LastTrade()
        day = snap.day or _DayBar()
        price = (
            positive_or_none(last.p) or positive_or_none((snap.min or _DayBar()).c) or positive_or_none(day.c)
        )
        require(price is not None, NAME, f"snapshot for {symbol} has no price")
        stamp_ns = last.t or snap.updated
        quote = snap.lastQuote or _LastQuote()
        return Quote(
            symbol=symbol,
            price=price,
            previous_close=positive_or_none((snap.prevDay or _DayBar()).c),
            bid=positive_or_none(quote.p),
            ask=positive_or_none(quote.P),
            bid_size=positive_or_none(quote.s),
            ask_size=positive_or_none(quote.S),
            day_open=positive_or_none(day.o),
            day_high=positive_or_none(day.h),
            day_low=positive_or_none(day.l),
            volume=finite_or_none(day.v),
            timestamp=epoch_to_datetime(stamp_ns, "ns") if stamp_ns else datetime.now(UTC),
        )

    async def quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        return {s: await self.quote(s) for s in symbols}

    async def history(self, symbol: str, interval: Interval, start: datetime, end: datetime) -> PriceHistory:
        mult, span = TIMESPAN[interval]
        url = f"{self._base}/v2/aggs/ticker/{vendor_symbol(symbol)}/range/{mult}/{span}/{int(start.timestamp() * 1000)}/{int(end.timestamp() * 1000)}"
        pages = await self._paged(url, {"adjusted": "true", "sort": "asc", "limit": 50000}, _AggResponse)
        bars: list[Bar] = []
        for page in pages:
            for a in page.results or []:
                bar = Bar.sanitized(epoch_to_datetime(a.t, "ms"), a.o, a.h, a.l, a.c, a.v)
                if bar is not None:
                    bars.append(bar)
        require(bool(bars), NAME, f"no aggregates for {symbol}")
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)

    async def option_chain(
        self, symbol: str, expirations: Sequence[date] | None, max_expirations: int = 8
    ) -> OptionChain:
        payload = await self._http.get_json(
            NAME,
            f"{self._base}/v3/reference/options/contracts",
            params=self._params(
                {
                    "underlying_ticker": vendor_symbol(symbol),
                    "expired": "false",
                    "limit": 1000,
                    "sort": "expiration_date",
                    "order": "asc",
                }
            ),
        )
        refs = parse_wire(NAME, _ContractsResponse, payload).results or []
        all_exp = sorted({r.expiration_date for r in refs})
        require(bool(all_exp), NAME, f"no listed options for {symbol}")
        wanted = [e for e in (expirations or all_exp[:max_expirations]) if e in all_exp]
        require(bool(wanted), NAME, "requested expirations are not listed")
        contracts: list[OptionContract] = []
        spot: float | None = None
        stamp: datetime | None = None
        for exp in wanted:
            pages = await self._paged(
                f"{self._base}/v3/snapshot/options/{vendor_symbol(symbol)}",
                {"expiration_date": exp.isoformat(), "limit": 250},
                _OptChainResponse,
            )
            for page in pages:
                for r in page.results or []:
                    if r.details.contract_type not in ("call", "put"):
                        continue
                    ua = r.underlying_asset
                    if ua is not None and positive_or_none(ua.price) and spot is None:
                        spot = ua.price
                        stamp = epoch_to_datetime(ua.last_updated, "ns") if ua.last_updated else None
                    q = r.last_quote or _OptQuote()
                    contracts.append(
                        OptionContract(
                            contract_symbol=r.details.ticker.removeprefix("O:"),
                            kind=r.details.contract_type,
                            strike=r.details.strike_price,
                            expiration=r.details.expiration_date,
                            bid=positive_or_none(q.bid),
                            ask=positive_or_none(q.ask),
                            last=positive_or_none((r.last_trade or _OptLastTrade()).price)
                            or positive_or_none((r.day or _OptDay()).close),
                            volume=finite_or_none((r.day or _OptDay()).volume),
                            open_interest=finite_or_none(r.open_interest),
                            implied_volatility=positive_or_none(r.implied_volatility),
                        )
                    )
        require(bool(contracts), NAME, f"empty option chain for {symbol}")
        if spot is None:
            spot = (await self.quote(symbol)).price
        return OptionChain(
            underlying=symbol,
            underlying_price=spot,
            as_of=stamp or datetime.now(UTC),
            expirations=all_exp,
            contracts=contracts,
        )
