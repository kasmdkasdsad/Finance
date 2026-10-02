"""Alpaca options market data — the first :class:`~quantpulse.options.data.OptionsMarketDataProvider`.

* contracts: the paper trading API's ``/v2/options/contracts`` (status, tradability, open interest) — the
  paper host only, as everywhere in QuantPulse;
* chains: ``/v1beta1/options/snapshots/{underlying}`` (latest quote and trade, implied volatility, Greeks);
* latest quotes and trades, daily bars, and the live stream (``wss://…/v1beta1/{feed}``, MessagePack).

The free ``indicative`` feed is derived from OPRA and not firm; ``opra`` (a paid subscription) is the official
consolidated feed. The feed name travels with every quote. Quotes whose time Alpaca did not send carry no
time — their age is unknown, never invented.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any

from pydantic import Field

from quantpulse.core.errors import ProviderNotConfigured
from quantpulse.core.http import HttpClient
from quantpulse.options.contracts import ContractError, OptionContract, parse_occ
from quantpulse.options.data import ChainSnapshot, ContractInfo, OptionBar, OptionTrade
from quantpulse.options.quotes import Greeks, OptionQuote
from quantpulse.providers.alpaca_trading import PAPER_URL
from quantpulse.providers.base import WireModel, finite_or_none, parse_wire, positive_or_none

logger = logging.getLogger(__name__)
NAME = "alpaca"
MAX_PAGES = 20
STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/{feed}"


class _Trade(WireModel):
    t: datetime | None = None
    p: float | None = None
    s: float | None = None
    x: str | None = None


class _Quote(WireModel):
    t: datetime | None = None
    bp: float | None = None
    ap: float | None = None
    bs: float | None = None
    as_: float | None = Field(default=None, alias="as")


class _Bar(WireModel):
    t: datetime
    o: float | None = None
    h: float | None = None
    l: float | None = None  # noqa: E741
    c: float | None = None
    v: float | None = None


class _Greeks(WireModel):
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None


class _Snapshot(WireModel):
    latestQuote: _Quote | None = None
    latestTrade: _Trade | None = None
    impliedVolatility: float | None = None
    greeks: _Greeks | None = None
    dailyBar: _Bar | None = None


class _Snapshots(WireModel):
    snapshots: dict[str, _Snapshot] | None = None
    next_page_token: str | None = None


class _Contract(WireModel):
    symbol: str
    status: str | None = None
    tradable: bool | None = None
    open_interest: float | str | None = None
    open_interest_date: date | None = None
    close_price: float | str | None = None
    multiplier: float | str | None = None
    style: str | None = None


class _Contracts(WireModel):
    option_contracts: list[_Contract] | None = None
    next_page_token: str | None = None


class _Quotes(WireModel):
    quotes: dict[str, _Quote | None] | None = None


class _Trades(WireModel):
    trades: dict[str, _Trade | None] | None = None


class _Bars(WireModel):
    bars: dict[str, list[_Bar] | None] | None = None
    next_page_token: str | None = None


def _num(x: float | str | None) -> float | None:
    try:
        return finite_or_none(float(x)) if x is not None else None
    except (TypeError, ValueError):
        return None


class AlpacaOptionsProvider:
    name = NAME

    def __init__(
        self,
        http: HttpClient,
        key_id: str | None,
        secret: str | None,
        *,
        data_url: str = "https://data.alpaca.markets",
        feed: str = "indicative",
        underlying_quote: Callable[[str], Awaitable[tuple[float, datetime | None]]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = http
        self._key = key_id
        self._secret = secret
        self._data = data_url.rstrip("/")
        self._feed = feed
        self._underlying_quote = underlying_quote
        self._now = clock or (lambda: datetime.now(UTC))

    def configured(self) -> bool:
        return bool(self._key and self._secret)

    @property
    def feed(self) -> str:
        return self._feed

    def _headers(self) -> dict[str, str]:
        if not self.configured():
            raise ProviderNotConfigured(NAME, "QP_ALPACA_API_KEY_ID / QP_ALPACA_API_SECRET_KEY not set")
        return {"APCA-API-KEY-ID": self._key or "", "APCA-API-SECRET-KEY": self._secret or ""}

    # ------------------------------------------------------------------ contracts (the paper trading API)
    async def contracts(
        self,
        underlying: str,
        *,
        expiration_from: date | None = None,
        expiration_to: date | None = None,
        strike_from: float | None = None,
        strike_to: float | None = None,
    ) -> list[ContractInfo]:
        params: dict[str, Any] = {"underlying_symbols": underlying.upper(), "limit": 10000}
        if expiration_from:
            params["expiration_date_gte"] = expiration_from.isoformat()
        if expiration_to:
            params["expiration_date_lte"] = expiration_to.isoformat()
        if strike_from is not None:
            params["strike_price_gte"] = f"{strike_from:g}"
        if strike_to is not None:
            params["strike_price_lte"] = f"{strike_to:g}"
        out: list[ContractInfo] = []
        for _ in range(MAX_PAGES):
            payload = await self._http.get_json(NAME, f"{PAPER_URL}/v2/options/contracts", params=params,
                                                headers=self._headers())  # fmt: skip
            page = parse_wire(NAME, _Contracts, payload)
            for c in page.option_contracts or []:
                try:
                    contract = parse_occ(c.symbol)
                except ContractError:
                    continue
                mult = _num(c.multiplier)
                if mult and int(mult) != contract.multiplier:  # adjusted contracts (after a corporate action)
                    contract = OptionContract(contract.underlying, contract.expiration, contract.kind,
                                              contract.strike, multiplier=int(mult))  # fmt: skip
                out.append(ContractInfo(contract, bool(c.tradable), c.status or "unknown", _num(c.open_interest),
                                        c.open_interest_date, _num(c.close_price)))  # fmt: skip
            if not page.next_page_token:
                break
            params["page_token"] = page.next_page_token
        return out

    # ------------------------------------------------------------------ market data
    async def chain(
        self, underlying: str, *, expiration_from: date | None = None, expiration_to: date | None = None
    ) -> ChainSnapshot:
        params: dict[str, Any] = {"feed": self._feed, "limit": 1000}
        if expiration_from:
            params["expiration_date_gte"] = expiration_from.isoformat()
        if expiration_to:
            params["expiration_date_lte"] = expiration_to.isoformat()
        snaps: dict[str, _Snapshot] = {}
        for _ in range(MAX_PAGES):
            payload = await self._http.get_json(NAME, f"{self._data}/v1beta1/options/snapshots/{underlying.upper()}",
                                                params=params, headers=self._headers())  # fmt: skip
            page = parse_wire(NAME, _Snapshots, payload)
            snaps.update(page.snapshots or {})
            if not page.next_page_token:
                break
            params["page_token"] = page.next_page_token
        spot, spot_at = await self._spot(underlying)
        quotes = []
        for occ, s in snaps.items():
            q = self._quote(occ, s.latestQuote, s, spot, spot_at)
            if q is not None:
                quotes.append(q)
        quotes.sort(key=lambda q: (q.contract.expiration, q.contract.strike, q.contract.kind))
        notes = [] if self._feed == "opra" else ["indicative feed: derived from OPRA, not firm"]
        return ChainSnapshot(underlying.upper(), spot, spot_at, self._now(), self._feed, NAME, quotes, notes)

    async def _spot(self, underlying: str) -> tuple[float, datetime | None]:
        if self._underlying_quote is None:
            raise ProviderNotConfigured(NAME, "no underlying quote source")
        return await self._underlying_quote(underlying)

    def _quote(self, occ: str, q: _Quote | None, s: _Snapshot | None, spot: float | None,
               spot_at: datetime | None) -> OptionQuote | None:  # fmt: skip
        try:
            contract = parse_occ(occ)
        except ContractError:
            return None
        g = s.greeks if s is not None else None
        trade = s.latestTrade if s is not None else None
        return OptionQuote(
            contract=contract,
            bid=positive_or_none(q.bp) if q else None,
            ask=positive_or_none(q.ap) if q else None,
            quote_at=q.t if q else None,
            feed=self._feed if self._feed in ("opra", "indicative") else "unknown",  # type: ignore[arg-type]
            source=NAME,
            bid_size=finite_or_none(q.bs) if q else None,
            ask_size=finite_or_none(q.as_) if q else None,
            last=positive_or_none(trade.p) if trade else None,
            last_at=trade.t if trade else None,
            volume=finite_or_none(s.dailyBar.v) if s is not None and s.dailyBar else None,
            iv=positive_or_none(s.impliedVolatility) if s is not None else None,
            greeks=Greeks(
                delta=finite_or_none(g.delta), gamma=finite_or_none(g.gamma), theta=finite_or_none(g.theta),
                vega=finite_or_none(g.vega), rho=finite_or_none(g.rho), source="vendor",
            ) if g is not None else Greeks(),
            underlying_price=spot,
            underlying_at=spot_at,
        )  # fmt: skip

    async def latest_quotes(self, symbols: Sequence[str]) -> dict[str, OptionQuote]:
        if not symbols:
            return {}
        payload = await self._http.get_json(NAME, f"{self._data}/v1beta1/options/quotes/latest",
                                            params={"symbols": ",".join(symbols), "feed": self._feed},
                                            headers=self._headers())  # fmt: skip
        page = parse_wire(NAME, _Quotes, payload)
        out: dict[str, OptionQuote] = {}
        for occ, q in (page.quotes or {}).items():
            quote = self._quote(occ, q, None, None, None)
            if quote is not None:
                out[occ] = quote
        return out

    async def latest_trades(self, symbols: Sequence[str]) -> dict[str, OptionTrade]:
        if not symbols:
            return {}
        payload = await self._http.get_json(NAME, f"{self._data}/v1beta1/options/trades/latest",
                                            params={"symbols": ",".join(symbols), "feed": self._feed},
                                            headers=self._headers())  # fmt: skip
        page = parse_wire(NAME, _Trades, payload)
        return {
            occ: OptionTrade(occ, t.t, float(t.p or 0), float(t.s or 0), t.x)
            for occ, t in (page.trades or {}).items()
            if t is not None and t.t is not None and positive_or_none(t.p)
        }

    async def bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, list[OptionBar]]:
        params: dict[str, Any] = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start.isoformat(),
                                  "end": end.isoformat(), "limit": 10000}  # fmt: skip
        out: dict[str, list[OptionBar]] = {}
        for _ in range(MAX_PAGES):
            payload = await self._http.get_json(NAME, f"{self._data}/v1beta1/options/bars", params=params,
                                                headers=self._headers())  # fmt: skip
            page = parse_wire(NAME, _Bars, payload)
            for occ, rows in (page.bars or {}).items():
                for b in rows or []:
                    if None in (b.o, b.h, b.l, b.c):
                        continue
                    out.setdefault(occ, []).append(OptionBar(occ, b.t, b.o, b.h, b.l, b.c, b.v or 0.0))  # type: ignore[arg-type]
            if not page.next_page_token:
                break
            params["page_token"] = page.next_page_token
        return out

    # ------------------------------------------------------------------ streaming
    async def stream(
        self,
        symbols: Sequence[str],
        *,
        connect: Callable[[str], Any] | None = None,
    ) -> AsyncIterator[OptionQuote | OptionTrade]:
        """Live quotes and trades. Alpaca's option stream speaks MessagePack only. ``connect`` is injectable
        (tests); by default a ``websockets`` connection to the feed's URL."""
        import msgpack

        if connect is None:
            import websockets

            connect = websockets.connect
        headers = self._headers()
        async with connect(STREAM_URL.format(feed=self._feed)) as ws:
            await ws.send(msgpack.packb({"action": "auth", "key": headers["APCA-API-KEY-ID"],
                                         "secret": headers["APCA-API-SECRET-KEY"]}))  # fmt: skip
            await ws.send(
                msgpack.packb({"action": "subscribe", "quotes": list(symbols), "trades": list(symbols)})
            )
            async for raw in ws:
                for msg in msgpack.unpackb(raw, timestamp=3) if isinstance(raw, bytes) else []:
                    kind = msg.get("T")
                    if kind == "error":
                        raise ConnectionError(f"Alpaca stream error {msg.get('code')}: {msg.get('msg')}")
                    if kind == "q":
                        q = _Quote(t=_ts(msg.get("t")), bp=msg.get("bp"), ap=msg.get("ap"), bs=msg.get("bs"),
                                   **{"as": msg.get("as")})  # fmt: skip
                        quote = self._quote(str(msg.get("S")), q, None, None, None)
                        if quote is not None:
                            yield quote
                    elif kind == "t" and positive_or_none(msg.get("p")) and msg.get("t") is not None:
                        yield OptionTrade(str(msg.get("S")), _ts(msg.get("t")), float(msg["p"]),  # type: ignore[arg-type]
                                          float(msg.get("s") or 0), msg.get("x"))  # fmt: skip


def _ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if hasattr(value, "to_datetime"):  # a msgpack Timestamp
        return value.to_datetime()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
