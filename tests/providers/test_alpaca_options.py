"""The Alpaca options adapter against recorded-shape responses (respx): contracts from the PAPER trading host,
chains with quotes, trades, IV and Greeks, latest quotes and trades, bars and the MessagePack stream."""

from datetime import UTC, date, datetime

import httpx
import msgpack
import pytest
import respx

from quantpulse.core.errors import ProviderNotConfigured
from quantpulse.providers.alpaca_options import AlpacaOptionsProvider

NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
DATA = "https://data.alpaca.markets"
PAPER = "https://paper-api.alpaca.markets"


async def spot(_symbol):
    return 100.0, NOW


def provider(http, **kw):
    return AlpacaOptionsProvider(http, "PKTEST", "SECRET", underlying_quote=spot, clock=lambda: NOW, **kw)


SNAP = {
    "snapshots": {
        "AAPL261016C00100000": {
            "latestQuote": {"t": "2026-09-25T13:59:58Z", "bp": 3.1, "ap": 3.2, "bs": 12, "as": 9},
            "latestTrade": {"t": "2026-09-25T13:59:00Z", "p": 3.15, "s": 2, "x": "C"},
            "impliedVolatility": 0.31,
            "greeks": {"delta": 0.52, "gamma": 0.04, "theta": -0.06, "vega": 0.12, "rho": 0.03},
            "dailyBar": {"t": "2026-09-25T04:00:00Z", "o": 3, "h": 3.4, "l": 2.9, "c": 3.15, "v": 812},
        },
        "AAPL261016P00100000": {"latestQuote": {"bp": 2.9, "ap": 3.0}},  # no timestamp: age unknown
        "NOT-AN-OCC": {"latestQuote": {"bp": 1, "ap": 2}},
    },
    "next_page_token": None,
}


@respx.mock
async def test_a_chain_carries_quotes_greeks_feed_and_times(http):
    route = respx.get(f"{DATA}/v1beta1/options/snapshots/AAPL").respond(200, json=SNAP)
    chain = await provider(http).chain("aapl", expiration_from=date(2026, 10, 1))
    assert route.calls[0].request.url.params["feed"] == "indicative"
    assert route.calls[0].request.url.params["expiration_date_gte"] == "2026-10-01"
    assert chain.underlying == "AAPL" and chain.feed == "indicative" and chain.source == "alpaca"
    assert [q.symbol for q in chain.quotes] == ["AAPL261016C00100000", "AAPL261016P00100000"]
    c, p = chain.quotes
    assert (
        c.bid == 3.1
        and c.ask == 3.2
        and c.iv == 0.31
        and c.greeks.delta == 0.52
        and c.greeks.source == "vendor"
    )
    assert c.volume == 812 and c.last == 3.15 and c.quote_at == datetime(2026, 9, 25, 13, 59, 58, tzinfo=UTC)
    assert p.quote_at is None and p.greeks.delta is None  # never invented
    q = chain.quality(NOW)
    assert q["OPTIONS_DATA_FEED"] == "indicative" and q["OPTIONS_DATA_SOURCE"] == "alpaca"
    assert q["usable_for_execution"] == 1 and q["OPTIONS_DATA_QUALITY"] == "execution"
    assert "indicative" in chain.notes[0]


@respx.mock
async def test_contracts_come_from_the_paper_host_only_and_are_paged(http):
    page1 = {"option_contracts": [{"symbol": "AAPL261016C00100000", "status": "active", "tradable": True,
                                   "open_interest": "1520", "open_interest_date": "2026-09-24", "close_price": "3.10",
                                   "multiplier": "100"}], "next_page_token": "p2"}  # fmt: skip
    page2 = {"option_contracts": [{"symbol": "AAPL261016P00095000", "status": "inactive", "tradable": False},
                                  {"symbol": "garbage"}], "next_page_token": None}  # fmt: skip
    route = respx.get(f"{PAPER}/v2/options/contracts").mock(
        side_effect=[httpx.Response(200, json=page1), httpx.Response(200, json=page2)]
    )
    out = await provider(http).contracts(
        "AAPL", expiration_from=date(2026, 10, 1), strike_from=90, strike_to=110
    )
    assert [c.contract.symbol for c in out] == ["AAPL261016C00100000", "AAPL261016P00095000"]
    assert out[0].open_interest == 1520 and out[0].tradable and not out[1].tradable
    assert route.calls[0].request.url.host == "paper-api.alpaca.markets"
    assert route.calls[0].request.url.params["strike_price_gte"] == "90"
    assert route.calls[1].request.url.params["page_token"] == "p2"
    assert route.calls[0].request.headers["APCA-API-KEY-ID"] == "PKTEST"


@respx.mock
async def test_latest_quotes_trades_and_bars(http):
    respx.get(f"{DATA}/v1beta1/options/quotes/latest").respond(
        200, json={"quotes": {"AAPL261016C00100000": {"t": "2026-09-25T13:59:59Z", "bp": 3.0, "ap": 3.1}}}
    )
    respx.get(f"{DATA}/v1beta1/options/trades/latest").respond(
        200,
        json={
            "trades": {
                "AAPL261016C00100000": {"t": "2026-09-25T13:59:00Z", "p": 3.05, "s": 1},
                "AAPL261016P00100000": {"p": 1.0},
            }
        },
    )
    respx.get(f"{DATA}/v1beta1/options/bars").respond(
        200,
        json={
            "bars": {
                "AAPL261016C00100000": [
                    {"t": "2026-09-24T04:00:00Z", "o": 3, "h": 3.3, "l": 2.8, "c": 3.1, "v": 500},
                    {"t": "2026-09-23T04:00:00Z", "o": None},
                ]
            }
        },
    )
    p = provider(http, feed="opra")
    quotes = await p.latest_quotes(["AAPL261016C00100000"])
    assert quotes["AAPL261016C00100000"].feed == "opra" and quotes[
        "AAPL261016C00100000"
    ].mid == pytest.approx(3.05)
    trades = await p.latest_trades(["AAPL261016C00100000", "AAPL261016P00100000"])
    assert list(trades) == ["AAPL261016C00100000"]  # a trade without a time is not a trade
    bars = await p.bars(["AAPL261016C00100000"], date(2026, 9, 1), date(2026, 9, 25))
    assert len(bars["AAPL261016C00100000"]) == 1 and bars["AAPL261016C00100000"][0].close == 3.1
    assert await p.latest_quotes([]) == {}


async def test_without_keys_nothing_is_requested(http):
    p = AlpacaOptionsProvider(http, None, None, underlying_quote=spot)
    assert not p.configured()
    with pytest.raises(ProviderNotConfigured):
        await p.chain("AAPL")


class FakeSocket:
    """A websocket that answers with Alpaca-shaped MessagePack frames."""

    def __init__(self, frames):
        self.sent: list[dict] = []
        self._frames = frames

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, data):
        self.sent.append(msgpack.unpackb(data))

    def __aiter__(self):
        async def gen():
            for f in self._frames:
                yield msgpack.packb(f, datetime=True)

        return gen()


async def test_the_stream_authenticates_subscribes_and_decodes(http):
    ts = datetime(2026, 9, 25, 14, 0, 1, tzinfo=UTC)
    sock = FakeSocket([
        [{"T": "success", "msg": "authenticated"}],
        [{"T": "q", "S": "AAPL261016C00100000", "t": ts, "bp": 3.0, "ap": 3.1, "bs": 5, "as": 7},
         {"T": "t", "S": "AAPL261016C00100000", "t": ts, "p": 3.05, "s": 2, "x": "C"}],
    ])  # fmt: skip
    seen = [m async for m in provider(http).stream(["AAPL261016C00100000"], connect=lambda url: sock)]
    assert sock.sent[0]["action"] == "auth" and sock.sent[1] == {"action": "subscribe", "quotes": ["AAPL261016C00100000"],
                                                                 "trades": ["AAPL261016C00100000"]}  # fmt: skip
    quote, trade = seen
    assert quote.bid == 3.0 and quote.ask_size == 7 and quote.quote_at == ts
    assert trade.price == 3.05 and trade.at == ts


async def test_a_stream_error_is_raised_not_swallowed(http):
    sock = FakeSocket([[{"T": "error", "code": 402, "msg": "auth failed"}]])
    with pytest.raises(ConnectionError, match="auth failed"):
        async for _ in provider(http).stream(["X"], connect=lambda url: sock):
            pass
