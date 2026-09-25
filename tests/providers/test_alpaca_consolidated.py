"""Alpaca market data: bid/ask timestamps and the consolidated (SIP) quote used to measure spreads."""

import httpx
import respx

from quantpulse.providers.alpaca import Alpaca

SNAPSHOT = {
    "DELL": {
        "latestTrade": {"t": "2026-09-25T15:00:01Z", "p": 120.0},
        "latestQuote": {"t": "2026-09-25T14:58:00Z", "bp": 114.0, "ap": 126.0, "bs": 1, "as": 2},
        "dailyBar": {
            "t": "2026-09-25T04:00:00Z",
            "o": 119,
            "h": 121,
            "l": 118,
            "c": 120,
            "v": 1000,
            "vw": 119.8,
        },
        "prevDailyBar": {"t": "2026-09-24T04:00:00Z", "o": 118, "h": 120, "l": 117, "c": 119, "v": 900},
    }
}
LATEST = {"quotes": {"DELL": {"t": "2026-09-25T14:45:00Z", "bp": 119.98, "ap": 120.02}, "BRK.B": None}}


@respx.mock
async def test_snapshots_keep_the_bid_ask_time_and_the_feed(http):
    respx.get("https://data.alpaca.markets/v2/stocks/snapshots").mock(
        return_value=httpx.Response(200, json=SNAPSHOT)
    )
    q = (await Alpaca(http, "ID", "SECRET").quotes(["DELL"]))["DELL"]
    assert q.feed == "iex" and q.timestamp.minute == 0 and q.quote_timestamp.minute == 58
    assert (q.bid, q.ask) == (114.0, 126.0)


@respx.mock
async def test_consolidated_quotes_fall_back_to_the_delayed_feed_and_remember_refusals(http):
    feeds = []

    def handler(request):
        feed = request.url.params["feed"]
        feeds.append(feed)
        if feed == "sip":
            return httpx.Response(
                403, json={"message": "subscription does not permit querying recent SIP data"}
            )
        return httpx.Response(200, json=LATEST)

    respx.get("https://data.alpaca.markets/v2/stocks/quotes/latest").mock(side_effect=handler)
    alpaca = Alpaca(http, "ID", "SECRET")
    got = await alpaca.consolidated_quotes(["DELL", "BRK-B"])
    assert set(got) == {"DELL"} and got["DELL"].feed == "delayed_sip"
    assert (got["DELL"].bid, got["DELL"].ask) == (119.98, 120.02)
    assert feeds == ["sip", "delayed_sip"]
    await alpaca.consolidated_quotes(["DELL"])
    assert feeds == ["sip", "delayed_sip", "delayed_sip"]  # the refused real-time feed is not retried at once


@respx.mock
async def test_no_consolidated_feed_means_no_consolidated_quote(http):
    respx.get("https://data.alpaca.markets/v2/stocks/quotes/latest").mock(
        return_value=httpx.Response(403, json={"message": "forbidden"})
    )
    assert await Alpaca(http, "ID", "SECRET").consolidated_quotes(["DELL"]) == {}
    assert await Alpaca(http, None, None).consolidated_quotes(["DELL"]) == {}
