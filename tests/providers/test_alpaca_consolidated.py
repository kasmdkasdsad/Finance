"""Alpaca market data: bid/ask timestamps and the consolidated (SIP) quote used to measure spreads."""

import httpx
import pytest
import respx

from quantpulse.core.errors import ProviderHTTPError
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


@respx.mock
async def test_a_price_without_a_timestamp_is_never_stamped_now(http):
    snaps = {
        "OLD": {  # no trade time: the daily bar's close and its own time are used, never "now"
            "latestTrade": {"p": 50.0},
            "dailyBar": {"t": "2026-09-25T04:00:00Z", "o": 49, "h": 51, "l": 48, "c": 49.5, "v": 10},
        },
        "NONE": {"latestTrade": {"p": 50.0}},  # a price with no time at all is not a quote
        "DELL": SNAPSHOT["DELL"],
    }
    respx.get("https://data.alpaca.markets/v2/stocks/snapshots").mock(
        return_value=httpx.Response(200, json=snaps)
    )
    got = await Alpaca(http, "ID", "SECRET").quotes(["OLD", "NONE", "DELL"])
    assert "NONE" not in got
    assert got["OLD"].price == 49.5 and got["OLD"].timestamp.isoformat() == "2026-09-25T04:00:00+00:00"


@respx.mock
async def test_feed_status_records_what_the_subscription_refused(http):
    respx.get("https://data.alpaca.markets/v2/stocks/snapshots").mock(
        return_value=httpx.Response(403, json={"message": "subscription does not permit querying SIP"})
    )
    respx.get("https://data.alpaca.markets/v2/stocks/quotes/latest").mock(
        return_value=httpx.Response(403, json={"message": "forbidden"})
    )
    alpaca = Alpaca(http, "ID", "SECRET", stock_feed="sip")
    with pytest.raises(ProviderHTTPError):
        await alpaca.quotes(["DELL"])
    await alpaca.consolidated_quotes(["DELL"])
    status = alpaca.feed_status()
    assert status["stock_feed"] == "sip" and status["stock_feed_error"]["status"] == 403
    assert set(status["refused_feeds"]) == {"sip", "delayed_sip"}
    assert "SECRET" not in str(status) and "ID" not in str(status).replace("provider", "")
