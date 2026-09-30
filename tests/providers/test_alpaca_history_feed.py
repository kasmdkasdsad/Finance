"""Free full-market history: on Alpaca's free IEX plan, bars come from the consolidated (SIP) feed, which the
plan allows once the data is 15 minutes old. IEX bars carry only IEX's own few percent of the volume."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from quantpulse.core.errors import ProviderHTTPError
from quantpulse.providers.alpaca import SIP_HISTORY_DELAY, Alpaca

BAR = {"t": "2026-09-24T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 5_000_000}
ONE = "https://data.alpaca.markets/v2/stocks/AAPL/bars"
MULTI = "https://data.alpaca.markets/v2/stocks/bars"
REFUSED = {"message": "subscription does not permit querying recent SIP data"}


def _end(params: dict[str, str]) -> datetime:
    return datetime.fromisoformat(params["end"].replace("Z", "+00:00"))


@respx.mock
async def test_the_free_plan_reads_daily_bars_from_sip_held_back_16_minutes(http):
    seen: list[dict[str, str]] = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"bars": [BAR], "next_page_token": None})

    respx.get(ONE).mock(side_effect=handler)
    alpaca = Alpaca(http, "ID", "SECRET")  # the free plan's IEX feed
    now = datetime.now(UTC)

    await alpaca.history("AAPL", "1d", now - timedelta(days=30), now)
    assert seen[-1]["feed"] == "sip"
    assert timedelta(minutes=15) < now - _end(seen[-1]) <= SIP_HISTORY_DELAY  # today's bar as of 16 min ago

    old = now - timedelta(days=2)  # a window that ended long ago is read as asked
    await alpaca.history("AAPL", "1d", now - timedelta(days=30), old)
    assert seen[-1]["feed"] == "sip" and _end(seen[-1]) == old

    # Intraday: a live window stays real time on IEX; a finished one comes from SIP.
    await alpaca.history("AAPL", "1m", now - timedelta(hours=2), now)
    assert seen[-1]["feed"] == "iex" and _end(seen[-1]) == now
    await alpaca.history("AAPL", "1m", now - timedelta(days=3), old)
    assert seen[-1]["feed"] == "sip"
    assert alpaca.feed_status()["history_feed"] == "sip"


@respx.mock
async def test_a_paid_feed_is_read_as_configured(http):
    seen: list[dict[str, str]] = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"bars": [BAR], "next_page_token": None})

    respx.get(ONE).mock(side_effect=handler)
    now = datetime.now(UTC)
    await Alpaca(http, "ID", "SECRET", stock_feed="sip").history("AAPL", "1d", now - timedelta(days=30), now)
    assert seen[-1]["feed"] == "sip" and _end(seen[-1]) == now  # real time: nothing held back


@respx.mock
async def test_a_refused_sip_history_falls_back_to_iex_and_is_remembered(http):
    feeds: list[str] = []

    def handler(request):
        feeds.append(request.url.params["feed"])
        if request.url.params["feed"] == "sip":
            return httpx.Response(403, json=REFUSED)
        return httpx.Response(200, json={"bars": {"AAPL": [BAR]}})

    respx.get(MULTI).mock(side_effect=handler)
    alpaca = Alpaca(http, "ID", "SECRET")
    end = datetime.now(UTC)
    out = await alpaca.histories(["AAPL"], "1d", end - timedelta(days=30), end)
    assert set(out) == {"AAPL"} and feeds == ["sip", "iex"]

    await alpaca.histories(["AAPL"], "1d", end - timedelta(days=30), end)
    assert feeds == ["sip", "iex", "iex"]  # not asked again by this process
    status = alpaca.feed_status()
    assert status["history_feed"] == "iex" and status["history_feed_refused"] == 403


@respx.mock
async def test_an_error_that_is_not_the_feeds_never_switches_it(http):
    respx.get(ONE).mock(return_value=httpx.Response(403, json={"message": "forbidden"}))  # e.g. a revoked key
    alpaca = Alpaca(http, "ID", "SECRET")
    now = datetime.now(UTC)
    with pytest.raises(ProviderHTTPError):
        await alpaca.history("AAPL", "1d", now - timedelta(days=30), now)
    assert alpaca.history_feed == "sip"  # IEX failed too: the refusal was not SIP's

    respx.get(ONE).mock(return_value=httpx.Response(400, json={"message": "invalid symbol"}))
    with pytest.raises(ProviderHTTPError):
        await alpaca.history("AAPL", "1d", now - timedelta(days=30), now)
    assert alpaca.history_feed == "sip"
