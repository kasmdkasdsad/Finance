"""Yahoo, Polygon and Alpaca adapters against mocked vendor payloads (documented response shapes)."""

from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from quantpulse.core.errors import ProviderHTTPError, ProviderNotConfigured, ProviderParseError
from quantpulse.providers.alpaca import Alpaca
from quantpulse.providers.base import occ_parse
from quantpulse.providers.polygon import Polygon
from quantpulse.providers.yahoo import YahooFinance

T = 1790280000  # 2026-09-24T20:00:00Z
EXP1, EXP2 = 1791417600, 1792627200  # 2026-10-08 / 2026-10-22 00:00 UTC


def chart_payload(price=252.31, prev=249.87, closes=(250.0, None, 252.31), adj=None):
    n = len(closes)
    quote = {
        "open": [c and c - 1 for c in closes],
        "high": [c and c + 2 for c in closes],
        "low": [c and c - 2 for c in closes],
        "close": list(closes),
        "volume": [1_000_000] * n,
    }
    indicators = {"quote": [quote]}
    if adj is not None:
        indicators["adjclose"] = [{"adjclose": adj}]
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": "AAPL",
                        "fullExchangeName": "NasdaqGS",
                        "regularMarketPrice": price,
                        "regularMarketTime": T,
                        "chartPreviousClose": prev,
                        "regularMarketDayHigh": 253.1,
                        "regularMarketDayLow": 248.9,
                        "regularMarketVolume": 41e6,
                        "longName": "Apple Inc.",
                    },
                    "timestamp": [T - 86400 * (n - 1 - i) for i in range(n)],
                    "indicators": indicators,
                    "events": {
                        "dividends": {
                            "1": {"amount": 0.26, "date": T - 86400 * 30},
                            "2": {"amount": 0.26, "date": T - 86400 * 400},
                        }
                    },
                }
            ],
            "error": None,
        }
    }


def options_payload(expiration):
    def contract(kind, strike, bid, ask):
        return {
            "contractSymbol": f"AAPL261008{kind}{int(strike * 1000):08d}",
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "lastPrice": (bid + ask) / 2,
            "volume": 10,
            "openInterest": 100,
            "impliedVolatility": 0.3,
            "inTheMoney": False,
            "lastTradeDate": T,
            "expiration": expiration,
        }

    return {
        "optionChain": {
            "result": [
                {
                    "underlyingSymbol": "AAPL",
                    "expirationDates": [EXP1, EXP2],
                    "quote": {"regularMarketPrice": 252.31, "regularMarketTime": T},
                    "options": [
                        {
                            "expirationDate": expiration,
                            "calls": [contract("C", 260, 2.0, 2.1)],
                            "puts": [contract("P", 240, 1.5, 1.6)],
                        }
                    ],
                }
            ],
            "error": None,
        }
    }


def mock_crumb(ok=True):
    respx.get("https://fc.yahoo.com").mock(
        return_value=httpx.Response(404, headers={"set-cookie": "A3=abc; Domain=.yahoo.com; Path=/"})
    )
    respx.get("https://query1.finance.yahoo.com/v1/test/getcrumb").mock(
        return_value=httpx.Response(200, text="Xy.Crumb1")
        if ok
        else httpx.Response(429, text="Too Many Requests")
    )


# ----------------------------------------------------------------------------- Yahoo
@respx.mock
async def test_yahoo_batch_quotes_with_crumb(http):
    mock_crumb()
    route = respx.get("https://query1.finance.yahoo.com/v7/finance/quote").mock(
        return_value=httpx.Response(
            200,
            json={
                "quoteResponse": {
                    "result": [
                        {
                            "symbol": "AAPL",
                            "regularMarketPrice": 252.31,
                            "regularMarketPreviousClose": 249.87,
                            "bid": 0,
                            "ask": 252.4,
                            "regularMarketTime": T,
                            "marketCap": 3.7e12,
                            "sharesOutstanding": 1.46e10,
                            "trailingAnnualDividendYield": 0.0041,
                            "longName": "Apple Inc.",
                            "currency": "USD",
                        }
                    ],
                    "error": None,
                }
            },
        )
    )
    q = await YahooFinance(http).quote("AAPL")
    assert q.price == 252.31 and q.previous_close == 249.87
    assert q.bid is None and q.ask == 252.4  # zero bid (closed market) becomes None
    assert q.dividend_yield == pytest.approx(0.0041)
    assert q.timestamp == datetime(2026, 9, 24, 20, 0, tzinfo=UTC)
    assert route.calls[0].request.url.params["crumb"] == "Xy.Crumb1"


@respx.mock
async def test_yahoo_falls_back_to_keyless_chart_when_crumb_blocked(http):
    mock_crumb(ok=False)
    respx.get(host="query1.finance.yahoo.com", path="/v8/finance/chart/AAPL").mock(
        return_value=httpx.Response(200, json=chart_payload())
    )
    q = await YahooFinance(http).quote("AAPL")
    assert q.price == 252.31 and q.previous_close == 249.87 and q.name == "Apple Inc."


@respx.mock
async def test_yahoo_history_adjusts_whole_candle_and_skips_nulls(http):
    respx.get(host="query1.finance.yahoo.com", path="/v8/finance/chart/AAPL").mock(
        return_value=httpx.Response(
            200, json=chart_payload(closes=(100.0, None, 110.0), adj=[99.0, None, 110.0])
        )
    )
    h = await YahooFinance(http).history(
        "AAPL", "1d", datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 25, tzinfo=UTC)
    )
    assert len(h.bars) == 2
    first = h.bars[0]
    assert first.close == pytest.approx(99.0)
    assert first.open == pytest.approx(99.0 * 99.0 / 100.0)  # open scaled by adjclose/close
    assert first.high >= max(first.open, first.close)


@respx.mock
async def test_yahoo_dividend_yield_uses_trailing_twelve_months(http):
    respx.get(host="query1.finance.yahoo.com", path="/v8/finance/chart/AAPL").mock(
        return_value=httpx.Response(200, json=chart_payload())
    )
    import quantpulse.providers.yahoo as y

    y_now = datetime.fromtimestamp(T, UTC)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return y_now

    y.datetime = _DT  # freeze "now" inside the module
    try:
        dy = await YahooFinance(http).dividend_yield("AAPL")
    finally:
        y.datetime = datetime
    assert dy == pytest.approx(0.26 / 252.31)  # only the dividend inside the last 365 days


@respx.mock
async def test_yahoo_option_chain_multiple_expirations_and_crumb_refresh(http):
    mock_crumb()
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                401, json={"finance": {"error": {"code": "Unauthorized", "description": "Invalid Crumb"}}}
            )
        exp = int(request.url.params.get("date", EXP1))
        return httpx.Response(200, json=options_payload(exp))

    respx.get(host="query2.finance.yahoo.com", path="/v7/finance/options/AAPL").mock(side_effect=handler)
    chain = await YahooFinance(http).option_chain("AAPL", None, max_expirations=2)
    assert chain.expirations == [date(2026, 10, 8), date(2026, 10, 22)]
    assert {c.expiration for c in chain.contracts} == {date(2026, 10, 8), date(2026, 10, 22)}
    assert chain.underlying_price == 252.31
    assert {c.kind for c in chain.contracts} == {"call", "put"}


@respx.mock
async def test_yahoo_estimates_parse_trend_and_targets(http):
    mock_crumb()
    raw = lambda v: {"raw": v, "fmt": str(v)}
    respx.get(host="query2.finance.yahoo.com", path="/v10/finance/quoteSummary/AAPL").mock(
        return_value=httpx.Response(
            200,
            json={
                "quoteSummary": {
                    "result": [
                        {
                            "financialData": {
                                "targetMeanPrice": raw(270.5),
                                "numberOfAnalystOpinions": raw(41),
                                "recommendationKey": "buy",
                                "recommendationMean": raw(1.9),
                            },
                            "defaultKeyStatistics": {"beta": raw(1.21)},
                            "earningsTrend": {
                                "trend": [
                                    {
                                        "period": "0q",
                                        "endDate": "2026-09-30",
                                        "revenueEstimate": {"avg": raw(1.0e11)},
                                    },
                                    {
                                        "period": "0y",
                                        "endDate": "2026-09-30",
                                        "revenueEstimate": {
                                            "avg": raw(4.4e11),
                                            "numberOfAnalysts": raw(38),
                                            "growth": raw(0.06),
                                        },
                                        "earningsEstimate": {"avg": raw(8.1)},
                                    },
                                    {
                                        "period": "+1y",
                                        "endDate": "2027-09-30",
                                        "revenueEstimate": {"avg": raw(4.7e11), "growth": raw(0.07)},
                                    },
                                    {"period": "+5y", "growth": raw(0.09)},
                                    {"period": "-5y", "growth": {}},
                                ]
                            },
                        }
                    ],
                    "error": None,
                }
            },
        )
    )
    est = await YahooFinance(http).estimates("AAPL")
    assert est.target_mean_price == 270.5 and est.analyst_count == 41 and est.beta == 1.21
    assert est.long_term_growth == 0.09
    assert [p.period for p in est.periods] == ["0q", "0y", "+1y"]
    assert est.periods[1].analysts == 38


@respx.mock
async def test_yahoo_schema_drift_is_a_parse_error(http):
    respx.get(host="query1.finance.yahoo.com", path="/v8/finance/chart/AAPL").mock(
        return_value=httpx.Response(200, json={"chart": {"result": [{"meta": {"no_symbol": True}}]}})
    )
    with pytest.raises(ProviderParseError):
        await YahooFinance(http).history(
            "AAPL", "1d", datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 25, tzinfo=UTC)
        )


# ----------------------------------------------------------------------------- Polygon
@respx.mock
async def test_polygon_requires_key(http):
    with pytest.raises(ProviderNotConfigured):
        await Polygon(http, None).quote("AAPL")


@respx.mock
async def test_polygon_snapshot_and_class_share_symbol(http):
    route = respx.get("https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/BRK.B").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "OK",
                "ticker": {
                    "ticker": "BRK.B",
                    "updated": T * 10**9,
                    "day": {"o": 480, "h": 485, "l": 478, "c": 483, "v": 3e6},
                    "prevDay": {"c": 479.5},
                    "lastTrade": {"p": 483.2, "t": T * 10**9},
                    "lastQuote": {"p": 483.1, "P": 483.3, "s": 2, "S": 3},
                },
            },
        )
    )
    q = await Polygon(http, "KEY").quote("BRK-B")
    assert q.symbol == "BRK-B" and q.price == 483.2 and q.bid == 483.1 and q.ask == 483.3
    assert q.previous_close == 479.5
    assert route.calls[0].request.url.params["apiKey"] == "KEY"


@respx.mock
async def test_polygon_plan_restriction_surfaces_as_error(http):
    respx.get(host="api.polygon.io").mock(
        return_value=httpx.Response(
            403, json={"status": "NOT_AUTHORIZED", "message": "You are not entitled to this data."}
        )
    )
    with pytest.raises(ProviderHTTPError):
        await Polygon(http, "KEY").quote("AAPL")


@respx.mock
async def test_polygon_aggregates_follow_next_url_with_key(http):
    base = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day"
    pages = [
        {
            "status": "OK",
            "results": [{"t": (T - 86400) * 1000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}],
            "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/a/b?cursor=abc",
        },
        {"status": "OK", "results": [{"t": T * 1000, "o": 1.5, "h": 2.5, "l": 1, "c": 2, "v": 12}]},
    ]
    route = respx.get(url__startswith=base).mock(side_effect=[httpx.Response(200, json=p) for p in pages])
    h = await Polygon(http, "KEY").history(
        "AAPL", "1d", datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 25, tzinfo=UTC)
    )
    assert [b.close for b in h.bars] == [1.5, 2.0]
    second = route.calls[1].request.url.params
    assert second["cursor"] == "abc" and second["apiKey"] == "KEY"


@respx.mock
async def test_polygon_option_chain(http):
    respx.get("https://api.polygon.io/v3/reference/options/contracts").mock(
        return_value=httpx.Response(
            200, json={"results": [{"expiration_date": "2026-10-16"}, {"expiration_date": "2026-11-20"}]}
        )
    )
    respx.get("https://api.polygon.io/v3/snapshot/options/AAPL").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "details": {
                            "contract_type": "call",
                            "expiration_date": "2026-10-16",
                            "strike_price": 260,
                            "ticker": "O:AAPL261016C00260000",
                        },
                        "last_quote": {"bid": 3.1, "ask": 3.3},
                        "open_interest": 1200,
                        "implied_volatility": 0.27,
                        "underlying_asset": {"price": 252.3, "last_updated": T * 10**9},
                    }
                ],
            },
        )
    )
    chain = await Polygon(http, "KEY").option_chain("AAPL", [date(2026, 10, 16)])
    c = chain.contracts[0]
    assert (
        c.contract_symbol == "AAPL261016C00260000"
        and c.open_interest == 1200
        and chain.underlying_price == 252.3
    )
    assert chain.expirations == [date(2026, 10, 16), date(2026, 11, 20)]


# ----------------------------------------------------------------------------- Alpaca
def test_occ_symbol_parsing():
    assert occ_parse("AAPL261016C00260000") == ("AAPL", date(2026, 10, 16), "call", 260.0)
    assert occ_parse("O:SPY261218P00512500") == ("SPY", date(2026, 12, 18), "put", 512.5)
    with pytest.raises(ValueError):
        occ_parse("AAPL")


@respx.mock
async def test_alpaca_snapshots_bars_and_options(http):
    headers_seen = {}

    def snaps(request):
        headers_seen.update(request.headers)
        return httpx.Response(
            200,
            json={
                "AAPL": {
                    "latestTrade": {"t": "2026-09-24T19:59:59.5Z", "p": 252.3},
                    "latestQuote": {"t": "2026-09-24T19:59:59Z", "bp": 252.2, "ap": 252.4, "bs": 1, "as": 4},
                    "dailyBar": {
                        "t": "2026-09-24T04:00:00Z",
                        "o": 250,
                        "h": 253,
                        "l": 249,
                        "c": 252.3,
                        "v": 4e7,
                    },
                    "prevDailyBar": {
                        "t": "2026-09-23T04:00:00Z",
                        "o": 248,
                        "h": 251,
                        "l": 247,
                        "c": 249.9,
                        "v": 3e7,
                    },
                }
            },
        )

    respx.get("https://data.alpaca.markets/v2/stocks/snapshots").mock(side_effect=snaps)
    alpaca = Alpaca(http, "ID", "SECRET")
    q = await alpaca.quote("AAPL")
    assert q.price == 252.3 and q.ask_size == 4 and q.previous_close == 249.9
    assert headers_seen["apca-api-key-id"] == "ID"

    respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "bars": [{"t": "2026-09-23T04:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 5}],
                    "next_page_token": "p2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "bars": [{"t": "2026-09-24T04:00:00Z", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 6}],
                    "next_page_token": None,
                },
            ),
        ]
    )
    h = await alpaca.history(
        "AAPL", "1d", datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 25, tzinfo=UTC)
    )
    assert [b.close for b in h.bars] == [1.5, 1.8]

    respx.get("https://data.alpaca.markets/v1beta1/options/snapshots/AAPL").mock(
        return_value=httpx.Response(
            200,
            json={
                "snapshots": {
                    "AAPL261016C00260000": {"latestQuote": {"bp": 3.0, "ap": 3.2}, "impliedVolatility": 0.28},
                    "AAPL261016P00240000": {"latestQuote": {"bp": 2.0, "ap": 2.1}},
                    "GARBAGE": {"latestQuote": {"bp": 1, "ap": 2}},
                },
                "next_page_token": None,
            },
        )
    )
    chain = await alpaca.option_chain("AAPL", None)
    assert len(chain.contracts) == 2 and chain.underlying_price == 252.3
    assert {c.kind for c in chain.contracts} == {"call", "put"}


async def test_alpaca_requires_both_keys(http):
    with pytest.raises(ProviderNotConfigured):
        await Alpaca(http, "ID", None).quote("AAPL")
