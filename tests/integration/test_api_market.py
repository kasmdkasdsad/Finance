"""Market, rates, options and system endpoints: provenance, fallbacks, validation and streaming."""

import json

import httpx
import pytest
from starlette.testclient import TestClient

from quantpulse.api.app import create_app
from quantpulse.core.http import HttpClient
from quantpulse.services.container import Container

from .conftest import NOW, make_settings

T = int(NOW.timestamp())


def yahoo_chart(price=252.31, prev=249.87):
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "symbol": "AAPL",
                        "regularMarketPrice": price,
                        "regularMarketTime": T - 60,
                        "chartPreviousClose": prev,
                        "longName": "Apple Inc.",
                    },
                    "timestamp": [T - 86400, T - 60],
                    "indicators": {
                        "quote": [
                            {
                                "open": [248, 250],
                                "high": [251, 253],
                                "low": [247, 249],
                                "close": [prev, price],
                                "volume": [1e7, 2e7],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def block_yahoo_crumb(mock_net):
    mock_net.get("https://fc.yahoo.com").mock(return_value=httpx.Response(404))
    mock_net.get("https://query1.finance.yahoo.com/v1/test/getcrumb").mock(return_value=httpx.Response(429))


async def test_health_and_openapi(api):
    r = await api.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-request-id"] and float(r.headers["x-response-time-ms"]) >= 0
    spec = (await api.get("/openapi.json")).json()
    for path in (
        "/api/v1/market/quote/{symbol}",
        "/api/v1/options/price",
        "/api/v1/valuation/{symbol}/dcf",
        "/api/v1/portfolio/analyze",
        "/api/v1/vehicles/{vehicle_id}/dashboard",
        "/api/v1/sports/{league}/scoreboard",
        "/api/v1/picks/daily",
        "/api/v1/picks/email",
    ):
        assert path in spec["paths"]


async def test_offline_quote_is_labelled_synthetic(api):
    body = (await api.get("/api/v1/market/quote/aapl")).json()
    assert body["data"]["symbol"] == "AAPL"
    assert body["meta"]["status"] == "synthetic" and body["meta"]["provider"] == "synthetic"
    assert "live data disabled" in body["meta"]["message"]


async def test_live_quote_then_cache_then_warehouse_fallback(live_api, mock_net):
    block_yahoo_crumb(mock_net)
    chart = mock_net.get(host="query1.finance.yahoo.com", path="/v8/finance/chart/AAPL").mock(
        return_value=httpx.Response(200, json=yahoo_chart())
    )
    first = (await live_api.get("/api/v1/market/quote/AAPL")).json()
    assert first["meta"]["status"] == "live" and first["meta"]["provider"] == "yahoo"
    assert first["data"]["price"] == 252.31
    second = (await live_api.get("/api/v1/market/quote/AAPL")).json()
    assert second["meta"]["status"] == "cached" and chart.call_count == 1

    # Upstream goes down and the in-memory cache is lost (e.g. restart): the warehouse snapshot is served.
    live_api.container.cache.clear()
    chart.mock(return_value=httpx.Response(500))
    third_resp = await live_api.get("/api/v1/market/quote/AAPL")
    assert third_resp.status_code == 200, third_resp.text
    third = third_resp.json()
    assert third["meta"]["status"] == "stale"
    assert third["meta"]["provider"] == "warehouse:yahoo"
    assert third["data"]["price"] == 252.31
    assert any(not a["ok"] for a in third["meta"]["attempts"])


async def test_live_failure_without_history_falls_back_to_synthetic(live_api, mock_net):
    block_yahoo_crumb(mock_net)
    mock_net.get(host="query1.finance.yahoo.com").mock(return_value=httpx.Response(503))
    body = (await live_api.get("/api/v1/market/quote/MSFT")).json()
    assert body["meta"]["status"] == "synthetic"
    assert "yahoo" in body["meta"]["message"]


async def test_history_and_batch_quotes(api):
    h = (await api.get("/api/v1/market/history/SPY", params={"interval": "1d", "lookback_days": 60})).json()
    closes = [b["close"] for b in h["data"]["bars"]]
    assert 35 <= len(closes) <= 45
    q = (await api.get("/api/v1/market/quotes", params={"symbols": "SPY, aapl,SPY"})).json()
    assert set(q["quotes"]) == {"SPY", "AAPL"}


@pytest.mark.parametrize(
    ("method", "url", "payload"),
    [
        ("get", "/api/v1/market/quote/BAD$", None),
        ("get", "/api/v1/market/history/AAPL?interval=7m", None),
        ("get", "/api/v1/market/quotes?symbols=,", None),
        (
            "post",
            "/api/v1/options/price",
            {"strike": 100, "days_to_expiry": 30, "expiration": "2026-12-18", "spot": 100, "volatility": 0.2},
        ),
        ("post", "/api/v1/options/price", {"strike": 100, "days_to_expiry": 30}),
        (
            "post",
            "/api/v1/options/price",
            {"strike": -5, "days_to_expiry": 30, "spot": 100, "volatility": 0.2},
        ),
        (
            "post",
            "/api/v1/options/price",
            {"strike": 100, "days_to_expiry": 30, "spot": 100, "volatility": 0.2, "extra": 1},
        ),
        ("get", "/api/v1/options/AAPL/chain?expirations=not-a-date", None),
        ("get", "/api/v1/rates/at?years=0", None),
    ],
)
async def test_validation_errors_are_structured(api, method, url, payload):
    r = await getattr(api, method)(url, **({"json": payload} if payload is not None else {}))
    assert r.status_code == 422
    body = r.json()
    assert body["error"] in {"validation_error", "domain_error"} and body["request_id"]


async def test_bsm_pricing_with_explicit_inputs_matches_textbook(api):
    r = await api.post(
        "/api/v1/options/price",
        json={
            "kind": "call",
            "strike": 40,
            "days_to_expiry": 182.5,
            "spot": 42,
            "volatility": 0.2,
            "rate": 0.1,
            "dividend_yield": 0,
        },
    )
    body = r.json()
    assert r.status_code == 200
    assert body["data"]["greeks"]["price"] == pytest.approx(4.76, abs=0.01)
    assert body["data"]["counterpart_price"] == pytest.approx(0.81, abs=0.01)
    assert body["meta"]["sources"]["inputs"]["provider"] == "user"


async def test_bsm_implied_vol_and_arbitrage_rejection(api):
    ok = await api.post(
        "/api/v1/options/price",
        json={
            "kind": "call",
            "strike": 40,
            "days_to_expiry": 182.5,
            "spot": 42,
            "rate": 0.1,
            "dividend_yield": 0,
            "market_price": 4.7594,
        },
    )
    assert ok.json()["data"]["implied_volatility"] == pytest.approx(0.2, abs=1e-3)
    bad = await api.post(
        "/api/v1/options/price",
        json={
            "kind": "call",
            "strike": 40,
            "days_to_expiry": 182.5,
            "spot": 42,
            "rate": 0.1,
            "market_price": 50,
        },
    )
    assert bad.status_code == 422 and "no-arbitrage" in bad.json()["detail"]


async def test_bsm_with_symbol_uses_live_inputs(api):
    body = (
        await api.post("/api/v1/options/price", json={"symbol": "aapl", "strike": 200, "days_to_expiry": 30})
    ).json()
    inputs = body["data"]["inputs"]
    assert inputs["volatility_source"].startswith("live smile")
    assert "Treasury curve" in inputs["rate_source"]
    assert set(body["meta"]["sources"]) >= {"quote", "yield_curve", "option_chain"}
    assert body["meta"]["status"] == "synthetic"


async def test_options_chain_and_surface(api):
    chain = (await api.get("/api/v1/options/AAPL/chain", params={"max_expirations": 2})).json()
    contracts = chain["data"]["contracts"]
    assert len({c["expiration"] for c in contracts}) == 2
    solved = [c for c in contracts if c["model_iv"] is not None]
    assert len(solved) > 0.8 * len(contracts)
    # IV is only well-conditioned where vega is material: penny options are dominated by tick rounding and
    # deep in-the-money options have almost no time value. Compare liquid out-of-the-money strikes.
    spot = chain["data"]["underlying_price"]
    liquid = [
        c
        for c in solved
        if c["mid"] >= 1.0
        and ((c["kind"] == "call" and c["strike"] > spot) or (c["kind"] == "put" and c["strike"] < spot))
    ]
    assert len(liquid) > 20
    for c in liquid:
        assert c["model_iv"] == pytest.approx(c["implied_volatility"], abs=0.01)
        assert c["delta"] is not None and c["gamma"] > 0
    surface = (
        await api.get("/api/v1/options/AAPL/surface", params={"max_expirations": 4, "grid_points": 11})
    ).json()
    assert len(surface["data"]["iv_grid"]) == 4 and len(surface["data"]["iv_grid"][0]) == 11
    assert surface["data"]["smiles"][0]["atm_iv"] > 0


async def test_rates_endpoints(api):
    curve = (await api.get("/api/v1/rates/curve")).json()
    assert curve["meta"]["status"] == "synthetic" and len(curve["data"]["points"]) >= 10
    at = (await api.get("/api/v1/rates/at", params={"years": 0.25})).json()
    assert at["data"]["bey_rate"] == pytest.approx(0.0405)
    assert at["data"]["continuous_rate"] < at["data"]["bey_rate"]


async def test_sse_stream_emits_quote_events(api):
    r = await api.get(
        "/api/v1/market/stream", params={"symbols": "AAPL,MSFT", "max_events": 1, "interval": 1}
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [block for block in r.text.split("\n\n") if block.startswith("event: quote")]
    assert len(events) == 2
    payload = json.loads(events[0].split("data: ", 1)[1])
    assert payload["meta"]["status"] in {"synthetic", "cached"}


async def test_system_status_hides_secrets(tmp_path, clock):
    settings = make_settings(tmp_path, polygon_api_key="SECRET-POLYGON-KEY", smtp_password="SMTP-PASS")
    container = Container(settings, clock=clock, http=HttpClient())
    app = create_app(container=container)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
    ):
        await c.get("/api/v1/market/quote/AAPL")
        r = await c.get("/api/v1/system/status")
        session = (await c.get("/api/v1/market/session")).json()
    body = r.json()
    assert r.status_code == 200
    assert body["credentials"]["polygon"] is True and body["credentials"]["alpaca"] is False
    assert body["database"]["revision"] == body["database"]["head"] == "0008"
    assert "SECRET-POLYGON-KEY" not in r.text and "SMTP-PASS" not in r.text
    assert session["session"] == "regular" and session["is_trading_day"] is True


async def test_api_token_enforced(tmp_path, clock):
    settings = make_settings(tmp_path, api_token="s3cret")
    app = create_app(container=Container(settings, clock=clock, http=HttpClient()))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c,
    ):
        assert (await c.get("/health")).status_code == 200
        assert (await c.get("/api/v1/market/quote/AAPL")).status_code == 401
        wrong = await c.get("/api/v1/market/quote/AAPL", headers={"X-API-Key": "wrong"})
        assert wrong.status_code == 401
        right = await c.get("/api/v1/market/quote/AAPL", headers={"X-API-Key": "s3cret"})
        assert right.status_code == 200


def test_websocket_stream_and_auth(tmp_path, clock):
    settings = make_settings(tmp_path, api_token="tok")
    app = create_app(container=Container(settings, clock=clock, http=HttpClient()))
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/market/ws?symbols=AAPL,SPY&interval=1&api_key=tok") as ws:
            message = ws.receive_json()
        assert set(message) == {"AAPL", "SPY"}
        assert message["AAPL"]["meta"]["status"] in {"synthetic", "cached"}
        from starlette.websockets import WebSocketDisconnect

        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/api/v1/market/ws?symbols=AAPL") as ws,
        ):
            ws.receive_json()
        assert exc.value.code == 4401
