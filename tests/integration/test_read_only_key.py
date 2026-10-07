"""The read-only key (QP_API_READ_TOKEN) watches without the power to change anything: it reads (GET) the
monitoring pages, is refused for every order, control and setting, for the heavy market-data pages and for the
quote streams, and no page it reaches acts — against the fake Alpaca paper API.
"""

import hashlib
import re

import httpx
import pytest
from sqlalchemy import select
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from quantpulse.api.app import create_app
from quantpulse.core.clock import FakeClock
from quantpulse.db.models import Base
from quantpulse.reader import Reader
from quantpulse.readonly import readable
from quantpulse.services import preflight

from .conftest import NOW, make_settings
from .test_brain_cycle import brain_client, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS
from .test_cloud_security import CLOUD, HEADERS, TOKEN

READ = "read-only-key-0123456789abcdefghijklmnopqrstuv"
READER = {"X-API-Key": READ}
WITH_READER = {**CLOUD, "api_read_token": READ}
# GETs that only refresh records kept for display: the execution ledger copies each order's latest state from
# the reconciled order record, the model registry lists the models that predate it (once), and the Brain's
# hypothetical paper book is opened at its starting capital the first time it is shown. Nothing else may change.
REFRESHED = {"brain_executions", "model_registry", "brain_state:book"}


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


@pytest.fixture
def preflight_passes(monkeypatch):
    """The preflight itself is covered by the unit tests and by test_cloud_security."""
    monkeypatch.setattr(preflight, "enforce", lambda settings, environ=None: None)


def routes(app, get: bool) -> list[tuple[str, str]]:
    """Every (method, path) the API serves, with sample values for the path parameters."""
    out = []
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            if (method == "get") == get:
                out.append((method.upper(), re.sub(r"\{[^}]+\}", "1", path)))
    return out


def digest(rows) -> str:
    return hashlib.sha256(repr(sorted(map(repr, rows))).encode()).hexdigest()


async def snapshot(api) -> dict[str, str]:
    """Each table's contents, hashed (the Brain's state entry by entry, as it holds the switches too): what a
    request changed shows up as a hash that moved."""
    out = {}
    async with api.container.db.session() as s:
        for table in Base.metadata.sorted_tables:
            rows = (await s.execute(select(table))).all()
            if table.name == "brain_state":
                out |= {f"brain_state:{r._mapping['key']}": digest([r]) for r in rows}
            else:
                out[table.name] = digest(rows)
    return out


async def test_the_read_only_key_reads_the_monitoring_pages_and_is_refused_for_everything_else(
    tmp_path, monkeypatch, preflight_passes
):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED, **WITH_READER):
        assert "cycle" in await api.container.brain.supervisor.tick()  # a cycle, so the pages have content
        for path in (
            "/api/v1/system/status", "/api/v1/system/watchdog", "/api/v1/trading/account",
            "/api/v1/trading/positions", "/api/v1/trading/orders", f"{API}/status", f"{API}/positions",
            f"{API}/cycles", f"{API}/learning", f"{API}/kill-switch", f"{API}/research/operating",
            "/api/v1/options/status", "/api/v1/predictions/scorecard",
        ):  # fmt: skip
            r = await api.get(path, headers=READER)
            assert r.status_code == 200, (path, r.text[:300])
        # every order, control, switch and setting: refused, and the full key still works
        writes = routes(api._transport.app, get=False)
        assert len(writes) > 40
        for method, path in writes:
            r = await api.request(method, path, headers=READER, json={})
            assert r.status_code == 403, (method, path, r.status_code)
            assert "read-only" in r.json()["detail"]
        body = {"active": True, "reason": "test", "cancel_open_orders": False}
        assert (await api.post(f"{API}/kill-switch", json=body, headers=HEADERS)).status_code == 200
        # the heavy or third-party market-data pages and the quote stream: refused
        for path in (
            "/api/v1/forecast/UPA", "/api/v1/market/quote/UPA", "/api/v1/market/stream?symbols=UPA",
            "/api/v1/picks/daily", "/api/v1/options/UPA/chain", "/api/v1/stocks/UPA/report",
            "/api/v1/market/regime",
        ):  # fmt: skip
            assert (await api.get(path, headers=READER)).status_code == 403, path
        # a wrong key, or none, is still a 401
        assert (await api.get(f"{API}/status", headers={"X-API-Key": READ + "x"})).status_code == 401
        assert (await api.get(f"{API}/status")).status_code == 401


async def test_no_page_the_read_only_key_reaches_acts(tmp_path, monkeypatch, preflight_passes):
    """Every GET page the key may read, called with it after a cycle that traded: no order is sent or
    cancelled, and nothing is written but the two display refreshes."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED, **WITH_READER):
        assert "cycle" in await api.container.brain.supervisor.tick()
        assert api.fake.orders  # the cycle traded, so there is something a page could disturb
        log, orders = len(api.fake.log), {k: dict(v) for k, v in api.fake.orders.items()}
        before = await snapshot(api)
        pages = [(m, p) for m, p in routes(api._transport.app, get=True) if readable(m, p)]
        assert len(pages) > 80
        for _, path in pages:
            r = await api.get(path, headers=READER)
            assert r.status_code < 500, (path, r.status_code, r.text[:300])
            assert r.status_code != 403, path
        assert all(method == "GET" for method, _ in api.fake.log[log:])  # Alpaca was only read
        assert {k: dict(v) for k, v in api.fake.orders.items()} == orders
        after = await snapshot(api)
        changed = {t for t in before.keys() | after.keys() if before.get(t) != after.get(t)}
        assert changed <= REFRESHED, changed
        assert "brain_state:book" not in before or "brain_state:book" not in changed  # opened, never changed


def test_the_quote_websocket_refuses_the_read_only_key(tmp_path, preflight_passes):
    settings = make_settings(tmp_path, **WITH_READER)
    with TestClient(create_app(settings)) as client:
        for url in ("/api/v1/market/ws?symbols=SPY", f"/api/v1/market/ws?symbols=SPY&api_key={READ}"):
            with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(url, headers=READER):
                pass
            assert refused.value.code == 4401


async def test_the_read_only_key_counts_only_alongside_the_full_one(tmp_path, preflight_passes):
    clock = FakeClock(NOW)
    no_full = {"deployment": "cloud", "api_read_token": READ}  # the preflight would refuse this; the API too
    async for api in brain_client(tmp_path, clock, **OWNS, **no_full):
        assert (await api.get(f"{API}/status", headers=READER)).status_code == 401
    async for api in brain_client(tmp_path / "full", clock, **OWNS, **CLOUD):  # no read key configured
        assert (await api.get(f"{API}/status", headers=READER)).status_code == 401
        assert (await api.get(f"{API}/status", headers={"X-API-Key": TOKEN})).status_code == 200


async def test_through_the_reader_gateway_only_reading_gets_through(tmp_path, monkeypatch, preflight_passes):
    """The gateway in front of the real API: the read-only key reads the same pages, and nothing acts through
    it, not even with the API's full key."""
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED, **WITH_READER):
        assert "cycle" in await api.container.brain.supervisor.tick()
        log = len(api.fake.log)
        reader = Reader(READ, "http://testserver", transport=api._transport)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=reader), base_url="http://reader"
        ) as gw:
            for path in (f"{API}/positions", "/api/v1/trading/account", f"{API}/cycles?limit=1"):
                r = await gw.get(path, headers=READER)
                assert r.status_code == 200, (path, r.text[:300])
                assert r.json() == (await api.get(path, headers=HEADERS)).json()
            body = {"active": True, "reason": "test", "cancel_open_orders": True}
            assert (await gw.post(f"{API}/kill-switch", json=body, headers=HEADERS)).status_code == 401
            assert (await gw.post(f"{API}/kill-switch", json=body, headers=READER)).status_code == 405
            assert (await gw.post("/api/v1/trading/run", headers=READER)).status_code == 405
            assert (await gw.get("/api/v1/forecast/UPA", headers=READER)).status_code == 403
        assert (await api.get(f"{API}/kill-switch", headers=HEADERS)).json()["active"] is False
        assert all(method == "GET" for method, _ in api.fake.log[log:])
