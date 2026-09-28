"""The cloud API: it does not start on a bad configuration, needs the token for everything, and never shows
a secret — against the fake Alpaca paper API.
"""

import re
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from quantpulse.api.app import create_app
from quantpulse.core.clock import FakeClock
from quantpulse.core.passwords import hash_password
from quantpulse.services import preflight
from quantpulse.services.container import Container
from tests.fakes.alpaca_paper import FakeAlpacaPaper
from tests.pg import POSTGRES

from .conftest import NOW, make_settings
from .test_brain_cycle import brain_client, with_stock_model
from .test_brain_execution import API, ENABLED, OWNS
from .test_trading import KEY, SECRET

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "cloud-api-token-0123456789abcdefghijklmnop"
HEADERS = {"X-API-Key": TOKEN}
CLOUD = {"deployment": "cloud", "api_token": TOKEN}


@pytest.fixture(autouse=True)
def _no_network(mock_net):
    mock_net.get(url__startswith="https://en.wikipedia.org/").respond(503)
    return mock_net


@pytest.fixture
def preflight_passes(monkeypatch):
    """The auth tests run on the test database (SQLite unless QP_TEST_POSTGRES_URL): the preflight itself is
    covered by the unit tests and by ``test_a_bad_cloud_configuration_starts_nothing``."""
    monkeypatch.setattr(preflight, "enforce", lambda settings, environ=None: None)


# --------------------------------------------------------------------------- start-up
async def test_a_bad_cloud_configuration_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")  # the live-money API
    monkeypatch.setenv("QP_ALPACA_PAPER", "true")
    clock = FakeClock(NOW)
    fake = FakeAlpacaPaper(clock=clock)
    settings = make_settings(
        tmp_path, alpaca_api_key_id=KEY, alpaca_api_secret_key=SECRET, **CLOUD, **ENABLED
    )
    started: list[str] = []
    monkeypatch.setattr(Container, "startup", lambda self: started.append("startup"))
    app = create_app(settings)
    with pytest.raises(preflight.PreflightFailed, match="live-money"):
        async with app.router.lifespan_context(app):
            pass
    assert started == [] and fake.log == []  # no database, no supervisor, no poller, no Alpaca call
    assert not hasattr(app.state, "container")


# --------------------------------------------------------------------------- the token
async def test_in_the_cloud_every_api_request_needs_the_token(tmp_path, preflight_passes):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, **CLOUD):  # requests come from 127.0.0.1
        assert (await api.get("/health")).status_code == 200  # the liveness probe only
        for path in ("/api/v1/trading/status", f"{API}/status", f"{API}/supervisor", "/api/v1/system/status"):
            assert (await api.get(path)).status_code == 401, path
            assert (await api.get(path, headers={"X-API-Key": "wrong"})).status_code == 401, path
            assert (await api.get(path, headers=HEADERS)).status_code == 200, path
        # the Brain kill switch and orders: no exemption for this machine in the cloud
        body = {"active": True, "reason": "test", "cancel_open_orders": False}
        assert (await api.post(f"{API}/kill-switch", json=body)).status_code == 401
        assert (await api.post(f"{API}/kill-switch", json=body, headers=HEADERS)).status_code == 200


async def test_a_cloud_api_without_a_token_refuses_everything_even_locally(tmp_path, preflight_passes):
    clock = FakeClock(NOW)
    async for api in brain_client(tmp_path, clock, **OWNS, deployment="cloud"):  # no QP_API_TOKEN at all
        assert (await api.get("/api/v1/trading/status")).status_code == 401
        body = {"active": True, "reason": "x", "cancel_open_orders": False}
        assert (await api.post(f"{API}/kill-switch", json=body)).status_code == 401
        assert (await api.post("/api/v1/trading/run")).status_code == 401
    async for api in brain_client(tmp_path / "local", clock, **OWNS):  # locally, as before
        assert (await api.get("/api/v1/trading/status")).status_code == 200


def test_the_cloud_websocket_needs_the_token(tmp_path, preflight_passes):
    settings = make_settings(tmp_path, **CLOUD)
    with TestClient(create_app(settings)) as client:
        with (
            pytest.raises(WebSocketDisconnect) as refused,
            client.websocket_connect("/api/v1/market/ws?symbols=SPY"),
        ):
            pass
        assert refused.value.code == 4401


# --------------------------------------------------------------------------- no secret shown
async def test_no_response_and_no_log_line_contains_a_secret(tmp_path, monkeypatch, preflight_passes):
    with_stock_model(monkeypatch)
    clock = FakeClock(NOW)
    password = "dashboard password never shown"
    hashed = hash_password(password)
    logs = tmp_path / "logs"
    async for api in brain_client(tmp_path, clock, **OWNS, **ENABLED, **CLOUD, dashboard_password_hash=hashed,
                                  log_dir=str(logs), log_level="DEBUG"):  # fmt: skip
        api.headers.update(HEADERS)
        assert "cycle" in await api.container.brain.supervisor.tick()
        paths = [
            "/api/v1/trading/status", "/api/v1/trading/diagnostics", "/api/v1/trading/account",
            "/api/v1/system/status", f"{API}/status", f"{API}/execution", f"{API}/supervisor",
            f"{API}/positions", f"{API}/cycles", "/openapi.json",
        ]  # fmt: skip
        bodies = []
        for path in paths:
            r = await api.get(path)
            assert r.status_code == 200, (path, r.text[:300])
            bodies.append(r.text)
        text = "\n".join(bodies)
        for secret in (SECRET, TOKEN, hashed, password):
            assert secret not in text
        assert KEY not in text  # not even the key id
    written = "".join(p.read_text() for p in logs.glob("*.log*"))
    assert written  # the file log was written
    for secret in (SECRET, TOKEN, hashed, password, KEY):
        assert secret not in written


# --------------------------------------------------------------------------- the repository
def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line]


def test_no_credentials_are_committed():
    key_id = re.compile(r"\b[PA]K[A-Z0-9]{16,24}\b")  # the shape of a real Alpaca key id
    assignment = re.compile(
        r"^[ \t]*(?:export[ \t]+)?(?:APCA_API_SECRET_KEY|QP_ALPACA_API_SECRET_KEY|ALPACA_API_SECRET_KEY|QP_API_TOKEN"
        r"|APCA_API_KEY_ID|QP_ALPACA_API_KEY_ID|POSTGRES_PASSWORD|QP_DASHBOARD_PASSWORD_HASH)[ \t]*[=:][ \t]*['\"]?"
        r"([A-Za-z0-9/+$_\-]{16,})",
        re.MULTILINE,
    )
    offenders = []
    for path in tracked_files():
        if path.suffix in (".png", ".ico", ".jpg", ".gif", ".pyc", ".db") or not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if key_id.search(text) or any("${" not in m.group(0) for m in assignment.finditer(text)):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_env_files_are_ignored_by_git_and_docker():
    names = {p.relative_to(ROOT).as_posix() for p in tracked_files()}
    assert ".env" not in names and not any(n.endswith("/.env") for n in names)
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    dockerignore = (ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in gitignore and ".env.*" in gitignore and "!.env.example" in gitignore
    assert ".env" in dockerignore and ".env.*" in dockerignore
    example = (ROOT / ".env.example").read_text()
    for name in ("QP_ALPACA_API_KEY_ID", "QP_ALPACA_API_SECRET_KEY", "QP_API_TOKEN"):
        assert re.search(rf"^{name}=\s*$", example, re.MULTILINE), name  # present, and empty


@pytest.mark.skipif(not POSTGRES, reason="a cloud start needs PostgreSQL (set QP_TEST_POSTGRES_URL)")
async def test_a_complete_cloud_configuration_starts_on_postgres(tmp_path, monkeypatch):
    monkeypatch.setenv("QP_ALPACA_PAPER", "true")
    clock = FakeClock(NOW)
    hashed = hash_password("dashboard password never shown")
    async for api in brain_client(tmp_path, clock, **OWNS, **CLOUD, dashboard_password_hash=hashed):
        assert api.container.settings.database_url.startswith("postgresql+asyncpg://")
        api.headers.update(HEADERS)
        assert (await api.get(f"{API}/supervisor")).status_code == 200
