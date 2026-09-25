from datetime import UTC, datetime

import httpx
import pytest
import respx

from quantpulse.api.app import create_app
from quantpulse.config import Settings
from quantpulse.core.clock import FakeClock
from quantpulse.core.http import HttpClient
from quantpulse.services.container import Container

# Friday 2026-09-25 10:00 America/New_York — NYSE regular session.
NOW = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'api.db'}",
        polling_enabled=False,
        enable_live_data=False,
        market_providers=["yahoo"],
        sec_user_agent="QuantPulse Tests tests@example.com",
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def clock():
    return FakeClock(NOW)


@pytest.fixture
def mock_net():
    """Every outbound request must be explicitly mocked; unmocked calls fail (and trigger fallbacks)."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        yield router


async def _client(settings: Settings, clock: FakeClock):
    container = Container(settings, clock=clock, http=HttpClient(timeout=5, max_retries=0, backoff_base=0.0))
    app = create_app(container=container)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client,
    ):
        client.container = container  # type: ignore[attr-defined]
        yield client


@pytest.fixture
async def api(tmp_path, clock):
    """Offline API (live data disabled): every feed resolves to warehouse/synthetic fallbacks."""
    async for client in _client(make_settings(tmp_path), clock):
        yield client


@pytest.fixture
async def live_api(tmp_path, clock, mock_net):
    """Live-enabled API with all network access mocked via respx."""
    async for client in _client(make_settings(tmp_path, enable_live_data=True), clock):
        yield client
