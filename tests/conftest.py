import os

import pytest
import requests

from quantpulse.db import migrate
from quantpulse.db.session import Database

# Variables that could point the test run at a real (paper) account or change trading behaviour.
ACCOUNT_ENV_PREFIXES = ("QP_ALPACA", "APCA_", "ALPACA_", "QP_TRADING")


@pytest.fixture(autouse=True, scope="session")
def _isolated_from_real_accounts():
    """No test may ever reach a real Alpaca account, whatever the developer's shell holds:

    * Alpaca and trading variables are removed from the environment (tests never read ``.env`` either:
      every ``Settings`` is built with ``_env_file=None``);
    * real ``requests`` network I/O fails at once (the Alpaca SDK uses ``requests``; the fake Alpaca
      paper API is a transport adapter, not a network one). ``httpx`` traffic is mocked with respx."""
    mp = pytest.MonkeyPatch()
    for name in list(os.environ):
        if name.upper().startswith(ACCOUNT_ENV_PREFIXES):
            mp.delenv(name)

    def refuse(self: requests.adapters.HTTPAdapter, request: requests.PreparedRequest, *args, **kwargs):
        raise AssertionError(f"tests must never reach the network (requests to {request.url})")

    mp.setattr(requests.adapters.HTTPAdapter, "send", refuse)
    yield
    mp.undo()


@pytest.fixture
def db_url(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    migrate.upgrade(url)
    return url


@pytest.fixture
async def database(db_url):
    db = Database(db_url)
    yield db
    await db.dispose()
