import pytest

from quantpulse.db import migrate
from quantpulse.db.session import Database


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
