"""PostgreSQL as Render and other hosts hand it out: URL formats, SSL, the pool and its timeouts."""

import pytest

from quantpulse.config import Settings
from quantpulse.db.migrate import sync_url
from quantpulse.db.session import PoolSettings, create_engine, postgres_args

RENDER_INTERNAL = "postgresql://quantpulse:pw@dpg-abc123-a/quantpulse"
RENDER_EXTERNAL = (
    "postgres://quantpulse:pw@dpg-abc123-a.virginia-postgres.render.com/quantpulse?sslmode=require"
)


@pytest.mark.parametrize(
    "given",
    [
        "postgres://u:p@h:5432/db",
        "postgresql://u:p@h:5432/db",
        "postgresql+psycopg://u:p@h:5432/db",
        "postgresql+psycopg2://u:p@h:5432/db",
        "postgresql+asyncpg://u:p@h:5432/db",
        "  postgresql://u:p@h:5432/db  ",
    ],
)
def test_every_postgres_url_form_runs_on_asyncpg(given):
    assert Settings(_env_file=None, database_url=given).database_url == "postgresql+asyncpg://u:p@h:5432/db"


def test_sqlite_urls_are_left_alone():
    url = "sqlite+aiosqlite:///./data/quantpulse.db"
    assert Settings(_env_file=None, database_url=url).database_url == url


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (RENDER_INTERNAL, "postgresql+psycopg://quantpulse:pw@dpg-abc123-a/quantpulse?connect_timeout=10"),
        (
            RENDER_EXTERNAL,
            "postgresql+psycopg://quantpulse:pw@dpg-abc123-a.virginia-postgres.render.com/quantpulse"
            "?sslmode=require&connect_timeout=10",
        ),
        ("postgresql+asyncpg://u:p@h/db?ssl=require", "postgresql+psycopg://u:p@h/db?sslmode=require&connect_timeout=10"),
        ("sqlite+aiosqlite:///x.db", "sqlite:///x.db"),
    ],
)  # fmt: skip
def test_migrations_use_the_sync_driver_with_ssl_and_a_connect_timeout(given, expected):
    assert sync_url(given) == expected


@pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"])
def test_sslmode_is_handed_to_asyncpg_as_it_is(mode):
    url, args = postgres_args(f"postgresql+asyncpg://u:p@h/db?sslmode={mode}&x=1", PoolSettings())
    assert url == "postgresql+asyncpg://u:p@h/db?x=1"  # asyncpg would reject sslmode in the URL
    assert args["connect_args"]["ssl"] == mode


def test_an_unknown_sslmode_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown sslmode"):
        postgres_args("postgresql+asyncpg://u:p@h/db?sslmode=yes-please", PoolSettings())


def test_no_sslmode_leaves_asyncpg_its_default():
    _, args = postgres_args("postgresql+asyncpg://u:p@h/db", PoolSettings())
    assert "ssl" not in args["connect_args"]


def test_the_pool_and_timeouts_come_from_the_settings():
    pool = PoolSettings(
        size=3, max_overflow=2, connect_timeout=7.0, command_timeout=45.0, recycle_seconds=900
    )
    _, args = postgres_args("postgresql+asyncpg://u:p@h/db", pool)
    assert args["pool_size"] == 3 and args["max_overflow"] == 2 and args["pool_recycle"] == 900
    assert args["pool_timeout"] == 30  # waiting for a free connection fails instead of hanging
    connect = args["connect_args"]
    assert connect["timeout"] == 7.0 and connect["command_timeout"] == 45.0
    assert connect["server_settings"] == {"application_name": "quantpulse"}


async def test_the_engine_is_built_with_pre_ping_and_the_pool_without_connecting():
    engine = create_engine("postgresql+asyncpg://u:p@nowhere.invalid/db", pool=PoolSettings(size=4))
    try:
        assert engine.pool.size() == 4 and engine.pool._pre_ping  # a dropped idle connection is replaced
    finally:
        await engine.dispose()  # a clean pool shutdown needs no connection either


def test_settings_bound_the_pool():
    with pytest.raises(ValueError):
        Settings(_env_file=None, db_pool_size=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, db_startup_wait_seconds=-1)


@pytest.mark.skipif(not __import__("tests.pg", fromlist=["POSTGRES"]).POSTGRES, reason="needs PostgreSQL")
@pytest.mark.parametrize("mode", ["disable", "prefer"])
async def test_a_real_postgres_connects_with_the_ssl_modes_render_uses(tmp_path, mode):
    from sqlalchemy import text

    from quantpulse.db import migrate
    from quantpulse.db.session import Database
    from tests.pg import database_url

    url = database_url(tmp_path / "ssl")
    db = Database(f"{url}?sslmode={mode}", pool=PoolSettings(size=1, connect_timeout=5))
    try:
        async with db.session() as s:
            assert (await s.execute(text("SELECT 1"))).scalar_one() == 1
        migrate.upgrade(f"{url}?sslmode={mode}")  # the pre-deploy migration with the same URL
        assert migrate.current_revision(url) == migrate.head_revision()
    finally:
        await db.dispose()
