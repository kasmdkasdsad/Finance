"""Programmatic Alembic helpers (used at API start-up and by the ``quantpulse-migrate`` CLI)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from quantpulse.db.session import _ensure_sqlite_dir

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def sync_url(url: str) -> str:
    """Alembic runs synchronously; swap async drivers for their sync counterparts (psycopg for PostgreSQL,
    which reads ``sslmode`` natively and gets a connect timeout so a migration never hangs on the network)."""
    url = url.strip().replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")
    for prefix in ("postgres://", "postgresql://"):  # as Render and other hosts hand them out
        if url.startswith(prefix):
            url = "postgresql+psycopg://" + url[len(prefix) :]
    if not url.startswith("postgresql+psycopg"):
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    if "ssl" in query and "sslmode" not in query:
        query["sslmode"] = query.pop("ssl")
    query.setdefault("connect_timeout", "10")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    _ensure_sqlite_dir(url)
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str) -> None:
    command.downgrade(alembic_config(url), revision)


def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config("sqlite://")).get_current_head()


def current_revision(url: str) -> str | None:
    engine = create_engine(sync_url(url))
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()
