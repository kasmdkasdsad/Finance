"""Programmatic Alembic helpers (used at API start-up and by the ``quantpulse-migrate`` CLI)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from quantpulse.db.session import _ensure_sqlite_dir

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def sync_url(url: str) -> str:
    """Alembic runs synchronously; swap async drivers for their sync counterparts."""
    return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


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
