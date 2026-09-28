"""Alembic environment. Runs migrations synchronously (async drivers are swapped for sync ones)."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from quantpulse.db import models  # noqa: F401  (registers tables on Base.metadata)
from quantpulse.db.base import Base
from quantpulse.db.migrate import sync_url

config = context.config
MIGRATION_LOCK = 7_402_117_331  # any fixed 64-bit number: the advisory lock QuantPulse migrations take
target_metadata = Base.metadata


def _url() -> str:
    x_args = context.get_x_argument(as_dictionary=True)
    url = x_args.get("url") or config.get_main_option("sqlalchemy.url")
    if not url:
        from quantpulse.config import get_settings

        url = get_settings().database_url
    return sync_url(url)


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            if connection.dialect.name == "postgresql":
                # two instances starting together migrate one after the other, never side by side (the lock
                # is released when this transaction ends)
                connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK})
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
