"""Async engine / session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def _ensure_sqlite_dir(url: str) -> None:
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            raw = url[len(prefix) :].split("?", 1)[0]
            if raw and raw != ":memory:":
                Path(raw).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class PoolSettings:
    """Connection pool and timeouts for PostgreSQL (SQLite ignores them)."""

    size: int = 5
    max_overflow: int = 5
    connect_timeout: float = 10.0
    command_timeout: float = 120.0
    recycle_seconds: int = 1800


def postgres_args(url: str, pool: PoolSettings) -> tuple[str, dict[str, Any]]:
    """The URL asyncpg accepts and the engine arguments: a small pool (a managed database allows few
    connections), pre-ping and recycling (the network can drop idle connections), connect and statement
    timeouts (a stalled database fails a call instead of hanging it), and ``sslmode=…`` — which asyncpg does not
    understand in a URL — turned into its ``ssl`` argument."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    connect: dict[str, Any] = {
        "timeout": pool.connect_timeout,
        "command_timeout": pool.command_timeout,
        "server_settings": {"application_name": "quantpulse"},
    }
    sslmode = query.pop("sslmode", None) or query.pop("ssl", None)
    if sslmode and sslmode != "disable":
        connect["ssl"] = "require" if sslmode in ("require", "prefer", "allow") else sslmode
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    return clean, {
        "pool_size": pool.size,
        "max_overflow": pool.max_overflow,
        "pool_timeout": 30,
        "pool_recycle": pool.recycle_seconds,
        "connect_args": connect,
    }


def create_engine(url: str, echo: bool = False, pool: PoolSettings | None = None) -> AsyncEngine:
    _ensure_sqlite_dir(url)
    if url.startswith("postgresql+asyncpg"):
        url, extra = postgres_args(url, pool or PoolSettings())
        return create_async_engine(url, echo=echo, pool_pre_ping=True, **extra)
    engine = create_async_engine(url, echo=echo, pool_pre_ping=True)
    if url.startswith("sqlite"):

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            if ":memory:" not in url:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine


async def database_now(session: AsyncSession) -> datetime:
    """The database server's current time (UTC, timezone-aware): one clock that every instance shares.

    ``clock_timestamp()`` on PostgreSQL (the real time, not the transaction's start); ``now`` in SQLite (the
    local file's clock, which is this machine's)."""
    if session.get_bind().dialect.name == "postgresql":
        value: datetime = (await session.execute(text("SELECT clock_timestamp()"))).scalar_one()
        return value.astimezone(UTC)
    raw: Any = (await session.execute(text("SELECT strftime('%Y-%m-%d %H:%M:%f', 'now')"))).scalar_one()
    return datetime.fromisoformat(str(raw)).replace(tzinfo=UTC)


class Database:
    """Owns the engine and hands out transactional sessions."""

    def __init__(self, url: str, echo: bool = False, pool: PoolSettings | None = None) -> None:
        self.url = url
        self.engine = create_engine(url, echo=echo, pool=pool)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A session wrapped in a transaction: commits on success, rolls back on error."""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self.engine.dispose()
