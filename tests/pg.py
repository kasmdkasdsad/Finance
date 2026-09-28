"""Opt-in PostgreSQL for the test suite: with ``QP_TEST_POSTGRES_URL`` set (e.g.
``postgresql+asyncpg://postgres@127.0.0.1:5432``) every test database is a real PostgreSQL database instead
of a SQLite file — the same code, migrations and assertions, on the cloud deployment's database.

The database name is derived from the test's path, so a test that "restarts" the API on the same path gets
the same database back (exactly like the SQLite file), and every other test gets its own.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

POSTGRES = os.environ.get("QP_TEST_POSTGRES_URL", "").rstrip("/")


def database_url(path: Path) -> str:
    """A database for this test path: a SQLite file, or (opt-in) a PostgreSQL database of its own."""
    if not POSTGRES:
        return f"sqlite+aiosqlite:///{path}"
    import psycopg

    name = "qp_test_" + hashlib.sha1(str(path).encode()).hexdigest()[:16]
    admin = POSTGRES.replace("+asyncpg", "").replace("+psycopg", "")
    with psycopg.connect(f"{admin}/postgres", autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}"')
    return f"{POSTGRES}/{name}"
