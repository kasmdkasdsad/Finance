"""Copy the whole QuantPulse database from one place to another — the PC's SQLite file into the cloud's
PostgreSQL — so the Brain's history moves with it: cycles, agent runs, opinions, consensus, decisions,
predictions and their outcomes, reflections, performance, the execution ledger, positions and their
theses, trading days and evaluations, opportunities and their outcomes, events, reviews, the strategy
shadow and the trading service's orders and audit trail.

Both databases are brought to the same migration head first. The target must be empty (nothing is merged
or overwritten). Tables are copied in foreign-key order, in chunks, through the same SQLAlchemy table
definitions, so every column type (JSON, UTC datetimes) converts correctly; PostgreSQL's id sequences are
then moved past the copied ids. Every table's row count is compared at the end: a mismatch is an error.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy import Engine, create_engine, func, inspect, select, text

from quantpulse.db import migrate, models  # noqa: F401  (registers tables on Base.metadata)
from quantpulse.db.base import Base

logger = logging.getLogger(__name__)
CHUNK = 1000


class TransferError(RuntimeError):
    pass


def _engine(url: str) -> Engine:
    return create_engine(migrate.sync_url(url))


def _counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as conn:
        return {
            t.name: conn.execute(select(func.count()).select_from(t)).scalar_one()
            for t in Base.metadata.sorted_tables
        }


def transfer(source_url: str, target_url: str, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Copy every table from ``source_url`` into the empty database at ``target_url``; returns the counts."""
    migrate.upgrade(source_url)
    migrate.upgrade(target_url)
    src, dst = _engine(source_url), _engine(target_url)
    try:
        existing = {k: v for k, v in _counts(dst).items() if v}
        if existing:
            raise TransferError(f"the target database is not empty ({existing}): nothing was copied")
        before = _counts(src)
        with src.connect() as read, dst.begin() as write:
            for table in Base.metadata.sorted_tables:
                total = before[table.name]
                if not total:
                    continue
                copied = 0
                result = read.execution_options(stream_results=True).execute(select(table))
                while rows := result.fetchmany(CHUNK):
                    write.execute(table.insert(), [dict(r._mapping) for r in rows])
                    copied += len(rows)
                progress(f"{table.name}: {copied} row(s)")
            if dst.dialect.name == "postgresql":
                _reset_sequences(write)
        after = _counts(dst)
        wrong = {t: (before[t], after[t]) for t in before if before[t] != after[t]}
        if wrong:
            raise TransferError(f"row counts differ after the copy (source, target): {wrong}")
        return {
            "tables": len(before),
            "rows": sum(before.values()),
            "counts": {k: v for k, v in before.items() if v},
        }
    finally:
        src.dispose()
        dst.dispose()


def _reset_sequences(conn: Any) -> None:
    """Move each serial id sequence past the highest copied id (inserted ids bypass the sequence)."""
    insp = inspect(conn)
    for table in Base.metadata.sorted_tables:
        pk = [c for c in table.primary_key.columns if c.autoincrement is not False and _is_int(c)]
        if len(pk) != 1:
            continue
        col = pk[0].name
        seq = conn.execute(
            text("SELECT pg_get_serial_sequence(:t, :c)"), {"t": table.name, "c": col}
        ).scalar()
        if seq and insp.has_table(table.name):
            conn.execute(
                text(f'SELECT setval(:s, COALESCE((SELECT MAX("{col}") FROM "{table.name}"), 0) + 1, false)'),
                {"s": seq},
            )


def _is_int(column: Any) -> bool:
    try:
        return column.type.python_type is int
    except NotImplementedError:
        return False
