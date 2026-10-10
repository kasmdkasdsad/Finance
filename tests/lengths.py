"""String-length guard for the test suite.

PostgreSQL enforces ``VARCHAR(n)``; SQLite silently stores longer values. With this guard installed, every ORM
flush in the (SQLite) test suite checks each string against its column's length, so a value that would fail
in the cloud deployment fails the test instead."""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import String, event, inspect
from sqlalchemy.orm import Session

REPORT = os.environ.get("QP_TEST_LENGTH_REPORT")  # a file to collect violations into instead of raising


def _check(session: Session, _ctx: Any, _instances: Any) -> None:
    for obj in [*session.new, *session.dirty]:
        mapper = inspect(obj).mapper
        for col in mapper.columns:
            if not isinstance(col.type, String) or not col.type.length:
                continue
            value = getattr(obj, col.key, None)
            if isinstance(value, str) and len(value) > col.type.length:
                msg = f"{mapper.class_.__name__}.{col.key}: {len(value)} > {col.type.length} ({value[:60]!r})"
                if REPORT:
                    with open(REPORT, "a") as fh:
                        fh.write(msg + "\n")
                else:
                    raise ValueError(f"value too long for VARCHAR({col.type.length}) — {msg}")


def install() -> None:
    if not event.contains(Session, "before_flush", _check):
        event.listen(Session, "before_flush", _check)
