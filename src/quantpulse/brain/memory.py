"""Structured, searchable memory.

Tiers
    ``short_term``  the recent state of the market and the portfolio (expires after about a day);
    ``working``     the current investigation of a cycle (focus, disagreements, open questions);
    ``long_term``   experience worth keeping: regime changes, decisions and why, lessons;
    ``strategy``    what has worked or failed, per strategy (written by the strategy lab);
    ``agent``       what has been measured about each agent (written by the performance review).

Rows hold a one-line summary plus structured data and tags — never raw model transcripts. A ``key`` makes
a memory unique within its tier (the latest market state replaces the previous one); memories without a
key accumulate.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, or_, select

from quantpulse.db.models import BrainMemoryRow
from quantpulse.db.session import Database

SHORT_TERM, WORKING, LONG_TERM, STRATEGY, AGENT = "short_term", "working", "long_term", "strategy", "agent"
TIERS = (SHORT_TERM, WORKING, LONG_TERM, STRATEGY, AGENT)


def _row_dict(r: BrainMemoryRow) -> dict[str, Any]:
    return {
        "id": r.id,
        "tier": r.tier,
        "kind": r.kind,
        "subject": r.subject,
        "key": r.key,
        "summary": r.summary,
        "data": r.data,
        "tags": r.tags,
        "importance": r.importance,
        "cycle_id": r.cycle_id,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
        "expires_at": r.expires_at,
    }


class MemoryStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def remember(
        self,
        tier: str,
        kind: str,
        subject: str,
        summary: str,
        now: datetime,
        *,
        data: dict[str, Any] | None = None,
        key: str | None = None,
        tags: Sequence[str] = (),
        importance: float = 0.5,
        ttl: timedelta | None = None,
        cycle_id: int | None = None,
    ) -> int:
        if tier not in TIERS:
            raise ValueError(f"unknown memory tier {tier!r}")
        async with self._db.session() as s:
            row = None
            if key is not None:
                row = (
                    await s.scalars(
                        select(BrainMemoryRow).where(BrainMemoryRow.tier == tier, BrainMemoryRow.key == key)
                    )
                ).first()
            if row is None:
                row = BrainMemoryRow(tier=tier, key=key, created_at=now)
                s.add(row)
            row.kind, row.subject, row.summary = kind, subject, summary
            row.data, row.tags, row.importance = dict(data or {}), list(tags), importance
            row.cycle_id, row.updated_at = cycle_id, now
            row.expires_at = now + ttl if ttl is not None else None
            await s.flush()
            return row.id

    async def recall(
        self,
        *,
        tier: str | None = None,
        kind: str | None = None,
        subject: str | None = None,
        tag: str | None = None,
        text: str | None = None,
        now: datetime | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        stmt = select(BrainMemoryRow).order_by(BrainMemoryRow.updated_at.desc(), BrainMemoryRow.id.desc())
        if tier:
            stmt = stmt.where(BrainMemoryRow.tier == tier)
        if kind:
            stmt = stmt.where(BrainMemoryRow.kind == kind)
        if subject:
            stmt = stmt.where(BrainMemoryRow.subject == subject)
        if text:
            like = f"%{text}%"
            stmt = stmt.where(or_(BrainMemoryRow.summary.ilike(like), BrainMemoryRow.subject.ilike(like)))
        if now is not None:
            stmt = stmt.where(or_(BrainMemoryRow.expires_at.is_(None), BrainMemoryRow.expires_at > now))
        async with self._db.session() as s:
            rows = list((await s.scalars(stmt.limit(limit * (4 if tag else 1)))).all())
        if tag:
            rows = [r for r in rows if tag in (r.tags or [])]
        return [_row_dict(r) for r in rows[:limit]]

    async def latest(self, tier: str, kind: str, subject: str | None = None) -> dict[str, Any] | None:
        got = await self.recall(tier=tier, kind=kind, subject=subject, limit=1)
        return got[0] if got else None

    async def purge_expired(self, now: datetime) -> int:
        async with self._db.session() as s:
            result = await s.execute(
                delete(BrainMemoryRow).where(
                    BrainMemoryRow.expires_at.is_not(None), BrainMemoryRow.expires_at <= now
                )
            )
            return int(result.rowcount or 0)  # type: ignore[attr-defined]
