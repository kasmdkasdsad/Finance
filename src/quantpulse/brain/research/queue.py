"""The persistent research queue: every question the Brain wants answered, every run and its result.

* **One question, once.** A question is identified by its kind and parameters (its key); while one is queued or
  running, asking it again only raises its priority.
* **Priority = expected information value.** ``value × uncertainty × staleness ÷ cost`` (+ a bonus for a
  person's question): what the answer is worth, how unsettled the current conclusion is (no conclusion or an
  UNPROVEN one: high; SUPPORTED: lower), how long since it was last answered, and what it costs to run. The
  components are stored with the job, so the order can always be explained.
* **Restartable.** Jobs live in the database. A job left ``running`` by a process that stopped (a reboot, a
  crash, a deploy) is found by :meth:`ResearchQueue.recover` and queued again. ``attempts`` counts the runs that
  ended badly (an error, a timeout, a crash, memory); after ``max_attempts`` of those it fails. Being stopped for
  the open, a pause or a shutdown costs nothing.
* **Kept.** Finished jobs stay: they are the experiment history.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update

from quantpulse.db.research_models import BrainResearchJobRow
from quantpulse.db.session import Database

COST_WEIGHT = {"light": 1.0, "medium": 1.5, "heavy": 2.5}
UNCERTAINTY = {None: 1.0, "UNPROVEN": 1.0, "INCONCLUSIVE": 0.8, "REFUTED": 0.6, "SUPPORTED": 0.5}
SOURCE_BONUS = {"person": 5.0, "follow_up": 1.0, "event": 1.0, "system": 0.0}
OPEN = ("queued", "running")


def job_key(kind: str, params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16] if params else "-"
    return f"{kind}:{digest}"[:160]


def priority(
    value: float,
    cost: str,
    *,
    last_status: str | None,
    age: timedelta | None,
    refresh: timedelta | None,
    source: str,
) -> tuple[float, dict[str, Any]]:
    uncertainty = UNCERTAINTY.get(last_status, 1.0)
    # never answered: as stale as it gets
    staleness = 3.0 if age is None or refresh is None else 1.0 + min(2.0, max(0.0, age / refresh))
    bonus = SOURCE_BONUS.get(source, 0.0)
    score = value * uncertainty * staleness / COST_WEIGHT.get(cost, 1.5) + bonus
    return round(score, 4), {
        "value": value,
        "uncertainty": uncertainty,
        "staleness": round(staleness, 3),
        "cost": cost,
        "cost_weight": COST_WEIGHT.get(cost, 1.5),
        "bonus": bonus,
        "last_conclusion": last_status,
    }


def _out(r: BrainResearchJobRow) -> dict[str, Any]:
    return {
        "id": r.id,
        "kind": r.kind,
        "key": r.key,
        "question": r.question,
        "params": r.params,
        "source": r.source,
        "parent_id": r.parent_id,
        "cost": r.cost,
        "priority": r.priority,
        "priority_detail": r.priority_detail,
        "status": r.status,
        "attempts": r.attempts,
        "max_attempts": r.max_attempts,
        "not_before": r.not_before.isoformat() if r.not_before else None,
        "created_at": r.created_at.isoformat(),
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "heartbeat_at": r.heartbeat_at.isoformat() if r.heartbeat_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "holder": r.holder,
        "duration_ms": r.duration_ms,
        "peak_rss_mb": r.peak_rss_mb,
        "result": r.result,
        "error": r.error,
    }


class ResearchQueue:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def enqueue(
        self,
        *,
        kind: str,
        question: str,
        params: dict[str, Any],
        cost: str,
        priority_: float,
        priority_detail: dict[str, Any],
        source: str,
        now: datetime,
        parent_id: int | None = None,
        not_before: datetime | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        key = job_key(kind, params)
        async with self._db.session() as s:
            existing = await s.scalar(
                select(BrainResearchJobRow).where(
                    BrainResearchJobRow.key == key, BrainResearchJobRow.status.in_(OPEN)
                )
            )
            if existing is not None:
                if priority_ > existing.priority:
                    existing.priority, existing.priority_detail = priority_, priority_detail
                return {**_out(existing), "deduplicated": True}
            row = BrainResearchJobRow(
                kind=kind,
                key=key,
                question=question,
                params=params,
                source=source,
                parent_id=parent_id,
                cost=cost,
                priority=priority_,
                priority_detail=priority_detail,
                status="queued",
                attempts=0,
                max_attempts=max_attempts,
                not_before=not_before,
                created_at=now,
                result={},
            )
            s.add(row)
            await s.flush()
            return _out(row)

    async def next_jobs(
        self, now: datetime, limit: int = 5, costs: tuple[str, ...] | None = None
    ) -> list[dict[str, Any]]:
        stmt = select(BrainResearchJobRow).where(BrainResearchJobRow.status == "queued")
        if costs is not None:
            stmt = stmt.where(BrainResearchJobRow.cost.in_(costs))
        stmt = stmt.order_by(BrainResearchJobRow.priority.desc(), BrainResearchJobRow.id)
        async with self._db.session() as s:
            rows = (await s.scalars(stmt.limit(limit * 4))).all()
        return [_out(r) for r in rows if r.not_before is None or r.not_before <= now][:limit]

    async def claim(self, job_id: int, holder: str, now: datetime) -> bool:
        """queued → running for this process (atomic: two processes never run the same job)."""
        async with self._db.session() as s:
            result = await s.execute(
                update(BrainResearchJobRow)
                .where(BrainResearchJobRow.id == job_id, BrainResearchJobRow.status == "queued")
                .values(
                    status="running",
                    holder=holder[:96],
                    started_at=now,
                    heartbeat_at=now,
                    attempts=BrainResearchJobRow.attempts + 1,
                    error=None,
                )
            )
            return (result.rowcount or 0) == 1  # type: ignore[attr-defined]

    async def heartbeat(self, job_id: int, now: datetime) -> None:
        async with self._db.session() as s:
            await s.execute(
                update(BrainResearchJobRow)
                .where(BrainResearchJobRow.id == job_id, BrainResearchJobRow.status == "running")
                .values(heartbeat_at=now)
            )

    async def finish(
        self,
        job_id: int,
        *,
        status: str,
        now: datetime,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        peak_rss_mb: float | None = None,
    ) -> None:
        async with self._db.session() as s:
            row = await s.get(BrainResearchJobRow, job_id)
            if row is None:
                return
            row.status, row.finished_at, row.error = status, now, (error or None)
            row.result = result if result is not None else row.result
            row.peak_rss_mb = peak_rss_mb
            if row.started_at is not None:
                row.duration_ms = int((now - row.started_at).total_seconds() * 1000)

    async def requeue(
        self, job_id: int, *, reason: str, now: datetime, delay: timedelta, charge: bool = True
    ) -> str:
        """Back to the queue after an interruption or failure — or ``failed`` once attempts run out. An
        interruption that is not the job's doing (the open, a pause, a stop) is not charged: the attempt is given
        back. A failure, a timeout, a crash or running short of memory is."""
        async with self._db.session() as s:
            row = await s.get(BrainResearchJobRow, job_id)
            if row is None:
                return "missing"
            if not charge:
                row.attempts = max(0, row.attempts - 1)
            elif row.attempts >= row.max_attempts:
                row.status, row.finished_at, row.error = "failed", now, f"{reason} (attempt {row.attempts})"
                return "failed"
            row.status, row.holder, row.error, row.not_before = "queued", None, reason, now + delay
            return "queued"

    async def recover(
        self, now: datetime, holder: str, stale_after: timedelta, keep: frozenset[int] = frozenset()
    ) -> list[int]:
        """Jobs left running by another process (or silent for too long): queued again. Restart recovery."""
        async with self._db.session() as s:
            rows = (
                await s.scalars(select(BrainResearchJobRow).where(BrainResearchJobRow.status == "running"))
            ).all()
            stale = [
                r.id
                for r in rows
                if r.id not in keep
                and (r.holder != holder or r.heartbeat_at is None or now - r.heartbeat_at > stale_after)
            ]
        for job_id in stale:
            await self.requeue(
                job_id, reason="interrupted (the process stopped or went silent)", now=now, delay=timedelta(0)
            )
        return stale

    async def cancel(self, job_id: int, now: datetime, reason: str) -> bool:
        async with self._db.session() as s:
            row = await s.get(BrainResearchJobRow, job_id)
            if row is None or row.status != "queued":
                return False
            row.status, row.finished_at, row.error = "cancelled", now, reason
            return True

    async def last_finished(self, kind: str, key: str | None = None) -> dict[str, Any] | None:
        stmt = select(BrainResearchJobRow).where(
            BrainResearchJobRow.kind == kind, BrainResearchJobRow.status.in_(("done", "failed"))
        )
        if key is not None:
            stmt = stmt.where(BrainResearchJobRow.key == key)
        async with self._db.session() as s:
            row = await s.scalar(stmt.order_by(BrainResearchJobRow.finished_at.desc()).limit(1))
            return _out(row) if row is not None else None

    async def is_open(self, key: str) -> bool:
        stmt = (
            select(func.count())
            .select_from(BrainResearchJobRow)
            .where(BrainResearchJobRow.key == key, BrainResearchJobRow.status.in_(OPEN))
        )
        async with self._db.session() as s:
            return int(await s.scalar(stmt) or 0) > 0

    async def get(self, job_id: int) -> dict[str, Any] | None:
        async with self._db.session() as s:
            row = await s.get(BrainResearchJobRow, job_id)
            return _out(row) if row is not None else None

    async def jobs(
        self, *, status: str | None = None, kind: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        stmt = select(BrainResearchJobRow)
        if status:
            stmt = stmt.where(BrainResearchJobRow.status == status)
        if kind:
            stmt = stmt.where(BrainResearchJobRow.kind == kind)
        order = (
            (BrainResearchJobRow.priority.desc(),) if status == "queued" else (BrainResearchJobRow.id.desc(),)
        )
        async with self._db.session() as s:
            return [_out(r) for r in (await s.scalars(stmt.order_by(*order).limit(limit))).all()]

    async def counts(self) -> dict[str, int]:
        async with self._db.session() as s:
            rows = (
                await s.execute(
                    select(BrainResearchJobRow.status, func.count()).group_by(BrainResearchJobRow.status)
                )
            ).all()
        return {str(status): int(n) for status, n in rows}
