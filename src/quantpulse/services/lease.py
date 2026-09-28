"""A database lease: at most one QuantPulse process supervises the Brain and sends orders.

Two cloud instances (a deployment overlapping a restart, a second replica, a PC and a server pointed at the
same database) must never run the Brain side by side. The lease is a row in ``service_leases``:

* **acquire** — atomically, in one statement: take the row if it is free, expired, or already ours
  (``UPDATE … WHERE holder = me OR expires_at < now``); if there is no row yet, insert it — a second insert
  fails on the primary key. On PostgreSQL a racing ``UPDATE`` waits for the first to commit and re-checks the
  condition against the new row, so exactly one process wins; SQLite serialises writes. Holding it renews it;
* **held** — read back from the database immediately before an order is sent, so a process that lost the
  lease (a long pause, a network partition) stops at once;
* **release** — on a clean shutdown, so the next process takes over without waiting. A released row is marked
  (its expiry is set just *before* its last heartbeat, which a renewal never does), so the next holder can
  tell a clean hand-over from a process that died holding the lease (:meth:`info` ``released``).

A process that crashes simply stops renewing: its lease expires after ``ttl`` and another process may take
over — never earlier. A restarted process is a new holder and waits like any other. Order ids are also
deterministic per slot (the order manager's write-ahead record), so even across a takeover no order is sent
twice.

**Whose clock.** In production (``db_time``) every comparison and every timestamp uses the *database server's*
clock (``clock_timestamp()`` on PostgreSQL), read in the same transaction as the update: two instances whose
own clocks disagree — by seconds or by hours — still agree on who holds the lease, so a standby whose clock
runs ahead can never think a live leader's lease has expired. Tests use the injected (fake) clock instead, to
step through expiries deterministically.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import case, or_, select, update
from sqlalchemy.exc import IntegrityError

from quantpulse.core import runtime
from quantpulse.core.clock import Clock
from quantpulse.db.models import ServiceLeaseRow
from quantpulse.db.session import Database, database_now

logger = logging.getLogger(__name__)
BRAIN_LEASE = "brain-supervisor"
DEFAULT_TTL = timedelta(seconds=180)


def holder_id() -> str:
    """This process: the instance (Render's instance id, or the host name), pid and a random suffix (a
    restarted process is a new holder)."""
    return f"{runtime.current().instance}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class Lease:
    def __init__(
        self,
        db: Database,
        clock: Clock,
        name: str = BRAIN_LEASE,
        ttl: timedelta = DEFAULT_TTL,
        holder: str | None = None,
        *,
        db_time: bool = False,
    ) -> None:
        self._db = db
        self._clock = clock
        self._db_time = db_time
        self.name = name
        self.ttl = ttl
        self.holder = holder or holder_id()
        self._heartbeat: asyncio.Task[None] | None = None
        self.lost_at: datetime | None = None  # when this process last found the lease held by another

    async def _now(self, session: Any) -> datetime:
        return await database_now(session) if self._db_time else self._clock.now()

    async def acquire(self) -> bool:
        """Take (or renew) the lease; ``False`` while another live process holds it."""
        async with self._db.session() as s:
            now = await self._now(s)
            until = now + self.ttl
            result = await s.execute(
                update(ServiceLeaseRow)
                .where(
                    ServiceLeaseRow.name == self.name,
                    or_(ServiceLeaseRow.holder == self.holder, ServiceLeaseRow.expires_at < now),
                )
                .values(
                    holder=self.holder,
                    heartbeat_at=now,
                    expires_at=until,
                    # a takeover starts a new tenure; a renewal keeps the original start
                    acquired_at=case(
                        (ServiceLeaseRow.holder == self.holder, ServiceLeaseRow.acquired_at), else_=now
                    ),
                )
                .execution_options(synchronize_session=False)
            )
            won = bool(getattr(result, "rowcount", 0))
        if not won:
            try:
                async with self._db.session() as s:
                    s.add(ServiceLeaseRow(name=self.name, holder=self.holder, acquired_at=now,
                                          heartbeat_at=now, expires_at=until))  # fmt: skip
                won = True
            except IntegrityError:
                won = False  # the row exists and is held by a live process
        if won:
            self.lost_at = None
        elif self.lost_at is None:
            self.lost_at = now
        return won

    async def held(self) -> bool:
        """Is the lease ours and live right now (read from the database, not remembered)?"""
        async with self._db.session() as s:
            now = await self._now(s)
            row = await s.get(ServiceLeaseRow, self.name)
        return row is not None and row.holder == self.holder and row.expires_at > now

    async def release(self) -> None:
        self.stop_heartbeat()
        async with self._db.session() as s:
            now = await self._now(s)
            row = await s.get(ServiceLeaseRow, self.name)
            if row is not None and row.holder == self.holder:
                row.heartbeat_at, row.expires_at = now, now - timedelta(seconds=1)

    async def info(self) -> dict[str, Any]:
        async with self._db.session() as s:
            now = await self._now(s)
            row = (await s.scalars(select(ServiceLeaseRow).where(ServiceLeaseRow.name == self.name))).first()
        if row is None:
            return {
                "name": self.name,
                "holder": None,
                "mine": False,
                "live": False,
                "this_process": self.holder,
            }
        live = row.expires_at > now
        return {
            "name": self.name,
            "holder": row.holder,
            "mine": row.holder == self.holder and live,
            "live": live,
            "acquired_at": row.acquired_at.isoformat() if row.acquired_at else None,
            "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
            "heartbeat_age_seconds": round((now - row.heartbeat_at).total_seconds(), 1)
            if row.heartbeat_at
            else None,
            "expires_at": row.expires_at.isoformat(),
            "expires_in_seconds": round((row.expires_at - now).total_seconds(), 1),
            "clock": "database" if self._db_time else "process",
            # handed over cleanly (a renewal always sets the expiry after the heartbeat; a release before it)
            "released": bool(row.heartbeat_at and row.expires_at < row.heartbeat_at),
            "this_process": self.holder,
        }

    # ------------------------------------------------------------------ heartbeat
    def start_heartbeat(self, interval: float | None = None) -> None:
        """Renew in the background while held (a long tick must not let the lease lapse)."""
        if self._heartbeat is not None and not self._heartbeat.done():
            return
        every = interval if interval is not None else max(self.ttl.total_seconds() / 3, 1.0)

        async def beat() -> None:
            while True:
                await asyncio.sleep(every)
                try:
                    if not await self.acquire():
                        logger.warning(
                            "lease %s lost to %s: this process stands by", self.name, "another holder"
                        )
                        return
                except Exception:  # a database hiccup: the lease simply lapses if it lasts; retried
                    logger.exception("renewing lease %s failed", self.name)

        self._heartbeat = asyncio.create_task(beat(), name=f"lease:{self.name}")

    def stop_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with contextlib.suppress(Exception):
                self._heartbeat = None
