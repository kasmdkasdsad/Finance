"""The research subsystem as one service: the operating model, the queue and its scheduler, the learning ledger,
the improvement lifecycle and the resource governor — what the API, the dashboard and the Brain use."""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Callable
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError

from .catalog import CATALOG, PHASES
from .ledger import LearningLedger
from .lifecycle import Lifecycle, LifecycleError, promotion_refusal
from .operating import OperatingModel
from .queue import ResearchQueue
from .resources import ResourceGovernor, Snapshot, measure
from .scheduler import ResearchScheduler

RULES = (
    "Research reads the record and writes only its own results, conclusions, questions and preparation notes.",
    "It never sends an order and never changes a setting, a limit, a kill switch, a weight or a strategy in production.",
    "A conclusion is UNPROVEN until its sample reaches the minimum and a statistical test supports it.",
    "An improvement moves one tested stage at a time; only a person promotes it to production.",
    "Execution and safety come first: no research runs in the session, and research stops when memory runs short.",
)


class ResearchService:
    def __init__(
        self, settings: Settings, clock: Clock, brain: Any, *, reference: Any = None, market: Any = None,
        read_resources: Callable[[], Snapshot] = measure,
    ) -> None:  # fmt: skip
        self._s = settings
        self._clock = clock
        self._brain = brain
        self.ledger = LearningLedger(brain.db)
        self.lifecycle = Lifecycle(brain.db)
        self.queue = ResearchQueue(brain.db)
        self.governor = ResourceGovernor(settings, read_resources)
        self.operating = OperatingModel(settings, clock, brain, self.lifecycle)
        holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.scheduler = ResearchScheduler(
            settings, clock, brain, self.queue, self.ledger, self.lifecycle, self.governor, self.operating, holder,
            reference=reference, market=market,
        )  # fmt: skip

    async def tick(self) -> str:
        return await self.scheduler.tick()

    async def stop(self) -> int:
        return await self.scheduler.halt("the process is stopping")

    # ------------------------------------------------------------------ views
    async def status(self) -> dict[str, Any]:
        snap = self.governor.snapshot()
        may, why = self.governor.may_start("medium", snap)
        return {
            "operating": await self.operating.status(),
            "enabled": self._s.research_enabled,
            "running": self.scheduler.running(),
            "last_tick": self.scheduler.last_tick,
            "queue": await self.queue.counts(),
            "resources": {
                "memory_pct": snap.memory_pct,
                "memory_source": snap.memory_source,
                "load_per_cpu": snap.load_per_cpu,
                "rss_mb": snap.rss_mb,
                "may_start": may,
                "detail": why,
                "limits": {
                    "max_concurrent": self._s.research_max_concurrent,
                    "start_below_memory_pct": self._s.research_max_memory_pct,
                    "stop_above_memory_pct": self._s.research_abort_memory_pct,
                    "max_load_per_cpu": self._s.research_max_load,
                    "job_timeout_minutes": self._s.research_job_timeout_minutes,
                },
            },
            "ledger": await self.ledger.summary(),
            "lifecycle": await self.lifecycle.counts(),
            "rules": list(RULES),
        }

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"kind": s.kind, "question": s.question, "phase": s.phase, "cost": s.cost, "value": s.value,
             "refresh_hours": s.refresh.total_seconds() / 3600 if s.refresh else None, "owner_only": s.owner_only}
            for s in sorted(CATALOG.values(), key=lambda s: (PHASES.index(s.phase), -s.value))
        ]  # fmt: skip

    async def jobs(
        self, *, status: str | None = None, kind: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self.queue.jobs(status=status, kind=kind, limit=limit)

    async def job(self, job_id: int) -> dict[str, Any]:
        found = await self.queue.get(job_id)
        if found is None:
            raise DomainError(f"no research job {job_id}")
        return found

    async def ask(self, kind: str, question: str | None, params: dict[str, Any]) -> dict[str, Any]:
        """A person's question: queued with a bonus, answered while the market is closed."""
        try:
            return await self.scheduler.ask(kind, question, params, source="person")
        except ValueError as exc:
            raise DomainError(str(exc)) from exc

    async def cancel(self, job_id: int) -> dict[str, Any]:
        if not await self.queue.cancel(job_id, self._clock.now(), "cancelled by a person"):
            raise DomainError(f"research job {job_id} is not queued")
        return await self.job(job_id)

    async def learnings(self, **filters: Any) -> list[dict[str, Any]]:
        return await self.ledger.learnings(**filters)

    async def hypotheses(self, **filters: Any) -> list[dict[str, Any]]:
        return await self.lifecycle.hypotheses(**filters)

    # ------------------------------------------------------------------ a person's decisions
    async def promote(self, hypothesis_id: int, by: str, note: str) -> dict[str, Any]:
        """EVALUATION → PRODUCTION, by a person. A strategy also goes through the lab's own promotion gates
        (validated, paper-tracked long enough, not short of the benchmark); anything else is an approved change
        for a person to implement — nothing in trading changes by itself."""
        h = await self.lifecycle.get(hypothesis_id)
        if h is None:
            raise DomainError(f"no hypothesis {hypothesis_id}")
        refused = promotion_refusal(h["stage"], h["history"], h["key"], by, note)
        if refused:  # every check before the lab is touched
            raise DomainError(refused)
        if h["kind"] == "strategy":
            strategy_id, _, version = str(h["source_ref"]).partition("@v")
            await self._brain.lab.set_status(
                strategy_id, int(version), "promoted", by[:16]
            )  # the lab's own gates
        try:
            return await self.lifecycle.promote(hypothesis_id, by=by, note=note, now=self._clock.now())
        except LifecycleError as exc:
            raise DomainError(str(exc)) from exc

    async def reject(self, hypothesis_id: int, by: str, note: str) -> dict[str, Any]:
        try:
            return await self.lifecycle.reject(hypothesis_id, by=by, note=note, now=self._clock.now())
        except LifecycleError as exc:
            raise DomainError(str(exc)) from exc
