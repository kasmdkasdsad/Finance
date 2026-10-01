"""The closed-market research scheduler.

The background poller ticks it once a minute — separately from the supervisor, so research never delays a
supervisor tick. Each tick:

1. **Only where the Brain is supervised.** A process that does not hold the supervisor lease runs no research.
2. **Restart recovery.** Jobs left running by a stopped process (or silent for 10 minutes) are queued again.
3. **Execution first.** Outside the research windows (see :mod:`.operating`) — the open session, and pre-market
   after 09:15 — no job starts, and any running job is stopped and queued again. So is everything when the
   supervisor is paused or the process is stopping.
4. **Resource limits.** Above the abort memory threshold running jobs are stopped and queued again; a job starts
   only while memory and CPU leave room (heavy jobs need more), and never more than
   ``QP_RESEARCH_MAX_CONCURRENT`` at once.
5. **Standing questions** whose answers are stale are queued (at most every 10 minutes), then the most valuable
   queued jobs start, each with a timeout and a heartbeat. Results, conclusions and follow-up questions are
   recorded when a job finishes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.logging_config import log_event

from .catalog import CATALOG, JobSpec
from .handlers import JobContext, _brief
from .ledger import LearningLedger
from .lifecycle import Lifecycle
from .operating import OperatingModel, mode_at, research_costs
from .queue import ResearchQueue, job_key, priority
from .resources import ResourceGovernor

logger = logging.getLogger(__name__)
STALE = timedelta(minutes=10)
HEARTBEAT_SECONDS = 30.0
SEED_EVERY = timedelta(minutes=10)
RETRY_AFTER = {
    "error": timedelta(minutes=30),
    "timeout": timedelta(hours=2),
    "interrupted": timedelta(minutes=5),
}


class ResearchScheduler:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        brain: Any,
        queue: ResearchQueue,
        ledger: LearningLedger,
        lifecycle: Lifecycle,
        governor: ResourceGovernor,
        operating: OperatingModel,
        holder: str,
        reference: Any = None,
        market: Any = None,
    ) -> None:
        self._s = settings
        self._clock = clock
        self._brain = brain
        self.queue = queue
        self.ledger = ledger
        self.lifecycle = lifecycle
        self.governor = governor
        self.operating = operating
        self.holder = holder
        self._reference = reference
        self._market = market
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._why_stopped: dict[int, tuple[str, bool]] = {}  # job -> (reason, charged as an attempt)
        self._recovered = False
        self._last_seed: datetime | None = None
        self.last_tick: dict[str, Any] = {}

    # ------------------------------------------------------------------ asking questions
    async def _uncertainty(self, spec: JobSpec) -> str | None:
        """How settled the current conclusions on the question's topic are (the least settled counts)."""
        if not spec.topic:
            return None
        current = await self.ledger.learnings(current_only=True, limit=5000)
        mine = [r["status"] for r in current if r["topic"].startswith(spec.topic)]
        for status in ("UNPROVEN", "INCONCLUSIVE", "REFUTED", "SUPPORTED"):
            if status in mine:
                return status
        return None

    async def ask(
        self,
        kind: str,
        question: str | None = None,
        params: dict[str, Any] | None = None,
        *,
        source: str = "person",
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        spec = CATALOG.get(kind)
        if spec is None:
            raise ValueError(f"unknown research job {kind!r} (one of: {', '.join(sorted(CATALOG))})")
        params = params or {}
        now = self._clock.now()
        last = await self.queue.last_finished(kind, job_key(kind, params))
        age = now - datetime.fromisoformat(last["finished_at"]) if last and last.get("finished_at") else None
        score, detail = priority(
            spec.value,
            spec.cost,
            last_status=await self._uncertainty(spec),
            age=age,
            refresh=spec.refresh,
            source=source,
        )
        return await self.queue.enqueue(
            kind=kind,
            question=question or spec.question,
            params=params,
            cost=spec.cost,
            priority_=score,
            priority_detail=detail,
            source=source,
            now=now,
            parent_id=parent_id,
        )

    async def seed(self, now: datetime) -> list[str]:
        """Queue every standing question whose last answer is older than its refresh interval."""
        if self._last_seed is not None and now - self._last_seed < SEED_EVERY:
            return []
        self._last_seed = now
        owns = self._s.brain_owns_account and self._brain.trading.broker.configured()
        queued: list[str] = []
        for spec in CATALOG.values():
            if spec.refresh is None or (spec.owner_only and not owns):
                continue
            key = job_key(spec.kind, {})
            if await self.queue.is_open(key):
                continue
            last = await self.queue.last_finished(spec.kind, key)
            if (
                last
                and last.get("finished_at")
                and now - datetime.fromisoformat(last["finished_at"]) < spec.refresh
            ):
                continue
            await self.ask(spec.kind, source="system")
            queued.append(spec.kind)
        return queued

    # ------------------------------------------------------------------ the tick
    async def tick(self) -> str:
        s = self._s
        if not s.research_enabled:
            return "disabled (QP_RESEARCH_ENABLED=false)"
        now = self._clock.now()
        if self._brain.supervisor.stopping:
            await self.halt("the process is stopping")
            return "stopping"
        lease = self._brain.trading.lease
        if lease is not None and not await lease.held():
            await self.halt("this process does not supervise the Brain")
            return "standby: research runs only in the process that supervises the Brain"
        # restart recovery: jobs left running by a stopped process (never one this process is running)
        recovered = await self.queue.recover(
            now, self.holder if self._recovered else "", STALE, keep=frozenset(self._tasks)
        )
        self._recovered = True
        state = await self._brain.store.get_state("supervisor") or {}
        if state.get("paused"):
            await self.halt("the supervisor is paused")
            return "paused with the supervisor"
        costs = research_costs(now)
        if costs is None:
            stopped = await self.halt(f"{mode_at(now)}: execution has priority")
            return f"{mode_at(now)}: no research (execution and safety first)" + (
                f"; {stopped} stopped" if stopped else ""
            )
        stop, why = self.governor.must_stop()
        if stop:
            await self.halt(why, charge=True)  # a job that keeps running memory short eventually fails
            return why
        seeded = await self.seed(now)
        started: list[str] = []
        held: str | None = None
        while len(self._tasks) < s.research_max_concurrent:
            job = await self._pick(now, costs)
            if job is None:
                break
            ok, held = self.governor.may_start(job["cost"])
            if not ok:
                break
            if await self.queue.claim(job["id"], self.holder, now):
                job = await self.queue.get(job["id"]) or job
                self._tasks[job["id"]] = asyncio.create_task(self._run(job), name=f"research-{job['id']}")
                started.append(f"{job['kind']}#{job['id']}")
        self.last_tick = {
            "at": now.isoformat(),
            "mode": mode_at(now),
            "recovered": recovered,
            "seeded": seeded,
            "started": started,
            "running": sorted(self._tasks),
            "held": held if not started else None,
        }
        parts = [f"{len(self._tasks)} running"]
        if started:
            parts.append("started " + ", ".join(started))
        if recovered:
            parts.append(f"recovered {len(recovered)} interrupted job(s)")
        if held and not started:
            parts.append(held)
        return "; ".join(parts)

    async def _pick(self, now: datetime, costs: tuple[str, ...]) -> dict[str, Any] | None:
        owns = self._s.brain_owns_account and self._brain.trading.broker.configured()
        for job in await self.queue.next_jobs(now, limit=10, costs=costs):
            spec = CATALOG.get(job["kind"])
            if spec is None:
                await self.queue.cancel(job["id"], now, "no such research job any more")
                continue
            if spec.owner_only and not owns:
                await self.queue.cancel(job["id"], now, "needs the Brain to own the Alpaca paper account")
                continue
            if job["id"] in self._tasks:
                continue
            return job
        return None

    # ------------------------------------------------------------------ running a job
    async def _heartbeat(self, job_id: int) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            with contextlib.suppress(Exception):
                await self.queue.heartbeat(job_id, self._clock.now())

    async def _run(self, job: dict[str, Any]) -> None:
        spec = CATALOG[job["kind"]]
        ctx = JobContext(
            job=job,
            brain=self._brain,
            settings=self._s,
            clock=self._clock,
            ledger=self.ledger,
            lifecycle=self.lifecycle,
            reference=self._reference,
            market=self._market,
        )
        timeout = (spec.timeout or timedelta(minutes=self._s.research_job_timeout_minutes)).total_seconds()
        peak = self.governor.snapshot().rss_mb
        beat = asyncio.create_task(self._heartbeat(job["id"]))
        try:
            result = await asyncio.wait_for(spec.handler(ctx), timeout)
            peak = max(peak, self.governor.snapshot().rss_mb)
            out = {"result": _brief(result), "learned": ctx.learned, "follow_ups": ctx.follow_ups}
            await self.queue.finish(
                job["id"], status="done", result=out, now=self._clock.now(), peak_rss_mb=peak
            )
            for f in ctx.follow_ups:
                with contextlib.suppress(ValueError):
                    await self.ask(
                        f["kind"], f["question"], f["params"], source="follow_up", parent_id=job["id"]
                    )
            log_event(
                logger,
                "research.done",
                f"research {job['kind']}#{job['id']}: done",
                learned=len(ctx.learned),
                follow_ups=len(ctx.follow_ups),
            )
        except asyncio.CancelledError:
            reason, charge = self._why_stopped.pop(job["id"], ("interrupted", True))
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    self.queue.requeue(
                        job["id"],
                        reason=f"stopped: {reason}",
                        now=self._clock.now(),
                        delay=RETRY_AFTER["interrupted"],
                        charge=charge,
                    )
                )
            raise
        except TimeoutError:
            await self.queue.requeue(
                job["id"],
                reason=f"timed out after {timeout:.0f} s",
                now=self._clock.now(),
                delay=RETRY_AFTER["timeout"],
            )
        except Exception as exc:
            logger.warning("research job %s#%s failed: %s", job["kind"], job["id"], exc, exc_info=True)
            await self.queue.requeue(
                job["id"],
                reason=f"{type(exc).__name__}: {exc}"[:500],
                now=self._clock.now(),
                delay=RETRY_AFTER["error"],
            )
        finally:
            beat.cancel()
            self._tasks.pop(job["id"], None)
            self._why_stopped.pop(job["id"], None)

    async def halt(self, reason: str, *, charge: bool = False) -> int:
        """Stop every running job (each is queued again): the market opened, memory is short, or a stop. Only a
        memory stop counts as one of the job's attempts."""
        running = list(self._tasks.items())
        for job_id, task in running:
            self._why_stopped[job_id] = (reason, charge)
            task.cancel()
        for _, task in running:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if running:
            log_event(logger, "research.halted", f"research stopped: {reason}", jobs=len(running))
        return len(running)

    async def idle(self) -> None:
        """Wait for the running jobs to finish (tests; a deliberate drain)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)

    async def run_now(self, job_id: int) -> dict[str, Any] | None:
        """Run one queued job to completion in this call (a person's request through the API, or a test)."""
        if not await self.queue.claim(job_id, self.holder, self._clock.now()):
            return await self.queue.get(job_id)
        job = await self.queue.get(job_id)
        assert job is not None
        task = asyncio.create_task(self._run(job))
        self._tasks[job_id] = task
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return await self.queue.get(job_id)

    def running(self) -> list[int]:
        return sorted(self._tasks)
