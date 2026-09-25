"""Background jobs with progress, for work that takes longer than an HTTP request should wait.

A job is keyed by what it computes, so asking twice for the same work joins the running job instead of
starting a second one. Callers can wait for a job with a timeout and, when it is not done yet, hand the
job's live progress back to the client (the API answers ``202 Accepted`` with the job, which the UI
polls). Finished jobs are kept briefly so their outcome can still be read.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError

logger = logging.getLogger(__name__)

KEEP_FINISHED = 50


class JobStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    kind: str
    key: str
    description: str
    created_at: datetime
    status: JobStatus = JobStatus.RUNNING
    progress: float = 0.0
    stage: str = "starting"
    finished_at: datetime | None = None
    error: str | None = None
    result: Any = field(default=None, repr=False)
    task: asyncio.Task[Any] | None = field(default=None, repr=False)

    def update(self, progress: float, stage: str | None = None) -> None:
        """Report progress (0-1). Safe to call from worker threads: plain attribute writes."""
        self.progress = min(max(float(progress), self.progress), 1.0)
        if stage:
            self.stage = stage

    def reporter(self, lo: float, hi: float) -> Callable[[float, str], None]:
        """A progress callback mapping a sub-task's 0-1 onto ``[lo, hi]`` of this job."""

        def report(fraction: float, stage: str) -> None:
            self.update(lo + (hi - lo) * min(max(fraction, 0.0), 1.0), stage)

        return report

    @property
    def done(self) -> bool:
        return self.status is not JobStatus.RUNNING


class JobPending(Exception):
    """The requested result is being computed by ``job``."""

    def __init__(self, job: Job) -> None:
        super().__init__(f"{job.description}: {job.progress:.0%} ({job.stage})")
        self.job = job


class JobRegistry:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._by_key: dict[str, str] = {}
        self._ids = itertools.count(1)

    def start(self, kind: str, key: str, description: str, work: Callable[[Job], Awaitable[Any]]) -> Job:
        """Start ``work`` in the background, or return the job already computing ``key``."""
        running = self.running(key)
        if running is not None:
            return running
        job = Job(
            id=f"{kind}-{next(self._ids)}",
            kind=kind,
            key=key,
            description=description,
            created_at=self._clock.now(),
        )
        self._jobs[job.id] = job
        self._by_key[key] = job.id
        job.task = asyncio.create_task(self._run(job, work))
        self._trim()
        return job

    async def _run(self, job: Job, work: Callable[[Job], Awaitable[Any]]) -> Any:
        try:
            job.result = await work(job)
        except asyncio.CancelledError:
            job.status, job.error, job.finished_at = JobStatus.FAILED, "cancelled", self._clock.now()
            raise
        except DomainError as exc:  # an expected refusal (e.g. not enough history): no traceback
            logger.warning("job %s failed: %s", job.id, exc)
            job.status, job.error, job.finished_at = JobStatus.FAILED, str(exc), self._clock.now()
            raise
        except Exception as exc:
            logger.exception("job %s failed", job.id)
            job.status, job.error, job.finished_at = (
                JobStatus.FAILED,
                str(exc) or type(exc).__name__,
                self._clock.now(),
            )
            raise
        job.status, job.progress, job.stage, job.finished_at = JobStatus.DONE, 1.0, "done", self._clock.now()
        return job.result

    async def wait(self, job: Job, timeout: float | None) -> Any:
        """The job's result; raises :class:`JobPending` if it is still running after ``timeout`` seconds
        and re-raises the job's own error if it failed."""
        assert job.task is not None
        if not job.task.done():
            if timeout is not None and timeout <= 0:
                raise JobPending(job)
            try:
                await asyncio.wait_for(asyncio.shield(job.task), timeout)
            except TimeoutError:
                raise JobPending(job) from None
        return job.task.result()

    def running(self, key: str) -> Job | None:
        job_id = self._by_key.get(key)
        job = self._jobs.get(job_id) if job_id else None
        return job if job is not None and not job.done else None

    def latest(self, key: str) -> Job | None:
        job_id = self._by_key.get(key)
        return self._jobs.get(job_id) if job_id else None

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(reversed(self._jobs.values()))

    def _trim(self) -> None:
        finished = [j for j in self._jobs.values() if j.done]
        for job in finished[: max(0, len(finished) - KEEP_FINISHED)]:
            del self._jobs[job.id]

    async def shutdown(self) -> None:
        tasks = [j.task for j in self._jobs.values() if j.task is not None and not j.task.done()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
