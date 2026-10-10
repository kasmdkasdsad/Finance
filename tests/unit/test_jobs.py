import asyncio
from datetime import UTC, datetime

import pytest

from quantpulse.core.clock import FakeClock
from quantpulse.core.jobs import JobPending, JobRegistry, JobStatus


async def test_jobs_are_keyed_report_progress_and_can_be_awaited():
    clock = FakeClock(datetime(2026, 9, 25, tzinfo=UTC))
    jobs = JobRegistry(clock)
    gate = asyncio.Event()

    async def work(job):
        job.update(0.25, "downloading")
        report = job.reporter(0.5, 1.0)
        report(0.5, "training")
        await gate.wait()
        return 42

    job = jobs.start("model", "k1", "Model", work)
    assert jobs.start("model", "k1", "Model", work) is job  # the same work is joined, not repeated
    await asyncio.sleep(0)
    assert (job.progress, job.stage) == (0.75, "training")
    job.update(0.1)  # progress never goes backwards
    assert job.progress == 0.75
    with pytest.raises(JobPending) as pending:
        await jobs.wait(job, 0)
    assert pending.value.job is job and "75%" in str(pending.value)
    with pytest.raises(JobPending):
        await jobs.wait(job, 0.01)
    gate.set()
    assert await jobs.wait(job, 5) == 42
    assert job.status is JobStatus.DONE and job.progress == 1.0 and job.finished_at is not None
    assert jobs.running("k1") is None and jobs.latest("k1") is job and jobs.get(job.id) is job
    again = jobs.start("model", "k1", "Model", work)
    assert again is not job and jobs.list()[0] is again
    await jobs.wait(again, 5)


async def test_failed_and_cancelled_jobs():
    jobs = JobRegistry(FakeClock())

    async def boom(job):
        raise ValueError("bad input")

    job = jobs.start("research", "k", "Research", boom)
    with pytest.raises(ValueError, match="bad input"):
        await jobs.wait(job, 5)
    assert job.status is JobStatus.FAILED and job.error == "bad input"

    async def forever(job):
        await asyncio.Event().wait()

    slow = jobs.start("model", "k2", "Slow", forever)
    await asyncio.sleep(0)
    await jobs.shutdown()
    assert slow.status is JobStatus.FAILED and slow.error == "cancelled"
