"""Background jobs: long model runs, signal research and track-record backfills."""

from __future__ import annotations

from fastapi import APIRouter

from quantpulse.api.deps import ContainerDep
from quantpulse.core.errors import NotFoundError
from quantpulse.schemas.jobs import JobOut
from quantpulse.services.container import Container

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobOut], summary="Recent background jobs, newest first")
async def list_jobs(c: Container = ContainerDep) -> list[JobOut]:
    now = c.clock.now()
    return [JobOut.of(j, now) for j in c.jobs.list()]


@router.get("/{job_id}", response_model=JobOut, summary="Progress of one background job")
async def get_job(job_id: str, c: Container = ContainerDep) -> JobOut:
    job = c.jobs.get(job_id)
    if job is None:
        raise NotFoundError(f"no job {job_id!r} (finished jobs are kept for a while only)")
    return JobOut.of(job, c.clock.now())
