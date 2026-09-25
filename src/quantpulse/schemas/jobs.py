"""Background jobs (long model runs and backfills) and their progress."""

from __future__ import annotations

from datetime import datetime

from pydantic import AwareDatetime, Field

from quantpulse.core.jobs import Job
from quantpulse.schemas.common import StrictModel


class JobOut(StrictModel):
    id: str
    kind: str
    description: str
    status: str = Field(description="running, done or failed")
    progress: float = Field(ge=0, le=1)
    stage: str
    created_at: AwareDatetime
    finished_at: AwareDatetime | None
    elapsed_seconds: float
    error: str | None

    @classmethod
    def of(cls, job: Job, now: datetime) -> JobOut:
        end = job.finished_at or now
        return cls(
            id=job.id,
            kind=job.kind,
            description=job.description,
            status=job.status.value,
            progress=round(job.progress, 4),
            stage=job.stage,
            created_at=job.created_at,
            finished_at=job.finished_at,
            elapsed_seconds=round((end - job.created_at).total_seconds(), 1),
            error=job.error,
        )
