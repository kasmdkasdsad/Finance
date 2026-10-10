"""The Brain's 24/7 operating model and closed-market research: modes, execution readiness, the research queue,
the learning ledger and the improvement lifecycle. Reading is open to the API's callers; asking a question,
cancelling a job and promoting or rejecting a hypothesis are controls (a person, from this machine or with the
token). Nothing here sends an order or changes a trading setting."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Path, Query
from pydantic import Field

from quantpulse.api.deps import ContainerDep
from quantpulse.api.routers.brain import ControlAuth
from quantpulse.schemas.common import StrictModel
from quantpulse.services.container import Container

router = APIRouter(prefix="/brain/research", tags=["brain research"])


class QuestionIn(StrictModel):
    kind: str = Field(max_length=40, description="A research job kind (see /brain/research/catalog)")
    question: str | None = Field(None, max_length=500, description="Your wording of the question")
    params: dict[str, Any] = Field(default_factory=dict)


class DecisionIn(StrictModel):
    by: str = Field(min_length=2, max_length=48, description="Your name: promotions are a person's decision")
    note: str = Field(min_length=3, max_length=2000, description="Why the evidence is (or is not) enough")


@router.get(
    "/status", summary="Operating mode, execution readiness, research queue, resources, ledger, lifecycle"
)
async def status(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.research.status()


@router.get("/operating", summary="The operating mode and loop, and today's execution readiness")
async def operating(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.research.operating.status()


@router.get("/catalog", summary="Every kind of research job: question, phase, cost, value, refresh")
async def catalog(c: Container = ContainerDep) -> list[dict[str, Any]]:
    return c.brain.research.catalog()


@router.get("/jobs", summary="The research queue and the experiment history")
async def jobs(
    status: str | None = Query(None, pattern="^(queued|running|done|failed|cancelled)$"),
    kind: str | None = Query(None, max_length=40),
    limit: int = Query(100, ge=1, le=500),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.research.jobs(status=status, kind=kind, limit=limit)


@router.get("/jobs/{job_id}", summary="One research job, its result and what it concluded")
async def job(job_id: int = Path(..., ge=1), c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.research.job(job_id)


@router.post("/questions", dependencies=ControlAuth, summary="Ask the Brain a research question (queued)")
async def ask(body: QuestionIn, c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.research.ask(body.kind, body.question, body.params)


@router.post("/jobs/{job_id}/cancel", dependencies=ControlAuth, summary="Cancel a queued research job")
async def cancel(job_id: int = Path(..., ge=1), c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.research.cancel(job_id)


@router.get(
    "/learnings", summary="The learning ledger: conclusions with their evidence (UNPROVEN until enough)"
)
async def learnings(
    status: str | None = Query(None, pattern="^(UNPROVEN|SUPPORTED|REFUTED|INCONCLUSIVE)$"),
    topic: str | None = Query(None, max_length=160),
    current: bool = Query(True, description="Only the latest conclusion per topic"),
    limit: int = Query(200, ge=1, le=1000),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.research.learnings(status=status, topic=topic, current_only=current, limit=limit)


@router.get(
    "/hypotheses", summary="Improvements through DISCOVERED → … → EVALUATION → (a person) → PRODUCTION"
)
async def hypotheses(
    stage: str | None = Query(None, max_length=20),
    kind: str | None = Query(None, max_length=32),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.research.hypotheses(stage=stage, kind=kind)


@router.post(
    "/hypotheses/{hypothesis_id}/promote",
    dependencies=ControlAuth,
    summary="Promote an evaluated hypothesis to production (a person's decision; strategies pass the lab's gates)",
)
async def promote(
    body: DecisionIn, hypothesis_id: int = Path(..., ge=1), c: Container = ContainerDep
) -> dict[str, Any]:
    return await c.brain.research.promote(hypothesis_id, body.by, body.note)


@router.post("/hypotheses/{hypothesis_id}/reject", dependencies=ControlAuth, summary="Reject a hypothesis")
async def reject(
    body: DecisionIn, hypothesis_id: int = Path(..., ge=1), c: Container = ContainerDep
) -> dict[str, Any]:
    return await c.brain.research.reject(hypothesis_id, body.by, body.note)
