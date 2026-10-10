"""The Market Evolution Monitor and the versioned model registry.

Changes are reported only after a false-discovery-rate control across everything measured, each with its
competing explanations (none assumed — automated liquidity provision included, which prices alone cannot
identify). Relationship estimates are appended, never overwritten. The registry shows every model's stage
and evidence; in-sample results are displayed for reference and never used by a gate.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from quantpulse.api.deps import ContainerDep, local_only_allowed
from quantpulse.services.container import Container

router = APIRouter(tags=["market evolution"])
APPROVE_PHRASE = "I APPROVE THIS MODEL"


async def control_allowed(request: Request) -> None:
    local_only_allowed(request, "evolution and registry controls")


ControlAuth = [Depends(control_allowed)]


class ApproveIn(BaseModel):
    by: str = Field(..., min_length=2, max_length=64, description="Who approves (recorded)")
    confirm: str = Field(..., description=f"Type exactly: {APPROVE_PHRASE}")


@router.get("/evolution/status", summary="How much of the market has been measured, and the last scan")
async def evolution_status(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.evolution.status()


@router.get("/evolution/changes", summary="Detected structural changes with their competing hypotheses")
async def evolution_changes(
    limit: int = Query(100, ge=1, le=1000), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.evolution.changes(limit)


@router.get("/evolution/relationships", summary="Relationship estimates over time (every estimate kept)")
async def evolution_relationships(
    limit: int = Query(200, ge=1, le=2000), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.evolution.relationships(limit)


@router.post("/evolution/run", dependencies=ControlAuth, summary="Measure and scan now (background)")
async def evolution_run(c: Container = ContainerDep) -> dict[str, Any]:
    job = c.evolution.start()
    return {"job_id": job.id, "status": job.status}


@router.get("/registry/models", summary="Every model version, its stage, role and evidence")
async def registry_models(
    slot: str | None = Query(None), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    await c.registry.bootstrap()
    return await c.registry.models(slot)


@router.post("/registry/models/{model_id}/advance", dependencies=ControlAuth,
             summary="Advance a model as far as its (out-of-sample) evidence allows")  # fmt: skip
async def registry_advance(model_id: int = Path(..., ge=1), c: Container = ContainerDep) -> dict[str, Any]:
    try:
        return await c.registry.advance(model_id)
    except KeyError:
        raise HTTPException(404, f"no model {model_id}") from None


@router.post("/registry/models/{model_id}/approve", dependencies=ControlAuth,
             summary="A person's approval (needed before an AI model can become authoritative)")  # fmt: skip
async def registry_approve(
    body: ApproveIn, model_id: int = Path(..., ge=1), c: Container = ContainerDep
) -> dict[str, Any]:
    if body.confirm != APPROVE_PHRASE:
        raise HTTPException(400, f"type exactly: {APPROVE_PHRASE}")
    try:
        return await c.registry.approve(model_id, body.by)
    except KeyError:
        raise HTTPException(404, f"no model {model_id}") from None
