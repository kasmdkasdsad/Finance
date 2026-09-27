"""The QuantPulse brain: agents, cycles (what it saw, which agents it ran, their findings, the consensus and
the proposed actions with the risk engine's verdict) and memory.

The brain never sends orders. ``POST /brain/run`` runs one analysis cycle; its proposals are checked by the
deterministic risk engine and recorded — nothing reaches the Alpaca paper account.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from quantpulse.api.deps import ContainerDep
from quantpulse.api.routers.trading import LOOPBACK
from quantpulse.schemas.brain import (
    AgentToggleIn,
    BrainAgentOut,
    BrainCycleOut,
    BrainCycleSummaryOut,
    BrainMemoryOut,
    BrainRunIn,
    BrainStatusOut,
)
from quantpulse.schemas.jobs import JobOut
from quantpulse.services.container import Container

router = APIRouter(prefix="/brain", tags=["brain"])
RUNNING = {202: {"model": JobOut, "description": "A cycle is running: poll /jobs/{id}"}}


async def control_allowed(request: Request) -> None:
    """Running a cycle or switching an agent on or off: like the trading order endpoints, only from this
    machine unless ``QP_API_TOKEN`` is set (and was verified for this request)."""
    container: Container = request.app.state.container
    if container.settings.api_token is not None:
        return
    host = request.client.host if request.client else None
    if host not in LOOPBACK:
        raise HTTPException(
            status_code=403,
            detail="brain controls accept remote requests only when QP_API_TOKEN is set",
        )


ControlAuth = [Depends(control_allowed)]


@router.get("/status", response_model=BrainStatusOut, summary="Mode, agents, last cycle, open predictions")
async def brain_status(c: Container = ContainerDep) -> BrainStatusOut:
    return BrainStatusOut.model_validate(await c.brain.status())


@router.get(
    "/agents", response_model=list[BrainAgentOut], summary="Registered agents, their runs and measured record"
)
async def agents(c: Container = ContainerDep) -> list[BrainAgentOut]:
    return [BrainAgentOut.model_validate(a) for a in await c.brain.agents()]


@router.get("/agents/{agent_id}", response_model=BrainAgentOut, summary="One agent")
async def agent(agent_id: str = Path(..., max_length=48), c: Container = ContainerDep) -> BrainAgentOut:
    for a in await c.brain.agents():
        if a["id"] == agent_id:
            return BrainAgentOut.model_validate(a)
    from quantpulse.core.errors import NotFoundError

    raise NotFoundError(f"agent {agent_id!r} not found")


@router.post(
    "/agents/{agent_id}",
    response_model=BrainAgentOut,
    dependencies=ControlAuth,
    summary="Enable or disable an agent",
)
async def toggle_agent(
    body: AgentToggleIn, agent_id: str = Path(..., max_length=48), c: Container = ContainerDep
) -> BrainAgentOut:
    return BrainAgentOut.model_validate(await c.brain.set_agent_enabled(agent_id, body.enabled))


@router.post(
    "/run",
    response_model=BrainCycleOut,
    responses=RUNNING,  # type: ignore[arg-type]
    dependencies=ControlAuth,
    summary="Run one brain cycle now (analysis and proposals only — never sends an order)",
)
async def run(
    body: BrainRunIn | None = None,
    wait: float = Query(60.0, ge=0, le=600, description="Seconds to wait before answering 202 with progress"),
    c: Container = ContainerDep,
) -> BrainCycleOut:
    body = body or BrainRunIn()
    return BrainCycleOut.model_validate(await c.brain.run(kind=body.kind, symbols=body.symbols, wait=wait))


@router.get("/cycles", response_model=list[BrainCycleSummaryOut], summary="Recent cycles, newest first")
async def cycles(
    limit: int = Query(20, ge=1, le=200), c: Container = ContainerDep
) -> list[BrainCycleSummaryOut]:
    return [BrainCycleSummaryOut.model_validate(x) for x in await c.brain.cycles(limit)]


@router.get("/cycles/{cycle_id}", response_model=BrainCycleOut, summary="One cycle in full")
async def cycle(cycle_id: int = Path(..., ge=1), c: Container = ContainerDep) -> BrainCycleOut:
    return BrainCycleOut.model_validate(await c.brain.cycle(cycle_id))


@router.get("/memory", response_model=list[BrainMemoryOut], summary="Structured memory, newest first")
async def memory(
    tier: str | None = Query(None, pattern="^(short_term|working|long_term|strategy|agent)$"),
    kind: str | None = Query(None, max_length=24),
    subject: str | None = Query(None, max_length=24),
    text: str | None = Query(None, max_length=100),
    limit: int = Query(50, ge=1, le=500),
    c: Container = ContainerDep,
) -> list[BrainMemoryOut]:
    rows = await c.brain.memories(tier=tier, kind=kind, subject=subject, text=text, limit=limit)
    return [BrainMemoryOut.model_validate(r) for r in rows]
