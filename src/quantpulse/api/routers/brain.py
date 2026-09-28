"""The QuantPulse brain: agents, cycles (what it saw, which agents it ran, their findings, the consensus and
the decisions with the risk engine's verdict), execution and memory.

``POST /brain/run`` runs one cycle. With ``QP_BRAIN_MODE=paper_execution`` the Brain owns the Alpaca
**paper** account and its decisions are executed by the trading service (reconciliation, fresh quotes, the
risk engine, the order manager and every trading switch); ``POST /brain/kill-switch`` stops new Brain
orders at once. In the other modes nothing reaches Alpaca.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from quantpulse.api.deps import ContainerDep
from quantpulse.api.routers.trading import LOOPBACK
from quantpulse.schemas.brain import (
    AgentToggleIn,
    BookResetIn,
    BrainAgentOut,
    BrainCycleOut,
    BrainCycleSummaryOut,
    BrainMemoryOut,
    BrainOpportunityOut,
    BrainRunIn,
    BrainStatusOut,
    ImprovementDecisionIn,
    StrategyIn,
    StrategyStatusIn,
    SupervisorIn,
)
from quantpulse.schemas.jobs import JobOut
from quantpulse.schemas.trading import KillSwitchIn, KillSwitchOut
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
    summary="Run one brain cycle now (in paper_execution its decisions go to the trading service)",
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


@router.get(
    "/opportunities", response_model=list[BrainOpportunityOut], summary="Detected opportunities, newest first"
)
async def opportunities(
    kind: str | None = Query(None, max_length=32),
    status: str | None = Query(None, max_length=24),
    limit: int = Query(100, ge=1, le=1000),
    c: Container = ContainerDep,
) -> list[BrainOpportunityOut]:
    rows = await c.brain.store.opportunities(kind=kind, status=status, limit=limit)
    return [BrainOpportunityOut.model_validate(r) for r in rows]


# ---------------------------------------------------------------------------------------------- learning
@router.post(
    "/learn",
    responses=RUNNING,  # type: ignore[arg-type]
    dependencies=ControlAuth,
    summary="Grade matured predictions against real prices, reflect on decisions, update track records",
)
async def learn(wait: float = Query(60.0, ge=0, le=600), c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.learn(wait=wait)


@router.get("/learning", summary="Predictions (open / graded), last learning pass, consensus calibration")
async def learning(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.learning()


@router.get("/performance", summary="Measured track records (only from graded predictions)")
async def performance(
    window: str | None = Query(None, pattern="^(all|90d)$"), c: Container = ContainerDep
) -> list[dict[str, Any]]:
    return await c.brain.store.performance(window)


@router.get("/reflections", summary="Decision-vs-outcome reflections and failure analyses, newest first")
async def reflections(
    category: str | None = Query(None, max_length=32),
    limit: int = Query(100, ge=1, le=1000),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.store.reflections(category=category, limit=limit)


# ---------------------------------------------------------------------------------------------- operation
@router.get("/supervisor", summary="The supervisor: session, schedule, event wake-ups, recent work")
async def supervisor(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.supervisor.status()


@router.post("/supervisor", dependencies=ControlAuth, summary="Pause or resume the supervisor")
async def pause_supervisor(body: SupervisorIn, c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.supervisor.set_paused(body.paused)


@router.get("/execution", summary="Who owns the account, the Brain kill switch, what would stop Brain orders")
async def execution(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.execution_status()


@router.get(
    "/positions", summary="The Alpaca account's positions and their theses (open and recently closed)"
)
async def positions(closed: int = Query(50, ge=0, le=500), c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.theses.positions(closed=closed)


@router.post(
    "/positions/{symbol}/adopt",
    dependencies=ControlAuth,
    summary="Adopt a position the Brain did not open (it then manages it, and new positions may resume)",
)
async def adopt(
    symbol: str = Path(..., max_length=24, pattern=r"^[A-Za-z0-9.\-]+$"), c: Container = ContainerDep
) -> dict[str, Any]:
    return await c.brain.theses.adopt(symbol)


@router.get("/sessions", summary="Trading days of the Alpaca paper account: pre-market checks and closes")
async def sessions(limit: int = Query(60, ge=1, le=500), c: Container = ContainerDep) -> dict[str, Any]:
    return {"sessions": await c.brain.sessions.sessions(limit)}


@router.get("/trades", summary="Recent trade decisions and how far each got (newest first)")
async def trades(limit: int = Query(50, ge=1, le=500), c: Container = ContainerDep) -> list[dict[str, Any]]:
    from quantpulse.brain.audit import trades as recent

    return await recent(c.brain.db, limit)


@router.get(
    "/decisions/{decision_id}/audit",
    summary="One decision's full trail: opportunity → data → agents → … → fill → position → outcome → learning",
)
async def audit(decision_id: int = Path(..., ge=1), c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.brain.audit import trail
    from quantpulse.core.errors import NotFoundError

    found = await trail(c.brain.db, decision_id)
    if found is None:
        raise NotFoundError(f"decision {decision_id} not found")
    return found


@router.get("/data-report", summary="How often market data stopped the Brain, and what SIP data would change")
async def data_report(days: int = Query(20, ge=1, le=365), c: Container = ContainerDep) -> dict[str, Any]:
    from datetime import timedelta

    from quantpulse.brain.data_report import data_report as report

    now = c.clock.now()
    return await report(c.brain.db, c.settings, now - timedelta(days=days), now)


@router.get(
    "/evaluation", summary="The 60-session evaluation: the Brain vs the benchmark vs the replaced strategy"
)
async def evaluation(c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.brain.scorecard import evaluation as report

    return await report(c.brain.db, c.settings)


@router.get(
    "/scorecard",
    summary="Learning measured separately: accuracy, calibration, decisions, luck, execution, risk",
)
async def scorecard(c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.brain.scorecard import scorecard as card

    return await card(c.brain.db, c.settings)


@router.get(
    "/execution-quality", summary="The Brain's real Alpaca paper fills: fill rate, slippage, time to fill"
)
async def execution_quality(c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.brain.scorecard import execution_quality as quality

    return await quality(c.brain.db)


@router.get("/shadow", summary="The replaced strategy's shadow portfolio (hypothetical; never an order)")
async def shadow(c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.brain.shadow import equity_of

    state = await c.brain.shadow.state()
    if state is None:
        return {"started": False, "note": "starts on the first session the Brain owns the account"}
    return {"started": True, "equity": round(equity_of(state), 2), **state}


@router.get(
    "/executions", summary="The execution ledger: every Brain order from the decision to its final state"
)
async def executions(limit: int = Query(100, ge=1, le=1000), c: Container = ContainerDep) -> dict[str, Any]:
    await c.brain.ledger.refresh()
    return {"executions": await c.brain.ledger.rows(limit)}


@router.get(
    "/execution-audit", summary="The latest final execution audits (every gate, and what was about to go)"
)
async def execution_audit(c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.brain.execution import AUDIT_KEY, AUDITS_KEY

    return {
        "latest": await c.brain.store.get_state(AUDIT_KEY),
        "history": ((await c.brain.store.get_state(AUDITS_KEY)) or {}).get("items", [])[::-1][:10],
    }


@router.post(
    "/execution-audit",
    dependencies=ControlAuth,
    summary="Run the execution audit now (checks and reports; reconciles with Alpaca; never sends an order)",
)
async def run_execution_audit(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.executor.audit(purpose="on_demand")


@router.get("/kill-switch", response_model=KillSwitchOut, summary="The Brain kill switch")
async def brain_kill_switch(c: Container = ContainerDep) -> KillSwitchOut:
    return await c.trading.brain_kill_switch()


@router.post(
    "/kill-switch",
    response_model=KillSwitchOut,
    dependencies=ControlAuth,
    summary="Stop new Brain-originated orders immediately (or allow them again)",
)
async def set_brain_kill_switch(body: KillSwitchIn, c: Container = ContainerDep) -> KillSwitchOut:
    return await c.trading.set_brain_kill_switch(body.active, body.reason, body.cancel_open_orders)


@router.get("/book", summary="The Brain's paper book: positions, simulated fills, equity curve, performance")
async def book(trades: int = Query(100, ge=1, le=2000), c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.book.view(trades_limit=trades)


@router.post(
    "/book/reset", dependencies=ControlAuth, summary="Start the paper book again (deletes its history)"
)
async def reset_book(body: BookResetIn, c: Container = ContainerDep) -> dict[str, Any]:
    from quantpulse.core.errors import DomainError

    if body.confirm.strip() != "RESET BOOK":
        raise DomainError('type exactly "RESET BOOK" to delete the paper book\'s history')
    await c.brain.book.reset()
    return await c.brain.book.view(trades_limit=10)


@router.get(
    "/models", summary="Language models: provider, tiers, today's token budget and usage, recent calls"
)
async def models(c: Container = ContainerDep) -> dict[str, Any]:
    return c.brain.models.status()


@router.get("/events", summary="What happened (market, portfolio, orders, agents, learning), newest first")
async def events(
    type: str | None = Query(None, max_length=40),
    subject: str | None = Query(None, max_length=24),
    limit: int = Query(100, ge=1, le=1000),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.bus.history(type=type, subject=subject, limit=limit)


# ---------------------------------------------------------------------------------------------- strategy lab
@router.get("/lab/templates", summary="Strategy templates the lab can propose")
async def lab_templates() -> dict[str, Any]:
    from quantpulse.brain.lab.spec import GRID, TEMPLATES

    return {"templates": TEMPLATES, "default_grid": GRID}


@router.get(
    "/lab/strategies", summary="Strategy versions in the lab, with their validation and paper results"
)
async def lab_strategies(
    status: str | None = Query(None, pattern="^(proposed|validated|rejected|paper|promoted|retired)$"),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.lab.strategies(status)


@router.post(
    "/lab/propose", dependencies=ControlAuth, summary="Propose every template not tried yet (version 1)"
)
async def lab_propose(c: Container = ContainerDep) -> list[dict[str, Any]]:
    return await c.brain.lab.propose()


@router.post("/lab/strategies", dependencies=ControlAuth, summary="Create a new version of a template")
async def lab_create(body: StrategyIn, c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.lab.new_version(body.template, "user", **body.overrides())


@router.get("/lab/strategies/{strategy_id}/{version}", summary="One version with its runs")
async def lab_strategy(
    strategy_id: str = Path(..., max_length=48), version: int = Path(..., ge=1), c: Container = ContainerDep
) -> dict[str, Any]:
    return await c.brain.lab.get(strategy_id, version)


@router.post(
    "/lab/strategies/{strategy_id}/{version}/validate",
    responses=RUNNING,  # type: ignore[arg-type]
    dependencies=ControlAuth,
    summary="Backtest, walk-forward, overfitting checks and stress tests (202 while it runs)",
)
async def lab_validate(
    strategy_id: str = Path(..., max_length=48),
    version: int = Path(..., ge=1),
    wait: float = Query(120.0, ge=0, le=600),
    c: Container = ContainerDep,
) -> dict[str, Any]:
    return await c.brain.validate_strategy(strategy_id, version, wait=wait)


@router.post(
    "/lab/strategies/{strategy_id}/{version}/status",
    dependencies=ControlAuth,
    summary="Start paper tracking, promote (only if validated and paper-tracked long enough) or retire",
)
async def lab_status(
    body: StrategyStatusIn,
    strategy_id: str = Path(..., max_length=48),
    version: int = Path(..., ge=1),
    c: Container = ContainerDep,
) -> dict[str, Any]:
    return await c.brain.lab.set_status(strategy_id, version, body.status, "user")


@router.post("/lab/paper", dependencies=ControlAuth, summary="Update the paper (shadow) portfolios now")
async def lab_paper(c: Container = ContainerDep) -> dict[str, Any]:
    return await c.brain.lab.paper_update()


@router.get("/lab/compare", summary="Versions side by side")
async def lab_compare(
    keys: str = Query(
        ..., max_length=500, description="Comma-separated, e.g. momentum_12_1@v1,momentum_12_1@v2"
    ),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.lab.compare([k.strip() for k in keys.split(",") if k.strip()])


# ---------------------------------------------------------------------------------------------- improvement
@router.get("/improvements", summary="Improvement proposals (problem, evidence, change, validation plan)")
async def improvements(
    status: str | None = Query(None, pattern="^(proposed|testing|validated|rejected|applied)$"),
    c: Container = ContainerDep,
) -> list[dict[str, Any]]:
    return await c.brain.improvements.proposals(status)


@router.post(
    "/improvements/review", dependencies=ControlAuth, summary="Analyse the record and propose improvements"
)
async def review_improvements(c: Container = ContainerDep) -> list[dict[str, Any]]:
    return await c.brain.improvements.review(c.clock.now())


@router.post(
    "/improvements/{improvement_id}",
    dependencies=ControlAuth,
    summary="Record a person's decision on a proposal (nothing is applied automatically)",
)
async def decide_improvement(
    body: ImprovementDecisionIn, improvement_id: int = Path(..., ge=1), c: Container = ContainerDep
) -> dict[str, Any]:
    from quantpulse.core.errors import NotFoundError

    try:
        return await c.brain.improvements.decide(
            improvement_id, body.status, "user", body.note, c.clock.now()
        )
    except KeyError:
        raise NotFoundError(f"improvement {improvement_id} not found") from None
