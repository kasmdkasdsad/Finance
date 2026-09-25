"""Alpaca **paper** trading: account, positions, orders, strategy cycles, risk, controls.

Every order-producing endpoint goes through the strategy's risk engine and the order manager. There is no
endpoint (and no setting) that reaches a live-money account. Endpoints that can create or cancel orders
also require either ``QP_API_TOKEN`` (checked for every /api request) or a request from this machine.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from quantpulse.api.deps import ContainerDep
from quantpulse.schemas.jobs import JobOut
from quantpulse.schemas.trading import (
    ActionOut,
    BrokerAccountOut,
    BrokerOrderOut,
    BrokerPositionOut,
    CancelAllIn,
    CloseAllIn,
    CycleOut,
    CycleSummary,
    DiagnosticOrderIn,
    DiagnosticOrderOut,
    KillSwitchIn,
    KillSwitchOut,
    ReconcileOut,
    RiskSnapshotOut,
    TradingDiagnosticsOut,
    TradingEventOut,
    TradingPerformanceOut,
    TradingStatus,
)
from quantpulse.services.container import Container

router = APIRouter(prefix="/trading", tags=["trading"])
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
RUNNING = {202: {"model": JobOut, "description": "A cycle is running: poll /jobs/{id} or /trading/job"}}


async def orders_allowed(request: Request) -> None:
    """With ``QP_API_TOKEN`` set, the key was already verified for this request. Without one, order
    endpoints refuse anything that does not come from this machine."""
    container: Container = request.app.state.container
    if container.settings.api_token is not None:
        return
    host = request.client.host if request.client else None
    if host not in LOOPBACK:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="order endpoints accept remote requests only when QP_API_TOKEN is set",
        )


OrderAuth = [Depends(orders_allowed)]


@router.get("/status", response_model=TradingStatus, summary="Mode, kill switch, market clock, next cycle")
async def trading_status(c: Container = ContainerDep) -> TradingStatus:
    return await c.trading.status()


@router.get(
    "/diagnostics",
    response_model=TradingDiagnosticsOut,
    summary="Read-only check of the Alpaca paper connection (never places or cancels an order)",
)
async def diagnostics(
    symbols: str = Query(
        "",
        max_length=300,
        description="Comma-separated symbols whose quotes (price, bid/ask, spread source, age) to inspect",
    ),
    c: Container = ContainerDep,
) -> TradingDiagnosticsOut:
    return await c.trading.diagnostics([x for x in symbols.split(",") if x.strip()])


@router.post(
    "/test-order",
    response_model=DiagnosticOrderOut,
    dependencies=OrderAuth,
    summary="Send exactly ONE small paper test order (needs the exact confirmation phrase)",
)
async def send_test_order(body: DiagnosticOrderIn, c: Container = ContainerDep) -> DiagnosticOrderOut:
    return await c.trading.test_order(body.confirm, body.symbol, body.mode, body.notional)


@router.get("/account", response_model=BrokerAccountOut, summary="Alpaca paper account (authoritative)")
async def account(c: Container = ContainerDep) -> BrokerAccountOut:
    return await c.trading.account()


@router.get("/positions", response_model=list[BrokerPositionOut], summary="Alpaca paper positions")
async def positions(c: Container = ContainerDep) -> list[BrokerPositionOut]:
    return await c.trading.positions()


@router.get("/orders", response_model=list[BrokerOrderOut], summary="Orders, newest first")
async def orders(
    order_status: str = Query("all", alias="status", pattern="^(open|closed|all)$"),
    limit: int = Query(100, ge=1, le=500),
    c: Container = ContainerDep,
) -> list[BrokerOrderOut]:
    return await c.trading.order_list(order_status, limit)


@router.get(
    "/proposed",
    response_model=CycleOut | None,
    summary="Latest cycle: regime, top opportunities, target portfolio, proposed trades and risk decisions",
)
async def proposed(c: Container = ContainerDep) -> CycleOut | None:
    return await c.trading.proposed()


@router.get("/risk", response_model=RiskSnapshotOut, summary="Risk limits and where the account stands")
async def risk(c: Container = ContainerDep) -> RiskSnapshotOut:
    return await c.trading.risk()


@router.get("/cycles", response_model=list[CycleSummary], summary="Recent strategy cycles")
async def cycles(limit: int = Query(20, ge=1, le=500), c: Container = ContainerDep) -> list[CycleSummary]:
    return await c.trading.cycles(limit)


@router.get("/cycles/{cycle_id}", response_model=CycleOut, summary="One cycle in full")
async def cycle(cycle_id: int = Path(..., ge=1), c: Container = ContainerDep) -> CycleOut:
    return await c.trading.cycle(cycle_id)


@router.get("/events", response_model=list[TradingEventOut], summary="Audit trail, newest first")
async def events(
    limit: int = Query(200, ge=1, le=2000),
    kind: list[str] | None = Query(None, description="Filter by event kind (repeatable)"),
    c: Container = ContainerDep,
) -> list[TradingEventOut]:
    return await c.trading.events(limit, kind)


@router.get("/performance", response_model=TradingPerformanceOut, summary="Performance from recorded data")
async def performance(c: Container = ContainerDep) -> TradingPerformanceOut:
    return await c.trading.performance()


@router.get("/job", response_model=JobOut | None, summary="The latest trading-cycle job")
async def job(c: Container = ContainerDep) -> JobOut | None:
    j = c.trading.cycle_job()
    return JobOut.of(j, c.clock.now()) if j is not None else None


@router.post(
    "/run",
    response_model=CycleOut,
    responses=RUNNING,  # type: ignore[arg-type]
    dependencies=OrderAuth,
    summary="Run one strategy cycle now (orders only in paper-execution mode; every order is risk-checked)",
)
async def run(
    dry_run: bool = Query(False, description="Force a dry run even when paper execution is enabled"),
    wait: float = Query(30.0, ge=0, le=600, description="Seconds to wait before answering 202 with progress"),
    c: Container = ContainerDep,
) -> CycleOut:
    return await c.trading.run(trigger="manual", dry_run=dry_run, wait=wait)


@router.post("/reconcile", response_model=ReconcileOut, summary="Reconcile local records with Alpaca now")
async def reconcile(c: Container = ContainerDep) -> ReconcileOut:
    return await c.trading.reconcile("manual")


@router.post(
    "/kill-switch",
    response_model=KillSwitchOut,
    dependencies=OrderAuth,
    summary="Turn the kill switch on (no new orders) or off",
)
async def kill_switch(body: KillSwitchIn, c: Container = ContainerDep) -> KillSwitchOut:
    return await c.trading.set_kill_switch(body.active, body.reason, body.cancel_open_orders)


@router.post(
    "/cancel-all",
    response_model=ActionOut,
    dependencies=OrderAuth,
    summary="Cancel every open order on the Alpaca paper account (confirm=true)",
)
async def cancel_all(body: CancelAllIn, c: Container = ContainerDep) -> ActionOut:
    return await c.trading.cancel_all(body.confirm)


@router.post(
    "/close-all",
    response_model=ActionOut,
    dependencies=OrderAuth,
    summary="Sell every position — requires the exact phrase 'CLOSE ALL' (a preview in dry-run mode)",
)
async def close_all(body: CloseAllIn, c: Container = ContainerDep) -> ActionOut:
    return await c.trading.close_all(body.confirm)
