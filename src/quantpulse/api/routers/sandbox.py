"""Paper-trading sandbox: simulated accounts, a self-learning agent, manual orders and training.

No endpoint here ever places a real order; all cash, positions and fills are simulated.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Response, status

from quantpulse.api.deps import ContainerDep
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.sandbox import (
    AccountCreate,
    AccountOut,
    AccountSummary,
    AccountUpdate,
    EquityPoint,
    JournalEntry,
    OrderIn,
    StepResult,
    TradeOut,
    TrainReport,
    TrainRequest,
)
from quantpulse.services.container import Container

router = APIRouter(prefix="/sandbox", tags=["sandbox"])
AccountId = Path(..., ge=1)


@router.get("/accounts", response_model=list[AccountOut], summary="List paper-trading accounts")
async def list_accounts(c: Container = ContainerDep) -> list[AccountOut]:
    return await c.sandbox.list_all()


@router.post(
    "/accounts",
    response_model=AccountOut,
    status_code=status.HTTP_201_CREATED,
    summary="Open a paper account (agent or manual)",
)
async def create_account(data: AccountCreate, c: Container = ContainerDep) -> AccountOut:
    return await c.sandbox.create(data)


@router.get(
    "/accounts/{account_id}",
    response_model=CompositeEnvelope[AccountSummary],
    summary="Account with positions marked to live prices and performance",
)
async def get_account(
    account_id: int = AccountId, c: Container = ContainerDep
) -> CompositeEnvelope[AccountSummary]:
    return await c.sandbox.summary(account_id)


@router.patch("/accounts/{account_id}", response_model=AccountOut, summary="Rename, change mode or strategy")
async def update_account(
    data: AccountUpdate, account_id: int = AccountId, c: Container = ContainerDep
) -> AccountOut:
    return await c.sandbox.update(account_id, data)


@router.delete("/accounts/{account_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(account_id: int = AccountId, c: Container = ContainerDep) -> Response:
    await c.sandbox.delete(account_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/accounts/{account_id}/reset", response_model=AccountOut, summary="Back to starting cash")
async def reset_account(
    account_id: int = AccountId,
    keep_learning: bool = Query(False, description="Keep the agent's learned factor weights."),
    c: Container = ContainerDep,
) -> AccountOut:
    return await c.sandbox.reset(account_id, keep_learning=keep_learning)


@router.post(
    "/accounts/{account_id}/orders",
    response_model=TradeOut,
    status_code=status.HTTP_201_CREATED,
    summary="Place a manual paper market order",
)
async def place_order(order: OrderIn, account_id: int = AccountId, c: Container = ContainerDep) -> TradeOut:
    return await c.sandbox.order(account_id, order)


@router.post(
    "/accounts/{account_id}/step",
    response_model=StepResult,
    summary="Run the agent now: learn from the last decision, re-rank and rebalance",
)
async def step(
    account_id: int = AccountId,
    force: bool = Query(False, description="Trade even if it already traded today or the market is closed."),
    c: Container = ContainerDep,
) -> StepResult:
    return await c.sandbox.step(account_id, force=force)


@router.post(
    "/accounts/{account_id}/train",
    response_model=CompositeEnvelope[TrainReport],
    summary="Walk-forward training on history (no look-ahead)",
)
async def train(
    req: TrainRequest, account_id: int = AccountId, c: Container = ContainerDep
) -> CompositeEnvelope[TrainReport]:
    return await c.sandbox.train(account_id, req)


@router.get("/accounts/{account_id}/trades", response_model=list[TradeOut], summary="Fills, newest first")
async def trades(
    account_id: int = AccountId, limit: int = Query(200, ge=1, le=2000), c: Container = ContainerDep
) -> list[TradeOut]:
    return await c.sandbox.trades(account_id, limit)


@router.get("/accounts/{account_id}/equity", response_model=list[EquityPoint], summary="Equity snapshots")
async def equity(
    account_id: int = AccountId, limit: int = Query(2000, ge=1, le=10000), c: Container = ContainerDep
) -> list[EquityPoint]:
    return await c.sandbox.equity(account_id, limit)


@router.get(
    "/accounts/{account_id}/journal",
    response_model=list[JournalEntry],
    summary="What the agent did and learned, newest first",
)
async def journal(
    account_id: int = AccountId, limit: int = Query(100, ge=1, le=1000), c: Container = ContainerDep
) -> list[JournalEntry]:
    return await c.sandbox.journal(account_id, limit)
