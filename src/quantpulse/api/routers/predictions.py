"""The prediction ledger: what was predicted, how it turned out, and the running scorecard."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import normalise_symbol
from quantpulse.schemas.predictions import LogResult, PredictionOut, ResolveResult, Scorecard
from quantpulse.services.container import Container

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=list[PredictionOut], summary="Logged predictions, newest first")
async def list_predictions(
    symbol: str | None = Query(None),
    status: Literal["open", "resolved", "void"] | None = Query(None),
    source: Literal["forecast", "model"] | None = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    c: Container = ContainerDep,
) -> list[PredictionOut]:
    return await c.predictions.list(
        symbol=normalise_symbol(symbol) if symbol else None, status=status, source=source, limit=limit
    )


@router.get("/scorecard", response_model=Scorecard, summary="Brier score, hit rate, coverage and calibration")
async def scorecard(symbol: str | None = Query(None), c: Container = ContainerDep) -> Scorecard:
    return await c.predictions.scorecard(normalise_symbol(symbol) if symbol else None)


@router.post("/log", response_model=LogResult, summary="Log today's predictions now (after the close)")
async def log_now(c: Container = ContainerDep) -> LogResult:
    return await c.predictions.log_daily()


@router.post(
    "/resolve", response_model=ResolveResult, summary="Grade predictions whose target date has passed"
)
async def resolve_now(c: Container = ContainerDep) -> ResolveResult:
    return await c.predictions.resolve_due()
