"""The prediction ledger: what was predicted, how it turned out, and the running scorecard."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import normalise_symbol
from quantpulse.core.jobs import JobPending
from quantpulse.schemas.jobs import JobOut
from quantpulse.schemas.predictions import (
    BackfillOut,
    BackfillStatus,
    LogResult,
    PredictionOut,
    ResolveResult,
    Scorecard,
)
from quantpulse.services.backfill import BackfillResult
from quantpulse.services.container import Container

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=list[PredictionOut], summary="Logged predictions, newest first")
async def list_predictions(
    symbol: str | None = Query(None),
    status: Literal["open", "resolved", "void"] | None = Query(None),
    source: Literal["forecast", "model"] | None = Query(None),
    origin: Literal["live", "backfill"] | None = Query(None, description="Live predictions or the replay"),
    limit: int = Query(200, ge=1, le=2000),
    c: Container = ContainerDep,
) -> list[PredictionOut]:
    return await c.predictions.list(
        symbol=normalise_symbol(symbol) if symbol else None,
        status=status,
        source=source,
        origin=origin,
        limit=limit,
    )


@router.get("/scorecard", response_model=Scorecard, summary="Brier score, hit rate, coverage and calibration")
async def scorecard(
    symbol: str | None = Query(None),
    origin: Literal["live", "backfill", "all"] = Query("all", description="Which record to score"),
    c: Container = ContainerDep,
) -> Scorecard:
    return await c.predictions.scorecard(normalise_symbol(symbol) if symbol else None, origin)


def _backfill_out(r: BackfillResult) -> BackfillOut:
    return BackfillOut(
        forecast_rows=r.forecast_rows,
        model_rows=r.model_rows,
        replaced=r.replaced,
        first_date=r.first_date,
        last_date=r.last_date,
        skipped=r.skipped,
    )


@router.post(
    "/backfill",
    response_model=BackfillOut,
    responses={202: {"model": JobOut, "description": "Started: poll the job"}},
    summary="Replay history point-in-time into the ledger (graded immediately, kept apart from live rows)",
)
async def backfill(
    sources: list[Literal["forecast", "model"]] = Query(["forecast", "model"]),
    replace: bool = Query(False, description="Delete earlier backfilled rows of these sources first"),
    wait: float = Query(0, ge=0, le=3600, description="Seconds to wait for the result"),
    c: Container = ContainerDep,
) -> BackfillOut | JSONResponse:
    job = c.backfill.start(tuple(dict.fromkeys(sources)), replace=replace)
    try:
        result: BackfillResult = await c.jobs.wait(job, wait)
    except JobPending:
        body = JobOut.of(job, c.clock.now())
        return JSONResponse(
            status_code=202, content=jsonable_encoder(body), headers={"Location": f"/api/v1/jobs/{job.id}"}
        )
    return _backfill_out(result)


@router.get("/backfill", response_model=BackfillStatus, summary="Backfill progress and ledger size")
async def backfill_status(c: Container = ContainerDep) -> BackfillStatus:
    job = c.backfill.job()
    counts = await c.predictions.counts()
    result = job.result if job is not None and isinstance(job.result, BackfillResult) else None
    return BackfillStatus(
        job=JobOut.of(job, c.clock.now()) if job else None,
        result=_backfill_out(result) if result else None,
        counts=list(counts),
    )


@router.post("/log", response_model=LogResult, summary="Log today's predictions now (after the close)")
async def log_now(c: Container = ContainerDep) -> LogResult:
    return await c.predictions.log_daily()


@router.post(
    "/resolve", response_model=ResolveResult, summary="Grade predictions whose target date has passed"
)
async def resolve_now(c: Container = ContainerDep) -> ResolveResult:
    return await c.predictions.resolve_due()
