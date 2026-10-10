"""The stock model lab, signal research and the market regime."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import parse_symbols
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.jobs import JobOut
from quantpulse.schemas.model import ModelReport, RegimeOut, ResearchReport, UniverseInfo
from quantpulse.services.container import Container

router = APIRouter(tags=["model"])


TRAINING: dict[int | str, dict[str, Any]] = {
    202: {"model": JobOut, "description": "Still computing: poll the job, then ask again"}
}


@router.get(
    "/model/report",
    response_model=CompositeEnvelope[ModelReport],
    responses=TRAINING,
    summary="Walk-forward stock models: out-of-sample skill, model comparison, backtest and live rankings",
)
async def model_report(
    horizon: int = Query(21, ge=5, le=63, description="Prediction horizon in trading days"),
    lookback_days: int = Query(1825, ge=900, le=3650, description="Calendar days of history"),
    top_k: int = Query(5, ge=1, le=20, description="Names held by the backtest portfolio"),
    symbols: str | None = Query(None, description="Comma-separated universe (default QP_MODEL_UNIVERSE)"),
    refresh: bool = Query(False, description="Recompute instead of reusing today's cached run"),
    wait: float | None = Query(
        None, ge=0, le=600, description="Seconds to wait for a run (default QP_MODEL_SYNC_WAIT_SECONDS)"
    ),
    c: Container = ContainerDep,
) -> CompositeEnvelope[ModelReport]:
    universe = parse_symbols(symbols, limit=100) if symbols else None
    return await c.model.report(horizon, lookback_days, top_k, universe, force=refresh, wait=wait)


@router.get("/model/universe", response_model=UniverseInfo, summary="Which stocks the model covers")
async def model_universe(c: Container = ContainerDep) -> UniverseInfo:
    return await c.model.universe_info()


@router.get(
    "/model/job",
    response_model=JobOut | None,
    summary="Progress of today's default model run (null when none has started)",
)
async def model_job(c: Container = ContainerDep) -> JobOut | None:
    job = c.model.model_job()
    return JobOut.of(job, c.clock.now()) if job is not None else None


@router.get(
    "/model/research",
    response_model=CompositeEnvelope[ResearchReport],
    responses=TRAINING,
    summary="Which signals ranked future returns, at which horizons",
)
async def research(
    horizon: int = Query(21, ge=1, le=63),
    lookback_days: int = Query(1825, ge=500, le=3650),
    symbols: str | None = Query(None),
    wait: float | None = Query(None, ge=0, le=600),
    c: Container = ContainerDep,
) -> CompositeEnvelope[ResearchReport]:
    universe = parse_symbols(symbols, limit=100) if symbols else None
    return await c.model.research(horizon, lookback_days, universe, wait=wait)


@router.get(
    "/market/regime",
    response_model=CompositeEnvelope[RegimeOut],
    summary="Trend, volatility, breadth and yield-curve regime with historical context",
)
async def regime(c: Container = ContainerDep) -> CompositeEnvelope[RegimeOut]:
    return await c.model.regime()
