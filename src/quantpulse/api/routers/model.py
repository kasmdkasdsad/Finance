"""The stock model lab, signal research and the market regime."""

from __future__ import annotations

from fastapi import APIRouter, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import parse_symbols
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.model import ModelReport, RegimeOut, ResearchReport
from quantpulse.services.container import Container

router = APIRouter(tags=["model"])


@router.get(
    "/model/report",
    response_model=CompositeEnvelope[ModelReport],
    summary="Walk-forward stock model: out-of-sample skill, backtest, calibration and live rankings",
)
async def model_report(
    horizon: int = Query(21, ge=5, le=63, description="Prediction horizon in trading days"),
    lookback_days: int = Query(1825, ge=900, le=3650, description="Calendar days of history"),
    top_k: int = Query(5, ge=1, le=20, description="Names held by the backtest portfolio"),
    symbols: str | None = Query(None, description="Comma-separated universe (default QP_PICKS_UNIVERSE)"),
    refresh: bool = Query(False, description="Recompute instead of reusing today's cached run"),
    c: Container = ContainerDep,
) -> CompositeEnvelope[ModelReport]:
    universe = parse_symbols(symbols, limit=60) if symbols else None
    return await c.model.report(horizon, lookback_days, top_k, universe, force=refresh)


@router.get(
    "/model/research",
    response_model=CompositeEnvelope[ResearchReport],
    summary="Which signals ranked future returns, at which horizons",
)
async def research(
    horizon: int = Query(21, ge=1, le=63),
    lookback_days: int = Query(1825, ge=500, le=3650),
    symbols: str | None = Query(None),
    c: Container = ContainerDep,
) -> CompositeEnvelope[ResearchReport]:
    universe = parse_symbols(symbols, limit=60) if symbols else None
    return await c.model.research(horizon, lookback_days, universe)


@router.get(
    "/market/regime",
    response_model=CompositeEnvelope[RegimeOut],
    summary="Trend, volatility, breadth and yield-curve regime with historical context",
)
async def regime(c: Container = ContainerDep) -> CompositeEnvelope[RegimeOut]:
    return await c.model.regime()
