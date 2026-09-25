"""Probabilistic price forecasts (volatility model + options-implied view + calibration)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import symbol_path
from quantpulse.core.errors import DomainError
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.forecast import StockForecast
from quantpulse.services.container import Container

router = APIRouter(prefix="/forecast", tags=["forecast"])


def parse_horizons(raw: str) -> list[int]:
    try:
        values = sorted({int(p) for p in raw.split(",") if p.strip()})
    except ValueError as exc:
        raise DomainError("horizons must be comma-separated whole numbers of trading days") from exc
    if not values or values[0] < 1 or values[-1] > 252 or len(values) > 6:
        raise DomainError("give 1-6 horizons between 1 and 252 trading days")
    return values


@router.get(
    "/{symbol}",
    response_model=CompositeEnvelope[StockForecast],
    summary="Price ranges and probabilities for the next days/weeks/months",
)
async def forecast(
    symbol: str = Depends(symbol_path),
    horizons: str = Query("5,21,63", description="Trading-day horizons, comma-separated"),
    target: float | None = Query(None, gt=0, description="Also report P(price above this level)"),
    options: bool = Query(True, description="Add the options-implied view"),
    calibrate: bool = Query(False, description="Walk-forward test of the forecaster on this stock (slower)"),
    with_model: bool = Query(False, description="Tilt the drift by the stock model's calibrated view"),
    c: Container = ContainerDep,
) -> CompositeEnvelope[StockForecast]:
    return await c.forecast.forecast(
        symbol,
        parse_horizons(horizons),
        target=target,
        include_options=options,
        calibrate=calibrate,
        with_model=with_model,
    )
