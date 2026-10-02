"""Options chain, volatility surface and Black-Scholes-Merton pricing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import normalise_symbol, parse_dates, symbol_path
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.options import BSMRequest, BSMResult, OptionChain, VolSurface
from quantpulse.services.container import Container

router = APIRouter(prefix="/options", tags=["options"])


@router.get(
    "/{symbol}/chain", response_model=CompositeEnvelope[OptionChain], summary="Chain with model IV & Greeks"
)
async def chain(
    symbol: str = Depends(symbol_path),
    expirations: str | None = Query(None, description="Comma-separated ISO dates; default = nearest N"),
    max_expirations: int = Query(4, ge=1, le=20),
    c: Container = ContainerDep,
) -> CompositeEnvelope[OptionChain]:
    return await c.options.analyzed_chain(symbol, parse_dates(expirations), max_expirations)


@router.get(
    "/{symbol}/surface", response_model=CompositeEnvelope[VolSurface], summary="Implied-volatility surface"
)
async def surface(
    symbol: str = Depends(symbol_path),
    max_expirations: int = Query(8, ge=1, le=20),
    min_moneyness: float = Query(0.7, gt=0.1, lt=1.0),
    max_moneyness: float = Query(1.3, gt=1.0, lt=3.0),
    grid_points: int = Query(25, ge=5, le=101),
    c: Container = ContainerDep,
) -> CompositeEnvelope[VolSurface]:
    return await c.options.surface(symbol, max_expirations, (min_moneyness, max_moneyness), grid_points)


@router.post(
    "/price", response_model=CompositeEnvelope[BSMResult], summary="BSM price & Greeks with live inputs"
)
async def price(req: BSMRequest, c: Container = ContainerDep) -> CompositeEnvelope[BSMResult]:
    if req.symbol is not None:
        req = req.model_copy(update={"symbol": normalise_symbol(req.symbol)})
    return await c.options.price(req)
