"""One-page stock intelligence report."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import symbol_path
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.stocks import StockReport
from quantpulse.services.container import Container

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get(
    "/{symbol}/report",
    response_model=CompositeEnvelope[StockReport],
    summary="Technicals, forecast ranges, model ranking, options view, DCF and the prediction track record",
)
async def report(
    symbol: str = Depends(symbol_path),
    target: float | None = Query(None, gt=0, description="Also report P(price above this level)"),
    model: bool = Query(True, description="Include the stock model's ranking"),
    valuation: bool = Query(True, description="Include the DCF summary"),
    options: bool = Query(True, description="Include the options-implied view"),
    c: Container = ContainerDep,
) -> CompositeEnvelope[StockReport]:
    return await c.stocks.report(
        symbol, target=target, include_model=model, include_valuation=valuation, include_options=options
    )
