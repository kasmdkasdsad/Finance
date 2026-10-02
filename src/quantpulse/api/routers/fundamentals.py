"""Fundamentals, consensus estimates and DCF valuation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from quantpulse.api.deps import ContainerDep
from quantpulse.api.params import symbol_path
from quantpulse.schemas.common import CompositeEnvelope, Envelope
from quantpulse.schemas.fundamentals import AnalystEstimates, CompanyFundamentals, DCFRequest, ValuationReport
from quantpulse.services.container import Container

router = APIRouter(tags=["fundamentals"])


@router.get(
    "/fundamentals/{symbol}",
    response_model=Envelope[CompanyFundamentals],
    summary="SEC XBRL annual statements",
)
async def fundamentals(
    symbol: str = Depends(symbol_path), c: Container = ContainerDep
) -> Envelope[CompanyFundamentals]:
    r = await c.fundamentals.fundamentals(symbol)
    return Envelope[CompanyFundamentals](data=r.value, meta=r.provenance)


@router.get(
    "/fundamentals/{symbol}/estimates",
    response_model=Envelope[AnalystEstimates],
    summary="Consensus estimates",
)
async def estimates(
    symbol: str = Depends(symbol_path), c: Container = ContainerDep
) -> Envelope[AnalystEstimates]:
    r = await c.fundamentals.estimates(symbol)
    return Envelope[AnalystEstimates](data=r.value, meta=r.provenance)


@router.post(
    "/valuation/{symbol}/dcf",
    response_model=CompositeEnvelope[ValuationReport],
    summary="Live DCF + Monte Carlo",
)
async def dcf(
    req: DCFRequest, symbol: str = Depends(symbol_path), c: Container = ContainerDep
) -> CompositeEnvelope[ValuationReport]:
    return await c.valuation.dcf(symbol, req)
