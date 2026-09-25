"""Treasury yield curve endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.schemas.common import Envelope
from quantpulse.schemas.options import RateAtTenor, YieldCurve
from quantpulse.services.container import Container

router = APIRouter(prefix="/rates", tags=["rates"])


@router.get("/curve", response_model=Envelope[YieldCurve], summary="Latest U.S. Treasury par yield curve")
async def curve(c: Container = ContainerDep) -> Envelope[YieldCurve]:
    r = await c.rates.curve()
    return Envelope[YieldCurve](data=r.value, meta=r.provenance)


@router.get("/at", response_model=Envelope[RateAtTenor], summary="Interpolated risk-free rate for a maturity")
async def rate_at(
    years: float = Query(..., gt=0, le=50), c: Container = ContainerDep
) -> Envelope[RateAtTenor]:
    value, resolved = await c.rates.rate_at(years)
    return Envelope[RateAtTenor](data=value, meta=resolved.provenance)
