"""Portfolio CRUD and risk-laboratory endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Path, Response, status

from quantpulse.api.deps import ContainerDep
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.portfolio import (
    AdHocRiskRequest,
    PortfolioIn,
    PortfolioOut,
    PortfolioRiskReport,
    RiskRequest,
)
from quantpulse.services.container import Container

router = APIRouter(tags=["portfolio"])
PortfolioId = Path(..., ge=1)


@router.get("/portfolios", response_model=list[PortfolioOut])
async def list_portfolios(c: Container = ContainerDep) -> list[PortfolioOut]:
    return await c.portfolio.list_all()


@router.post("/portfolios", response_model=PortfolioOut, status_code=status.HTTP_201_CREATED)
async def create_portfolio(data: PortfolioIn, c: Container = ContainerDep) -> PortfolioOut:
    return await c.portfolio.create(data)


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioOut)
async def get_portfolio(portfolio_id: int = PortfolioId, c: Container = ContainerDep) -> PortfolioOut:
    return await c.portfolio.get(portfolio_id)


@router.put("/portfolios/{portfolio_id}", response_model=PortfolioOut)
async def update_portfolio(
    data: PortfolioIn, portfolio_id: int = PortfolioId, c: Container = ContainerDep
) -> PortfolioOut:
    return await c.portfolio.update(portfolio_id, data)


@router.delete("/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(portfolio_id: int = PortfolioId, c: Container = ContainerDep) -> Response:
    await c.portfolio.delete(portfolio_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/portfolios/{portfolio_id}/risk", response_model=CompositeEnvelope[PortfolioRiskReport])
async def portfolio_risk(
    req: RiskRequest, portfolio_id: int = PortfolioId, c: Container = ContainerDep
) -> CompositeEnvelope[PortfolioRiskReport]:
    return await c.portfolio.risk_for(portfolio_id, req)


@router.post(
    "/portfolio/analyze",
    response_model=CompositeEnvelope[PortfolioRiskReport],
    summary="Risk for ad-hoc holdings",
)
async def analyze(
    req: AdHocRiskRequest, c: Container = ContainerDep
) -> CompositeEnvelope[PortfolioRiskReport]:
    return await c.portfolio.analyze(req.holdings, RiskRequest(**req.model_dump(exclude={"holdings"})))
