"""Health, system status and market-session endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from quantpulse import __version__
from quantpulse.api.deps import ContainerDep
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, next_open, session_at
from quantpulse.db import repositories as repo
from quantpulse.schemas.system import Health, IngestionEvent, MarketSessionOut, SystemStatus
from quantpulse.services.container import Container

public = APIRouter(tags=["system"])
router = APIRouter(tags=["system"])


@public.get("/health", response_model=Health, summary="Liveness probe (no auth)")
async def health() -> Health:
    return Health(status="ok", version=__version__)


@router.get(
    "/system/status", response_model=SystemStatus, summary="Provider health, cache, limiter and poller state"
)
async def system_status(c: Container = ContainerDep) -> SystemStatus:
    return SystemStatus.model_validate(await c.status())


@router.get("/system/ingestions", response_model=list[IngestionEvent], summary="Recent warehouse ingestions")
async def ingestions(
    limit: int = Query(50, ge=1, le=500), c: Container = ContainerDep
) -> list[IngestionEvent]:
    async with c.db.session() as s:
        rows = await repo.recent_ingestions(s, limit)
    return [
        IngestionEvent(
            dataset=r.dataset, key=r.key, provider=r.provider, rows=r.rows, created_at=r.created_at
        )
        for r in rows
    ]


@router.get("/market/session", response_model=MarketSessionOut, summary="NYSE session state")
async def market_session(c: Container = ContainerDep) -> MarketSessionOut:
    now = c.clock.now()
    return MarketSessionOut(
        now=now,
        session=session_at(now).value,
        is_trading_day=is_trading_day(now.astimezone(NEW_YORK).date()),
        next_open=next_open(now),
    )
