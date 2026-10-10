"""Daily stock picks (1–10 rating) and the email digest."""

from __future__ import annotations

from fastapi import APIRouter, Query

from quantpulse.api.deps import ContainerDep
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.picks import DailyPicks, PicksEmailRequest, PicksEmailResult, PicksMethod
from quantpulse.services.container import Container

router = APIRouter(prefix="/picks", tags=["picks"])


@router.get(
    "/daily", response_model=CompositeEnvelope[DailyPicks], summary="Today's ranked picks with 1-10 ratings"
)
async def daily(
    top_n: int = Query(10, ge=1, le=50),
    refresh: bool = Query(False, description="Bypass cached prices"),
    method: PicksMethod = Query(
        "auto",
        description="factors = hand-set rule; model = walk-forward stock model; blend = both; auto = blend "
        "only when the model has shown out-of-sample skill",
    ),
    forecast: bool = Query(True, description="Add 21-day price ranges from the volatility model"),
    c: Container = ContainerDep,
) -> CompositeEnvelope[DailyPicks]:
    return await c.picks.daily(top_n, force_refresh=refresh, method=method, with_forecast=forecast)


@router.post("/email", response_model=PicksEmailResult, summary="Email today's picks (SMTP)")
async def email(req: PicksEmailRequest, c: Container = ContainerDep) -> PicksEmailResult:
    return await c.picks.email_digest(
        c.notifier,
        [str(r) for r in req.recipients] if req.recipients else None,
        req.top_n,
        req.allow_synthetic,
    )
