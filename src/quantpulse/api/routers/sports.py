"""NFL / college-football scoreboards, power ratings and win probabilities."""

from __future__ import annotations

from fastapi import APIRouter

from quantpulse.api.deps import ContainerDep
from quantpulse.schemas.common import CompositeEnvelope
from quantpulse.schemas.sports import League, PowerRatings, Scoreboard
from quantpulse.services.container import Container

router = APIRouter(prefix="/sports", tags=["sports"])


@router.get(
    "/{league}/scoreboard",
    response_model=CompositeEnvelope[Scoreboard],
    summary="Live scores + win probability",
)
async def scoreboard(league: League, c: Container = ContainerDep) -> CompositeEnvelope[Scoreboard]:
    return await c.sports.scoreboard(league)


@router.get("/{league}/ratings", response_model=CompositeEnvelope[PowerRatings], summary="Elo power ratings")
async def ratings(league: League, c: Container = ContainerDep) -> CompositeEnvelope[PowerRatings]:
    return await c.sports.ratings(league)
