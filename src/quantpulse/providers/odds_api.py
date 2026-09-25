"""The Odds API (API key): consensus spreads and de-vigged moneyline probabilities across US bookmakers."""

from __future__ import annotations

import re
import statistics
from datetime import datetime

from quantpulse.core.errors import ProviderNotConfigured, ProviderParseError
from quantpulse.core.http import HttpClient
from quantpulse.domain.sports import devig_two_way
from quantpulse.providers.base import WireModel, parse_wire
from quantpulse.schemas.sports import League, MarketLine

NAME = "odds_api"
URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORT_KEYS: dict[str, str] = {"nfl": "americanfootball_nfl", "college-football": "americanfootball_ncaaf"}


class _Outcome(WireModel):
    name: str
    price: float
    point: float | None = None


class _Market(WireModel):
    key: str
    outcomes: list[_Outcome] = []


class _Bookmaker(WireModel):
    key: str
    markets: list[_Market] = []


class _Event(WireModel):
    id: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: list[_Bookmaker] = []


def normalise_team(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def consensus(event: _Event) -> MarketLine | None:
    spreads: list[float] = []
    probs: list[float] = []
    home_mls: list[float] = []
    away_mls: list[float] = []
    for book in event.bookmakers:
        for market in book.markets:
            outcomes = {o.name: o for o in market.outcomes}
            home, away = outcomes.get(event.home_team), outcomes.get(event.away_team)
            if home is None or away is None:
                continue
            if market.key == "spreads" and home.point is not None:
                spreads.append(home.point)
            elif market.key == "h2h":
                try:
                    probs.append(devig_two_way(home.price, away.price))
                except Exception:
                    continue
                home_mls.append(home.price)
                away_mls.append(away.price)
    if not spreads and not probs:
        return None
    return MarketLine(
        source="The Odds API consensus",
        spread_home=statistics.median(spreads) if spreads else None,
        home_moneyline=statistics.median(home_mls) if home_mls else None,
        away_moneyline=statistics.median(away_mls) if away_mls else None,
        home_implied_prob=statistics.median(probs) if probs else None,
        bookmakers=len(event.bookmakers),
    )


class OddsAPI:
    name = NAME

    def __init__(self, http: HttpClient, api_key: str | None, regions: str = "us") -> None:
        self._http = http
        self._key = api_key
        self._regions = regions
        self.requests_remaining: str | None = None
        self.requests_used: str | None = None

    def configured(self) -> bool:
        return bool(self._key)

    async def lines(self, league: League) -> dict[tuple[str, str], tuple[datetime, MarketLine]]:
        """Consensus lines keyed by (normalised home team, normalised away team)."""
        if not self._key:
            raise ProviderNotConfigured(NAME, "QP_ODDS_API_KEY not set")
        response = await self._http.request(
            NAME,
            "GET",
            URL.format(sport=SPORT_KEYS[league]),
            params={
                "apiKey": self._key,
                "regions": self._regions,
                "markets": "h2h,spreads",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )
        self.requests_remaining = response.headers.get("x-requests-remaining")
        self.requests_used = response.headers.get("x-requests-used")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderParseError(NAME, "invalid JSON") from exc
        out: dict[tuple[str, str], tuple[datetime, MarketLine]] = {}
        for item in payload if isinstance(payload, list) else []:
            event = parse_wire(NAME, _Event, item)
            line = consensus(event)
            if line is not None:
                out[(normalise_team(event.home_team), normalise_team(event.away_team))] = (
                    event.commence_time,
                    line,
                )
        return out
