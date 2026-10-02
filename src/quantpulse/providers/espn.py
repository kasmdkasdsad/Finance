"""ESPN public site API (keyless): NFL / college football scoreboards, live game state and lines.

ESPN blocks some datacenter IP ranges (HTTP 403 from its CDN); that surfaces as a provider error and the
sports hub falls back to the warehouse / synthetic slate with a visible badge.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import Field

from quantpulse.core.errors import ProviderNoData
from quantpulse.core.http import HttpClient
from quantpulse.domain.sports import devig_two_way, parse_clock
from quantpulse.providers.base import WireModel, finite_or_none, parse_wire
from quantpulse.schemas.sports import Game, League, MarketLine, TeamRef, TeamScore

NAME = "espn"
BASE = "https://site.api.espn.com/apis/site/v2/sports/football/{league}/scoreboard"
FBS_TEAMS_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{season}/types/2/groups/80/teams"
FBS_GROUP = "80"
_TEAM_REF_RE = re.compile(r"/teams/(\d+)")
_DETAILS_RE = re.compile(r"^\s*([A-Za-z0-9&.'\- ]+?)\s+([+-]?\d+(?:\.\d+)?)\s*$")


class _Team(WireModel):
    id: str
    abbreviation: str | None = None
    displayName: str | None = None
    shortDisplayName: str | None = None
    name: str | None = None
    logo: str | None = None


class _Record(WireModel):
    summary: str | None = None


class _Rank(WireModel):
    current: int | None = None


class _Competitor(WireModel):
    homeAway: str
    team: _Team
    score: str | None = None
    records: list[_Record] = []
    curatedRank: _Rank | None = None


class _StatusType(WireModel):
    state: str
    completed: bool = False
    detail: str | None = None
    shortDetail: str | None = None
    description: str | None = None


class _Status(WireModel):
    clock: float | None = None
    displayClock: str | None = None
    period: int | None = None
    type: _StatusType


class _OddsSide(WireModel):
    moneyLine: float | None = None
    favorite: bool | None = None


class _Odds(WireModel):
    details: str | None = None
    overUnder: float | None = None
    spread: float | None = None
    homeTeamOdds: _OddsSide | None = None
    awayTeamOdds: _OddsSide | None = None
    moneyline: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None


class _Probability(WireModel):
    homeWinPercentage: float | None = None


class _LastPlay(WireModel):
    probability: _Probability | None = None


class _Situation(WireModel):
    possession: str | None = None
    downDistanceText: str | None = None
    lastPlay: _LastPlay | None = None


class _Venue(WireModel):
    fullName: str | None = None


class _Broadcast(WireModel):
    names: list[str] = []


class _Competition(WireModel):
    id: str
    neutralSite: bool = False
    competitors: list[_Competitor]
    status: _Status | None = None
    odds: list[_Odds] | None = None
    situation: _Situation | None = None
    venue: _Venue | None = None
    broadcasts: list[_Broadcast] = []


class _Season(WireModel):
    year: int
    type: int


class _Week(WireModel):
    number: int | None = None


class _Event(WireModel):
    id: str
    date: datetime
    name: str
    season: _Season
    week: _Week | None = None
    competitions: list[_Competition]
    status: _Status | None = None


class _Ref(WireModel):
    ref: str = Field(alias="$ref")


class _RefPage(WireModel):
    items: list[_Ref] = []
    pageCount: int = 1


class _Scoreboard(WireModel):
    events: list[_Event] = []
    season: _Season | None = None
    week: _Week | None = None


def _american(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    else:
        text = str(value).strip().upper()
        if text in ("EVEN", "EV"):
            return 100.0
        try:
            v = float(text.replace("+", ""))
        except ValueError:
            return None
    return v if abs(v) >= 100 else None


def parse_market(odds: _Odds | None, home: _Team, away: _Team) -> MarketLine | None:
    if odds is None:
        return None
    spread_home: float | None = None
    details = (odds.details or "").strip()
    if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        spread_home = 0.0
    else:
        match = _DETAILS_RE.match(details)
        if match:
            fav, line = match.group(1).strip().upper(), abs(float(match.group(2)))
            if fav == (home.abbreviation or "").upper():
                spread_home = -line
            elif fav == (away.abbreviation or "").upper():
                spread_home = line
    if spread_home is None and odds.spread is not None:
        home_fav = odds.homeTeamOdds.favorite if odds.homeTeamOdds else None
        away_fav = odds.awayTeamOdds.favorite if odds.awayTeamOdds else None
        if home_fav is True or away_fav is False:
            spread_home = -abs(odds.spread)
        elif away_fav is True or home_fav is False:
            spread_home = abs(odds.spread)

    home_ml = _american(odds.homeTeamOdds.moneyLine) if odds.homeTeamOdds else None
    away_ml = _american(odds.awayTeamOdds.moneyLine) if odds.awayTeamOdds else None
    if (home_ml is None or away_ml is None) and isinstance(odds.moneyline, dict):

        def _nested(side: str) -> float | None:
            node = odds.moneyline.get(side) if odds.moneyline else None
            for key in ("close", "current", "open"):
                if isinstance(node, dict) and isinstance(node.get(key), dict):
                    val = _american(node[key].get("odds"))
                    if val is not None:
                        return val
            return None

        home_ml = home_ml or _nested("home")
        away_ml = away_ml or _nested("away")

    implied = devig_two_way(home_ml, away_ml) if home_ml is not None and away_ml is not None else None
    if spread_home is None and odds.overUnder is None and implied is None:
        return None
    source = str((odds.provider or {}).get("name") or "ESPN")
    return MarketLine(
        source=source,
        spread_home=spread_home,
        over_under=finite_or_none(odds.overUnder),
        home_moneyline=home_ml,
        away_moneyline=away_ml,
        home_implied_prob=implied,
        details=details or None,
    )


def parse_scoreboard(league: League, payload: Any) -> list[Game]:
    board = parse_wire(NAME, _Scoreboard, payload)
    games: list[Game] = []
    for ev in board.events:
        if not ev.competitions:
            continue
        comp = ev.competitions[0]
        home = next((c for c in comp.competitors if c.homeAway == "home"), None)
        away = next((c for c in comp.competitors if c.homeAway == "away"), None)
        if home is None or away is None:
            continue
        status = comp.status or ev.status
        if status is None:
            continue
        state = status.type.state if status.type.state in ("pre", "in", "post") else "pre"

        def team(c: _Competitor) -> TeamRef:
            rank = c.curatedRank.current if c.curatedRank and c.curatedRank.current else None
            return TeamRef(
                id=c.team.id,
                abbreviation=c.team.abbreviation or c.team.id,
                name=c.team.displayName or c.team.name or c.team.id,
                short_name=c.team.shortDisplayName,
                logo=c.team.logo,
                rank=rank if rank and rank <= 25 else None,
                record=c.records[0].summary if c.records else None,
            )

        def score(c: _Competitor, state: str = state) -> int | None:
            if state == "pre" or c.score in (None, ""):
                return None
            try:
                return int(float(c.score))
            except ValueError:
                return None

        situation = comp.situation
        espn_prob = None
        if situation and situation.lastPlay and situation.lastPlay.probability:
            p = finite_or_none(situation.lastPlay.probability.homeWinPercentage)
            espn_prob = p if p is not None and 0 <= p <= 1 else None
        clock_seconds = status.clock if status.clock is not None else parse_clock(status.displayClock)
        games.append(
            Game(
                event_id=ev.id,
                league=league,
                season=ev.season.year,
                season_type=ev.season.type,
                week=ev.week.number if ev.week else None,
                start_time=ev.date,
                name=ev.name,
                state=state,
                completed=status.type.completed,
                status_detail=status.type.shortDetail
                or status.type.detail
                or status.type.description
                or state,
                period=status.period or 0,
                display_clock=status.displayClock if clock_seconds is not None else None,
                home=TeamScore(team=team(home), score=score(home)),
                away=TeamScore(team=team(away), score=score(away)),
                neutral_site=comp.neutralSite,
                venue=comp.venue.fullName if comp.venue else None,
                broadcast=", ".join(n for b in comp.broadcasts for n in b.names) or None,
                market=parse_market(comp.odds[0] if comp.odds else None, home.team, away.team),
                possession_team_id=situation.possession if situation else None,
                down_distance=situation.downDistanceText if situation else None,
                espn_home_win_prob=espn_prob,
            )
        )
    return games


class ESPN:
    name = NAME

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    def configured(self) -> bool:
        return True

    async def scoreboard(
        self,
        league: League,
        *,
        season: int | None = None,
        season_type: int | None = None,
        week: int | None = None,
        dates: str | None = None,
    ) -> tuple[list[Game], dict[str, int | None]]:
        params: dict[str, object] = {"limit": 1000}
        if league == "college-football":
            params["groups"] = FBS_GROUP
        if dates:
            params["dates"] = dates
        elif season is not None:
            params["dates"] = season
        if season_type is not None:
            params["seasontype"] = season_type
        if week is not None:
            params["week"] = week
        payload = await self._http.get_json(NAME, BASE.format(league=league), params=params)
        games = parse_scoreboard(league, payload)
        board = parse_wire(NAME, _Scoreboard, payload)
        meta = {
            "season": board.season.year if board.season else (games[0].season if games else None),
            "season_type": board.season.type if board.season else (games[0].season_type if games else None),
            "week": board.week.number if board.week else (games[0].week if games else None),
        }
        return games, meta

    async def fbs_team_ids(self, season: int) -> set[str]:
        """ESPN team ids in the FBS group (80) for ``season`` (core API)."""
        payload = await self._http.get_json(NAME, FBS_TEAMS_URL.format(season=season), params={"limit": 400})
        page = parse_wire(NAME, _RefPage, payload)
        ids = {m.group(1) for item in page.items if (m := _TEAM_REF_RE.search(item.ref))}
        if len(ids) < 100:  # FBS has well over 100 programmes; anything less is a partial/odd payload
            raise ProviderNoData(NAME, f"FBS team list looks incomplete ({len(ids)} teams)")
        return ids
