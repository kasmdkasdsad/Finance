"""Sports analytics schemas: scoreboards, market lines, power ratings and win probabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import StrictModel

League = Literal["nfl", "college-football"]
GameState = Literal["pre", "in", "post"]


class TeamRef(StrictModel):
    id: str
    abbreviation: str
    name: str
    short_name: str | None = None
    logo: str | None = None
    rank: int | None = Field(default=None, description="Poll rank (college football)")
    record: str | None = None


class TeamScore(StrictModel):
    team: TeamRef
    score: int | None = None


class MarketLine(StrictModel):
    source: str
    spread_home: float | None = Field(default=None, description="Home spread (negative = home favoured).")
    over_under: float | None = None
    home_moneyline: float | None = None
    away_moneyline: float | None = None
    home_implied_prob: float | None = Field(default=None, description="De-vigged moneyline probability.")
    details: str | None = None
    bookmakers: int | None = None


class Game(StrictModel):
    event_id: str
    league: League
    season: int
    season_type: int = Field(description="1 = preseason, 2 = regular season, 3 = postseason")
    week: int | None = None
    start_time: AwareDatetime
    name: str
    state: GameState
    completed: bool
    status_detail: str
    period: int = 0
    display_clock: str | None = None
    home: TeamScore
    away: TeamScore
    neutral_site: bool = False
    venue: str | None = None
    broadcast: str | None = None
    market: MarketLine | None = None
    possession_team_id: str | None = None
    down_distance: str | None = None
    espn_home_win_prob: float | None = Field(default=None, ge=0, le=1)


class WinProbability(StrictModel):
    home: float = Field(ge=0, le=1)
    away: float = Field(ge=0, le=1)
    expected_margin_home: float
    pregame_home: float = Field(ge=0, le=1)
    elo_home: float = Field(ge=0, le=1)
    market_home: float | None = Field(default=None, ge=0, le=1)
    fraction_remaining: float = Field(ge=0, le=1)
    model: str


class GamePrediction(StrictModel):
    game: Game
    win_probability: WinProbability
    home_rating: float
    away_rating: float


class Scoreboard(StrictModel):
    league: League
    season: int | None
    week: int | None
    games: list[GamePrediction]


class PowerRating(StrictModel):
    rank: int
    team: TeamRef
    rating: float
    games: int
    wins: int
    losses: int
    ties: int
    points_for: int
    points_against: int
    avg_margin: float
    last_change: float


class EloParams(StrictModel):
    k: float = Field(gt=0)
    home_field: float = Field(ge=0)
    initial: float
    season_regression: float = Field(ge=0, le=1)
    points_per_elo: float = Field(gt=0)
    margin_sd: float = Field(gt=0)
    regulation_minutes: float = Field(gt=0)
    market_weight: float = Field(ge=0, le=1)
    non_fbs_initial: float | None = Field(
        default=None, description="College only: starting rating for FCS/lower-division opponents."
    )


class PowerRatings(StrictModel):
    league: League
    season: int
    as_of: AwareDatetime
    games_processed: int
    params: EloParams
    ratings: list[PowerRating]
