"""Football power ratings (Elo) and win-probability modelling.

Power ratings
    Margin-of-victory Elo in the style popularised by FiveThirtyEight's NFL model:
    ``shift = K · MOVmult · (actual − expected)`` with
    ``MOVmult = ln(|MOV| + 1) · 2.2 / (0.001 · EloDiff_winner + 2.2)`` (dampens autocorrelation for
    favourites). Ratings regress toward the mean between seasons.

Win probability
    Stern (1994) Brownian-motion model: the home team's final margin is normal with mean ``μ`` (pregame
    expected margin) and s.d. ``σ``. With lead ``L`` and fraction ``τ`` of regulation remaining,
    ``P(home win) = Φ((L + μτ) / (σ√τ))``. ``μ`` blends the market spread (when available) with the Elo
    spread. The model ignores possession and field position — it is a transparent baseline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from scipy.stats import norm

from quantpulse.core.errors import DomainError
from quantpulse.schemas.sports import EloParams, League

LEAGUE_PARAMS: dict[str, EloParams] = {
    "nfl": EloParams(
        k=20.0,
        home_field=48.0,
        initial=1500.0,
        season_regression=1.0 / 3.0,
        points_per_elo=25.0,
        margin_sd=13.45,
        regulation_minutes=60.0,
        market_weight=0.6,
    ),
    "college-football": EloParams(
        k=25.0,
        home_field=55.0,
        initial=1500.0,
        season_regression=0.4,
        points_per_elo=25.0,
        margin_sd=15.5,
        regulation_minutes=60.0,
        market_weight=0.6,
        non_fbs_initial=1200.0,
    ),
}

QUARTER_SECONDS = 900.0
NFL_OT_SECONDS = 600.0
COLLEGE_OT_EQUIVALENT_SECONDS = 300.0


@dataclass(slots=True)
class GameResult:
    event_id: str
    season: int
    start_time: datetime
    home_id: str
    away_id: str
    home_score: int
    away_score: int
    neutral_site: bool = False


@dataclass(slots=True)
class TeamRecord:
    games: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: int = 0
    points_against: int = 0
    last_change: float = 0.0


@dataclass
class EloModel:
    params: EloParams
    ratings: dict[str, float] = field(default_factory=dict)
    team_initial: dict[str, float] = field(default_factory=dict)
    records: dict[str, TeamRecord] = field(default_factory=dict)
    current_season: int | None = None
    games_processed: int = 0

    def initial(self, team_id: str) -> float:
        return self.team_initial.get(team_id, self.params.initial)

    def rating(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.initial(team_id))

    def _home_edge(self, neutral: bool) -> float:
        return 0.0 if neutral else self.params.home_field

    def expected_home(self, home_id: str, away_id: str, neutral: bool = False) -> float:
        diff = self.rating(home_id) + self._home_edge(neutral) - self.rating(away_id)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def expected_margin(self, home_id: str, away_id: str, neutral: bool = False) -> float:
        diff = self.rating(home_id) + self._home_edge(neutral) - self.rating(away_id)
        return diff / self.params.points_per_elo

    def start_season(self, season: int) -> None:
        """Regress every rating toward its starting level when a new season begins."""
        if self.current_season is not None and season > self.current_season:
            r = self.params.season_regression
            self.ratings = {t: v + r * (self.initial(t) - v) for t, v in self.ratings.items()}
            self.records = {}
        self.current_season = season

    def update(self, game: GameResult) -> float:
        if self.current_season is None or game.season != self.current_season:
            self.start_season(game.season)
        home, away = game.home_id, game.away_id
        expected = self.expected_home(home, away, game.neutral_site)
        margin = game.home_score - game.away_score
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        edge = self._home_edge(game.neutral_site)
        home_diff = self.rating(home) + edge - self.rating(away)
        winner_diff = home_diff if margin > 0 else -home_diff if margin < 0 else 0.0
        mov_mult = math.log(max(abs(margin), 1) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)
        shift = self.params.k * mov_mult * (actual - expected)
        self.ratings[home] = self.rating(home) + shift
        self.ratings[away] = self.rating(away) - shift
        for team, pf, pa, change in (
            (home, game.home_score, game.away_score, shift),
            (away, game.away_score, game.home_score, -shift),
        ):
            rec = self.records.setdefault(team, TeamRecord())
            rec.games += 1
            rec.points_for += pf
            rec.points_against += pa
            rec.last_change = change
            if pf > pa:
                rec.wins += 1
            elif pf < pa:
                rec.losses += 1
            else:
                rec.ties += 1
        self.games_processed += 1
        return shift

    def fit(self, games: Iterable[GameResult]) -> EloModel:
        seen: set[str] = set()
        for game in sorted(games, key=lambda g: (g.start_time, g.event_id)):
            if game.event_id in seen:
                continue
            seen.add(game.event_id)
            self.update(game)
        return self


def american_to_prob(moneyline: float) -> float:
    if moneyline == 0 or -100 < moneyline < 100:
        raise DomainError("American odds must be <= -100 or >= +100")
    if moneyline < 0:
        return -moneyline / (-moneyline + 100.0)
    return 100.0 / (moneyline + 100.0)


def devig_two_way(home_ml: float, away_ml: float) -> float:
    """Remove the bookmaker margin from a two-way market; returns the home probability."""
    ph, pa = american_to_prob(home_ml), american_to_prob(away_ml)
    return ph / (ph + pa)


def fraction_remaining(league: League | str, state: str, period: int, clock_seconds: float | None) -> float:
    """Share of regulation time left (overtime maps to a small positive remainder)."""
    if state == "pre":
        return 1.0
    if state == "post":
        return 0.0
    clock = max(0.0, clock_seconds or 0.0)
    if period <= 0:
        return 1.0
    if period <= 4:
        remaining = (4 - period) * QUARTER_SECONDS + min(clock, QUARTER_SECONDS)
        return max(0.0, min(1.0, remaining / (4 * QUARTER_SECONDS)))
    if league == "nfl":
        return max(0.0, min(clock, NFL_OT_SECONDS * 1.5)) / (4 * QUARTER_SECONDS)
    return COLLEGE_OT_EQUIVALENT_SECONDS / (4 * QUARTER_SECONDS)


def stern_win_probability(expected_margin: float, lead: float, tau: float, sigma: float) -> float:
    """``P(home wins)`` under the Brownian-motion model; ties at the horn count as 0.5."""
    if sigma <= 0:
        raise DomainError("sigma must be positive")
    if tau <= 0:
        return 1.0 if lead > 0 else 0.0 if lead < 0 else 0.5
    tau = min(tau, 1.0)
    return float(norm.cdf((lead + expected_margin * tau) / (sigma * math.sqrt(tau))))


def blended_expected_margin(
    elo_margin: float, market_spread_home: float | None, market_weight: float
) -> float:
    if market_spread_home is None:
        return elo_margin
    return market_weight * (-market_spread_home) + (1.0 - market_weight) * elo_margin


def parse_clock(display: str | None) -> float | None:
    """Parse ESPN display clocks such as ``'12:34'`` or ``'45.2'`` into seconds."""
    if not display:
        return None
    text = display.strip()
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return None


def rank_ratings(model: EloModel, team_ids: Sequence[str] | None = None) -> list[tuple[str, float]]:
    ids = team_ids if team_ids is not None else list(model.ratings)
    return sorted(((t, model.rating(t)) for t in ids), key=lambda x: -x[1])


def infer_non_fbs(games: Iterable[GameResult], min_games: int = 4) -> set[str]:
    """Fallback FBS detection: in an FBS-filtered feed, lower-division teams only appear in the handful of
    games they play against FBS opponents. Teams whose busiest season has fewer than ``min_games`` games
    are treated as non-FBS (only meaningful once at least one full season is in the data)."""
    counts: dict[tuple[str, int], int] = {}
    for g in games:
        for team in (g.home_id, g.away_id):
            counts[(team, g.season)] = counts.get((team, g.season), 0) + 1
    best: dict[str, int] = {}
    for (team, _season), n in counts.items():
        best[team] = max(best.get(team, 0), n)
    return {team for team, n in best.items() if n < min_games}
