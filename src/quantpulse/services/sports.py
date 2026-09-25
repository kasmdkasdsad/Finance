"""Sports analytics hub: live scoreboards, Elo power ratings and win probabilities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.domain.sports import (
    LEAGUE_PARAMS,
    EloModel,
    GameResult,
    blended_expected_margin,
    fraction_remaining,
    infer_non_fbs,
    parse_clock,
    stern_win_probability,
)
from quantpulse.providers import synthetic
from quantpulse.providers.espn import ESPN
from quantpulse.providers.odds_api import OddsAPI, normalise_team
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus
from quantpulse.schemas.sports import (
    Game,
    GamePrediction,
    League,
    MarketLine,
    PowerRating,
    PowerRatings,
    Scoreboard,
    TeamRef,
    WinProbability,
)

REGULAR_WEEKS: dict[str, int] = {"nfl": 18, "college-football": 16}
POSTSEASON_WEEKS: dict[str, int] = {"nfl": 5, "college-football": 1}


def to_result(game: Game) -> GameResult | None:
    if not game.completed or game.home.score is None or game.away.score is None:
        return None
    return GameResult(
        event_id=game.event_id,
        season=game.season,
        start_time=game.start_time,
        home_id=game.home.team.id,
        away_id=game.away.team.id,
        home_score=game.home.score,
        away_score=game.away.score,
        neutral_site=game.neutral_site,
    )


class SportsService:
    def __init__(
        self, settings: Settings, gateway: DataGateway, db: Database, clock: Clock, espn: ESPN, odds: OddsAPI
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._espn = espn
        self._odds = odds

    # ------------------------------------------------------------------ data
    async def games(
        self, league: League, *, force_refresh: bool = False
    ) -> Resolved[tuple[list[Game], dict]]:
        async def fetch() -> tuple[list[Game], dict]:
            return await self._espn.scoreboard(league)

        async def persist(value: tuple[list[Game], dict], provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_games(s, value[0])
                await repo.record_ingestion(s, "scoreboard", league, provider, rows)

        def synth() -> tuple[list[Game], dict]:
            now = self._clock.now()
            season, week = synthetic.current_week(league, now)
            return synthetic.synthetic_scoreboard(league, now), {
                "season": season,
                "season_type": 2,
                "week": week,
            }

        def ttl(value: tuple[list[Game], dict]) -> float:
            live = any(g.state == "in" for g in value[0])
            return self._settings.ttl_scoreboard_live if live else self._settings.ttl_scoreboard_idle

        return await self._gw.resolve(
            f"scoreboard:{league}",
            [Source(self._espn.name, fetch)],
            synth,
            ttl,
            on_live=persist,
            force_refresh=force_refresh,
        )

    async def _fetch_season(
        self, league: League, season: int, season_type: int, through_week: int
    ) -> list[Game]:
        weeks: list[tuple[int, int]] = []
        if season_type >= 2:
            last_regular = through_week if season_type == 2 else REGULAR_WEEKS[league]
            weeks += [(2, w) for w in range(1, last_regular + 1)]
        if season_type == 3:
            weeks += [(3, w) for w in range(1, min(through_week, POSTSEASON_WEEKS[league]) + 1)]
        results = await asyncio.gather(
            *(self._espn.scoreboard(league, season=season, season_type=st, week=w) for st, w in weeks)
        )
        return [g for games, _ in results for g in games if g.completed]

    async def season_results(
        self, league: League, meta: dict, *, synthetic_board: bool
    ) -> Resolved[list[Game]]:
        season = int(meta.get("season") or self._clock.now().year)
        season_type = int(meta.get("season_type") or 2)
        week = int(meta.get("week") or 1)

        async def fetch() -> list[Game]:
            games = await self._fetch_season(league, season, season_type, week)
            if self._settings.sports_include_prior_season:
                games += await self._fetch_season(league, season - 1, 3, POSTSEASON_WEEKS[league])
            return games

        async def persist(games: list[Game], provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_games(s, games)
                await repo.record_ingestion(s, "season_results", f"{league}:{season}", provider, rows)

        async def archive() -> tuple[list[Game], datetime, str] | None:
            async with self._db.session() as s:
                rows = await repo.completed_games(s, league, [season - 1, season])
            if not rows:
                return None
            games = [
                Game(
                    event_id=r.event_id,
                    league=league,
                    season=r.season,
                    season_type=r.season_type,
                    week=r.week,
                    start_time=r.start_time,
                    name=f"{r.away_name} at {r.home_name}",
                    state="post",
                    completed=True,
                    status_detail="Final",
                    home={
                        "team": TeamRef(id=r.home_team_id, abbreviation=r.home_team_id, name=r.home_name),
                        "score": r.home_score,
                    },
                    away={
                        "team": TeamRef(id=r.away_team_id, abbreviation=r.away_team_id, name=r.away_name),
                        "score": r.away_score,
                    },
                    neutral_site=r.neutral_site,
                )
                for r in rows
            ]
            return games, max(r.updated_at for r in rows), "espn"

        sources = [] if synthetic_board else [Source(self._espn.name, fetch)]
        return await self._gw.resolve(
            f"season:{league}:{season}:{season_type}:{week}",
            sources,
            lambda: synthetic.synthetic_season_results(league, self._clock.now()),
            self._settings.ttl_season_results,
            archive=None if synthetic_board else archive,
            on_live=persist,
        )

    async def market_lines(self, league: League) -> Resolved[dict] | None:
        if not self._odds.configured() or not self._gw.live_enabled:
            return None
        return await self._gw.resolve(
            f"odds:{league}",
            [Source(self._odds.name, lambda: self._odds.lines(league))],
            dict,
            self._settings.ttl_odds,
        )

    # ------------------------------------------------------------------ models
    async def fbs_teams(self, season: int) -> set[str] | None:
        """FBS membership (college only); ``None`` when ESPN's group listing is unavailable."""
        resolved = await self._gw.resolve(
            f"fbs:{season}",
            [Source(self._espn.name, lambda: self._espn.fbs_team_ids(season))],
            set,
            86400.0,
        )
        return resolved.value if resolved.value else None

    async def _model(
        self, league: League, results: list[Game], season: int
    ) -> tuple[EloModel, dict[str, TeamRef]]:
        params = LEAGUE_PARAMS[league]
        model = EloModel(params)
        teams: dict[str, TeamRef] = {}
        game_results = []
        for g in results:
            teams[g.home.team.id] = g.home.team
            teams[g.away.team.id] = g.away.team
            r = to_result(g)
            if r is not None:
                game_results.append(r)
        if params.non_fbs_initial is not None and game_results:
            fbs = await self.fbs_teams(season)
            if fbs is not None:
                non_fbs = set(teams) - fbs
            else:
                seasons = {g.season for g in game_results}
                non_fbs = infer_non_fbs(game_results) if len(seasons) > 1 else set()
            model.team_initial = dict.fromkeys(non_fbs, params.non_fbs_initial)
        model.fit(game_results)
        return model, teams

    async def _context(self, league: League):
        games_r = await self.games(league)
        _games, meta = games_r.value
        synthetic_board = games_r.status is DataStatus.SYNTHETIC
        results_r, lines_r = await asyncio.gather(
            self.season_results(league, meta, synthetic_board=synthetic_board), self.market_lines(league)
        )
        results = results_r.value
        warnings: list[str] = []
        if not synthetic_board and results_r.status is DataStatus.SYNTHETIC:
            warnings.append("Season results unavailable — ratings start from the league mean.")
            results = []
        season = int(meta.get("season") or self._clock.now().year)
        if synthetic_board:
            model, teams = await self._model_synthetic(league, results)
        else:
            model, teams = await self._model(league, results, season)
        return games_r, results_r, lines_r, model, teams, meta, warnings

    async def _model_synthetic(
        self, league: League, results: list[Game]
    ) -> tuple[EloModel, dict[str, TeamRef]]:
        model = EloModel(LEAGUE_PARAMS[league])
        teams: dict[str, TeamRef] = {}
        for g in results:
            teams[g.home.team.id] = g.home.team
            teams[g.away.team.id] = g.away.team
        model.fit([r for g in results if (r := to_result(g)) is not None])
        return model, teams

    def _predict(
        self, league: League, game: Game, model: EloModel, line: MarketLine | None
    ) -> GamePrediction:
        params = LEAGUE_PARAMS[league]
        home_id, away_id = game.home.team.id, game.away.team.id
        elo_margin = model.expected_margin(home_id, away_id, game.neutral_site)
        spread = line.spread_home if line else None
        mu = blended_expected_margin(elo_margin, spread, params.market_weight)
        clock = parse_clock(game.display_clock) if game.display_clock else None
        tau = fraction_remaining(league, game.state, game.period, clock)
        lead = (game.home.score or 0) - (game.away.score or 0) if game.state != "pre" else 0
        if game.state == "post":
            p_home = 1.0 if lead > 0 else 0.0 if lead < 0 else 0.5
        else:
            p_home = stern_win_probability(mu, lead, tau, params.margin_sd)
        pregame = stern_win_probability(mu, 0, 1.0, params.margin_sd)
        return GamePrediction(
            game=game,
            win_probability=WinProbability(
                home=round(p_home, 4),
                away=round(1 - p_home, 4),
                expected_margin_home=round(mu, 2),
                pregame_home=round(pregame, 4),
                elo_home=round(model.expected_home(home_id, away_id, game.neutral_site), 4),
                market_home=line.home_implied_prob if line else None,
                fraction_remaining=round(tau, 4),
                model="Elo + market spread blend, Stern (1994) Brownian-motion in-game update",
            ),
            home_rating=round(model.rating(home_id), 1),
            away_rating=round(model.rating(away_id), 1),
        )

    async def scoreboard(self, league: League) -> CompositeEnvelope[Scoreboard]:
        games_r, results_r, lines_r, model, _teams, meta, _warnings = await self._context(league)
        games, _ = games_r.value
        odds = lines_r.value if lines_r else {}
        predictions = []
        for g in sorted(games, key=lambda g: (g.start_time, g.event_id)):
            line = g.market
            match = odds.get((normalise_team(g.home.team.name), normalise_team(g.away.team.name)))
            if match is not None and abs(match[0] - g.start_time) <= timedelta(hours=12):
                line = match[1]
                g = g.model_copy(update={"market": line})
            predictions.append(self._predict(league, g, model, line))
        sources = {"scoreboard": games_r.provenance, "season_results": results_r.provenance}
        if lines_r is not None:
            sources["odds"] = lines_r.provenance
        board = Scoreboard(league=league, season=meta.get("season"), week=meta.get("week"), games=predictions)
        return CompositeEnvelope(data=board, meta=CompositeMeta.from_sources(sources, self._clock.now()))

    async def ratings(self, league: League) -> CompositeEnvelope[PowerRatings]:
        games_r, results_r, _lines, model, teams, meta, _warnings = await self._context(league)
        for g in games_r.value[0]:
            teams.setdefault(g.home.team.id, g.home.team)
            teams.setdefault(g.away.team.id, g.away.team)
        season = int(meta.get("season") or self._clock.now().year)
        # College: rank FBS programmes only (lower-division opponents are rated but not listed).
        ranked = sorted((t for t in teams if t not in model.team_initial), key=lambda t: -model.rating(t))
        ratings: list[PowerRating] = []
        for i, team_id in enumerate(ranked, start=1):
            rec = model.records.get(team_id)
            games = rec.games if rec else 0
            ratings.append(
                PowerRating(
                    rank=i,
                    team=teams[team_id],
                    rating=round(model.rating(team_id), 1),
                    games=games,
                    wins=rec.wins if rec else 0,
                    losses=rec.losses if rec else 0,
                    ties=rec.ties if rec else 0,
                    points_for=rec.points_for if rec else 0,
                    points_against=rec.points_against if rec else 0,
                    avg_margin=round((rec.points_for - rec.points_against) / games, 2)
                    if rec and games
                    else 0.0,
                    last_change=round(rec.last_change, 2) if rec else 0.0,
                )
            )
        if results_r.status in (DataStatus.LIVE, DataStatus.CACHED) and ratings:
            async with self._db.session() as s:
                await repo.save_ratings(s, league, season, ratings)
        data = PowerRatings(
            league=league,
            season=season,
            as_of=self._clock.now(),
            games_processed=model.games_processed,
            params=LEAGUE_PARAMS[league],
            ratings=ratings,
        )
        sources = {"scoreboard": games_r.provenance, "season_results": results_r.provenance}
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, self._clock.now()))
