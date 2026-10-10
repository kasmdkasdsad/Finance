from datetime import UTC, datetime, timedelta

import pytest
from scipy.stats import norm

from quantpulse.core.errors import DomainError
from quantpulse.domain.sports import (
    LEAGUE_PARAMS,
    EloModel,
    GameResult,
    american_to_prob,
    blended_expected_margin,
    devig_two_way,
    fraction_remaining,
    parse_clock,
    stern_win_probability,
)

T0 = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


def _game(i, home, away, hs, as_, season=2026, neutral=False):
    return GameResult(str(i), season, T0 + timedelta(days=i), home, away, hs, as_, neutral)


def test_elo_is_zero_sum_and_rewards_winner():
    m = EloModel(LEAGUE_PARAMS["nfl"])
    shift = m.update(_game(1, "A", "B", 30, 10))
    assert shift > 0
    assert m.rating("A") + m.rating("B") == pytest.approx(3000)
    assert m.records["A"].wins == 1 and m.records["B"].losses == 1


def test_upset_moves_ratings_more_than_expected_win():
    fav = EloModel(LEAGUE_PARAMS["nfl"], ratings={"A": 1650, "B": 1400})
    upset = EloModel(LEAGUE_PARAMS["nfl"], ratings={"A": 1650, "B": 1400})
    expected_shift = fav.update(_game(1, "A", "B", 24, 17))
    upset_shift = upset.update(_game(1, "A", "B", 17, 24))
    assert abs(upset_shift) > abs(expected_shift)


def test_expected_margin_and_probability_consistency():
    m = EloModel(LEAGUE_PARAMS["nfl"], ratings={"A": 1550, "B": 1500})
    assert m.expected_margin("A", "B") == pytest.approx((50 + 48) / 25)
    assert m.expected_margin("A", "B", neutral=True) == pytest.approx(2.0)
    assert m.expected_home("A", "B", neutral=True) == pytest.approx(1 / (1 + 10 ** (-50 / 400)))


def test_season_regression_and_dedupe():
    m = EloModel(LEAGUE_PARAMS["nfl"])
    games = [_game(1, "A", "B", 40, 0, season=2025), _game(1, "A", "B", 40, 0, season=2025)]
    m.fit(games)
    assert m.games_processed == 1
    before = m.rating("A")
    m.fit([_game(400, "C", "D", 20, 20, season=2026)])
    assert m.rating("A") == pytest.approx(before + (1500 - before) / 3)


def test_stern_model_limits():
    sigma = 13.45
    assert stern_win_probability(3.0, 0, 1.0, sigma) == pytest.approx(norm.cdf(3 / sigma))
    assert stern_win_probability(-7, 3, 0.0, sigma) == 1.0
    assert stern_win_probability(7, -1, 0.0, sigma) == 0.0
    assert stern_win_probability(7, 0, 0.0, sigma) == 0.5
    # Leading late is worth more than leading early
    assert stern_win_probability(0, 7, 0.1, sigma) > stern_win_probability(0, 7, 0.9, sigma) > 0.5
    with pytest.raises(DomainError):
        stern_win_probability(0, 0, 0.5, 0)


def test_fraction_remaining_and_clock():
    assert fraction_remaining("nfl", "pre", 0, None) == 1.0
    assert fraction_remaining("nfl", "post", 4, 0) == 0.0
    assert fraction_remaining("nfl", "in", 2, 0) == pytest.approx(0.5)  # halftime
    assert fraction_remaining("nfl", "in", 4, 120) == pytest.approx(120 / 3600)
    assert fraction_remaining("nfl", "in", 5, 600) == pytest.approx(600 / 3600)
    assert fraction_remaining("college-football", "in", 6, None) > 0
    assert parse_clock("12:34") == 754
    assert parse_clock("45.5") == 45.5
    assert parse_clock("bad") is None


def test_odds_helpers():
    assert american_to_prob(-150) == pytest.approx(0.6)
    assert american_to_prob(150) == pytest.approx(0.4)
    assert devig_two_way(-110, -110) == pytest.approx(0.5)
    assert devig_two_way(-200, 170) == pytest.approx((2 / 3) / (2 / 3 + 100 / 270))
    with pytest.raises(DomainError):
        american_to_prob(50)
    assert blended_expected_margin(4.0, -7.0, 0.6) == pytest.approx(0.6 * 7 + 0.4 * 4)
    assert blended_expected_margin(4.0, None, 0.6) == 4.0
