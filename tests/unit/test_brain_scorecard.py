"""The evaluation's return statistics without I/O: compounding, drawdown, and relative measures."""

import math

import pytest

from quantpulse.brain.scorecard import _relative, _returns_stats


def test_returns_compound_and_the_drawdown_is_peak_to_trough():
    stats = _returns_stats([0.10, -0.20, 0.05])
    assert stats["sessions"] == 3
    assert stats["total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    assert stats["max_drawdown"] == pytest.approx(-0.2)
    assert stats["up_days"] == pytest.approx(2 / 3, abs=1e-3)
    assert stats["volatility"] > 0 and stats["sharpe"] is not None and stats["sortino"] is not None


def test_too_little_data_reports_nothing_it_cannot_know():
    assert _returns_stats([]) == {"sessions": 0}
    one = _returns_stats([0.01])
    assert one["volatility"] is None and one["sharpe"] is None
    assert _relative([0.01], [0.0]) == {} and _relative([0.01, 0.02], [0.0]) == {}


def test_relative_measures_against_the_benchmark():
    bench = [0.01, -0.02, 0.015, 0.0, -0.005]
    twice = [2 * b for b in bench]
    rel = _relative(twice, bench)
    assert rel["beta"] == pytest.approx(2.0)
    assert rel["excess_return_annual"] == pytest.approx(sum(bench) / 5 * 252, abs=1e-5)
    same = _relative(bench, bench)
    assert same["tracking_error"] is None and same["information_ratio"] is None
    assert not any(isinstance(v, float) and math.isnan(v) for v in rel.values())
