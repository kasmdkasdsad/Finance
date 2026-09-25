from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quantpulse.domain import screener
from quantpulse.schemas.market import Bar, PriceHistory, Quote
from quantpulse.services.picks import closes_with_quote


def _walk(seed: int, n: int = 400, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(drift + 0.015 * rng.standard_normal(n)))


def test_rating_is_monotonic_and_bounded():
    ratings = [screener.rating_from_z(z) for z in np.linspace(-6, 6, 121)]
    assert ratings == sorted(ratings)
    assert ratings[0] == 1 and ratings[-1] == 10 and screener.rating_from_z(0.0) in (5, 6)


def test_compute_factors_needs_history_and_positive_prices():
    with pytest.raises(ValueError, match="at least 64"):
        screener.compute_factors(_walk(1, 63))
    bad = _walk(1, 100)
    bad[10] = 0.0
    with pytest.raises(ValueError):
        screener.compute_factors(bad)
    short = screener.compute_factors(_walk(1, 100))
    assert short.momentum_12_1 is None and short.risk_adjusted is None and short.momentum_3m is not None


def test_windowed_rsi_matches_full_history():
    closes = _walk(7, 3000)
    assert screener.rsi(closes[-screener.RSI_WINDOW :]) == pytest.approx(screener.rsi(closes), abs=1e-6)


def test_custom_weights_change_the_ranking_and_are_validated():
    universe = {
        "TREND": screener.compute_factors(_walk(3, drift=0.002)),
        "FLAT": screener.compute_factors(_walk(4)),
        "DOWN": screener.compute_factors(_walk(5, drift=-0.002)),
    }
    default = screener.screen(universe)
    assert default[0].symbol == "TREND" and default[-1].symbol == "DOWN"
    only_low_vol = dict.fromkeys(screener.WEIGHTS, 0.0) | {"low_volatility": 1.0}
    ranked = screener.screen(universe, only_low_vol)
    vols = {r.symbol: r.factors.volatility_3m for r in ranked}
    assert [r.symbol for r in ranked] == sorted(vols, key=lambda s: vols[s])
    for bad in ({"trend": 1.0}, dict.fromkeys(screener.WEIGHTS, 0.0), only_low_vol | {"trend": -0.1}):
        with pytest.raises(ValueError, match="weights"):
            screener.screen(universe, bad)


def _history(last_close_at: datetime) -> PriceHistory:
    bars = [
        Bar(timestamp=last_close_at - timedelta(days=i), open=10, high=20, low=9, close=10 + i, volume=1)
        for i in range(3, -1, -1)
    ]
    return PriceHistory(symbol="X", interval="1d", bars=bars)


def _quote(at: datetime, price: float = 99.0) -> Quote:
    return Quote(symbol="X", price=price, timestamp=at)


def test_live_quote_is_spliced_in_as_todays_close():
    thursday_close = datetime(2026, 9, 24, 20, 0, tzinfo=UTC)
    hist = _history(thursday_close)
    assert closes_with_quote(hist, _quote(thursday_close - timedelta(hours=1))) == [13, 12, 11, 10]
    friday_open = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
    assert closes_with_quote(hist, _quote(friday_open)) == [13, 12, 11, 10, 99.0]
    # A vendor that already publishes today's partial bar: refresh its close instead of duplicating the day.
    partial = _history(datetime(2026, 9, 25, 13, 30, tzinfo=UTC))
    assert closes_with_quote(partial, _quote(friday_open)) == [13, 12, 11, 99.0]
