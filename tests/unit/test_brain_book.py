"""The Brain's paper book without I/O: simulated fills pay the spread, slippage and costs against the trade,
and performance comes only from the book's own marks and fills."""

from datetime import UTC, date, datetime, timedelta

import pytest

from quantpulse.brain.book import MIN_DAYS, _daily, fill_price, metrics
from quantpulse.db.models import (
    BrainBookEquityRow,
    BrainBookTradeRow,
)
from quantpulse.services.trading_data import QuoteQuality


def test_a_fill_pays_half_the_believed_spread_plus_slippage_against_the_trade():
    tight = QuoteQuality(spread_bps=4.0, spread_source="SIP")
    buy, how, half = fill_price("buy", 100.0, tight, slippage_bps=5.0, default_half_spread_bps=5.0)
    assert half == 2.0 and buy == pytest.approx(100.0 * (1 + 7 / 10_000)) and "SIP" in how
    sell, _, _ = fill_price("sell", 100.0, tight, slippage_bps=5.0, default_half_spread_bps=5.0)
    assert sell == pytest.approx(100.0 * (1 - 7 / 10_000))
    # no believable bid/ask: an assumed half-spread, and the fill says so
    unknown = QuoteQuality(spread_bps=None, spread_source="unavailable")
    price, how, half = fill_price("buy", 100.0, unknown, slippage_bps=5.0, default_half_spread_bps=8.0)
    assert half == 8.0 and price == pytest.approx(100.13) and "assumed" in how
    assert fill_price("buy", 100.0, None, slippage_bps=0.0, default_half_spread_bps=0.0)[0] == 100.0


def trade(side, notional, slippage_bps, cost, realized=None, held=None):
    return BrainBookTradeRow(executed_at=datetime(2026, 9, 25, tzinfo=UTC), symbol="AAA", side=side, action=side,
                             qty=notional / 100, proposed_price=100.0, fill_price=100.0, notional=notional,
                             slippage_bps=slippage_bps, cost=cost, price_source="t", realized_pnl=realized,
                             holding_days=held)  # fmt: skip


def test_performance_comes_from_marks_and_fills_only():
    start = date(2026, 9, 1)
    daily = [
        (start + timedelta(days=i), 100_000 * (1 + 0.001 * i), 500.0 * (1 + 0.0005 * i)) for i in range(30)
    ]
    trades = [trade("buy", 20_000, 7.0, 2.0), trade("sell", 10_000, 7.0, 1.0, realized=500.0, held=10.0),
              trade("sell", 5_000, 7.0, 0.5, realized=-100.0, held=4.0)]  # fmt: skip
    m = metrics(daily, trades, 100_000)
    assert m["sessions"] == 30 and not m["too_short_to_judge"]
    assert m["total_return"] == pytest.approx(0.029, abs=1e-6)
    assert m["benchmark_return"] == pytest.approx(0.0145, abs=1e-6)
    assert m["excess_return"] == pytest.approx(0.0145, abs=1e-5)
    assert m["max_drawdown"] == 0.0 and m["current_drawdown"] == 0.0 and m["sharpe"] > 0
    assert m["costs"] == 3.5 and m["slippage_bps"] == pytest.approx(7.0)
    assert m["slippage"] == pytest.approx(35_000 * 7 / 10_000)
    assert m["closed_trades"] == 2 and m["closed_hit_rate"] == 0.5 and m["realized_pnl"] == 400.0
    assert m["avg_holding_days"] == 7.0 and m["turnover"] == pytest.approx(35_000 / 101_435, rel=0.01)

    falling = [(start + timedelta(days=i), 100_000 * (1 - 0.01 * i), 500.0) for i in range(5)]
    short = metrics(falling, [], 100_000)
    assert short["too_short_to_judge"] and short["sessions"] < MIN_DAYS
    assert short["current_drawdown"] == pytest.approx(-0.04) and short["max_drawdown"] < 0
    assert metrics([], [], 100_000)["sessions"] == 0


def test_only_trading_sessions_count_on_the_equity_curve():
    def mark(day, equity):
        return BrainBookEquityRow(at=datetime(2026, 9, 25, 20, tzinfo=UTC), day=day, equity=equity, cash=0.0,
                                  invested=equity, positions=1, benchmark_price=500.0)  # fmt: skip

    marks = [mark(date(2026, 9, 25), 100.0), mark(date(2026, 9, 25), 101.0),  # Friday, two marks: the last counts
             mark(date(2026, 9, 26), 101.0), mark(date(2026, 9, 27), 101.0),  # weekend research cycles
             mark(date(2026, 9, 28), 102.0)]  # fmt: skip
    assert _daily(marks) == [(date(2026, 9, 25), 101.0, 500.0), (date(2026, 9, 28), 102.0, 500.0)]
