"""The risk engine: every rule that stands between a proposed trade and the Alpaca paper account."""

from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.domain import trading_performance as perf
from quantpulse.providers.alpaca_trading import BrokerAccount, BrokerOrder, BrokerPosition
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_risk import OrderIntent, QuoteCheck, RiskBook, RiskLimits, losing_positions

NOW = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)


def account(equity=100_000.0, cash=None, last_equity=None, long_mv=None, **kw) -> BrokerAccount:
    cash = equity if cash is None else cash
    long_mv = equity - cash if long_mv is None else long_mv
    return BrokerAccount(
        account_number="…1234",
        status="ACTIVE",
        currency="USD",
        equity=equity,
        last_equity=equity if last_equity is None else last_equity,
        cash=cash,
        buying_power=cash * 2,
        long_market_value=long_mv,
        short_market_value=0.0,
        portfolio_value=equity,
        trading_blocked=kw.get("blocked", False),
        account_blocked=False,
        trade_suspended_by_user=False,
        pattern_day_trader=False,
        daytrade_count=0,
        multiplier=2.0,
    )


def position(symbol, qty, price=100.0, avg=100.0) -> BrokerPosition:
    return BrokerPosition(
        symbol,
        qty,
        qty,
        "long",
        avg,
        price,
        qty * price,
        qty * avg,
        qty * (price - avg),
        price / avg - 1,
        0.0,
        price,
    )


def open_order(symbol, side, qty, limit=100.0) -> BrokerOrder:
    return BrokerOrder(
        "0f0f",
        f"qp-x-{symbol}",
        symbol,
        side,
        "limit",
        "day",
        "accepted",
        qty,
        None,
        0.0,
        None,
        limit,
        NOW,
        NOW,
        NOW,
        None,
        None,
        None,
        None,
    )


def quote(price=100.0, status=DataStatus.LIVE, age=5.0, spread=2.0, adv=5e8) -> QuoteCheck:
    return QuoteCheck(price, status, "alpaca", age, spread, adv)


def book(acct=None, positions=None, orders=(), open_=True, kill=False, quotes=None, limits=None) -> RiskBook:
    positions = positions or {}
    quotes = quotes if quotes is not None else {s: quote() for s in ("AAPL", "MSFT", "NVDA", "X", *positions)}
    return RiskBook(limits or RiskLimits(), acct or account(), positions, list(orders), open_, kill, quotes)


def buy(symbol="AAPL", qty=100, price=100.0, **kw) -> OrderIntent:
    return OrderIntent(symbol, "buy", qty, price, kw.pop("kind", "entry"), "test", **kw)


def sell(symbol="AAPL", qty=100, price=100.0, **kw) -> OrderIntent:
    return OrderIntent(symbol, "sell", qty, price, kw.pop("kind", "trim"), "test", **kw)


def failed(decision) -> set[str]:
    return {c.name for c in decision.failures}


def test_a_normal_buy_is_approved_with_every_check_recorded():
    d = book().evaluate(buy(qty=100))
    assert d.approved and d.summary == "approved"
    names = {c.name for c in d.checks}
    assert names >= {
        "kill_switch",
        "account",
        "market_open",
        "live_data",
        "no_working_order",
        "daily_loss",
        "order_size",
        "position_limit",
        "total_exposure",
        "max_positions",
        "buying_power",
        "liquidity",
    }


def test_position_too_large():
    b = book(positions={"AAPL": position("AAPL", 250)})  # 25% already
    assert failed(b.evaluate(buy(qty=100))) == {"position_limit"}  # 35% > 30%


def test_total_exposure_too_large():
    acct = account(cash=10_000.0)  # 90% invested
    b = book(acct, positions={"MSFT": position("MSFT", 900)})
    assert "total_exposure" in failed(b.evaluate(buy(qty=80)))  # 98% > 95%


def test_too_many_positions():
    held = {f"H{i}": position(f"H{i}", 10) for i in range(8)}
    b = book(positions=held)
    assert failed(b.evaluate(buy(qty=10))) == {"max_positions"}
    assert b.evaluate(buy("H0", qty=10)).approved  # adding to a holding is not a new position


def test_order_too_large_or_too_small():
    assert failed(book().evaluate(buy(qty=160))) == {"order_size"}  # $16,000 > $15,000
    assert failed(book().evaluate(buy(qty=0.5))) == {"order_size"}  # $50 < $100
    held = {"AAPL": position("AAPL", 250)}
    exit_all = sell(qty=250, closes_position=True, kind="stop_loss")
    assert book(positions=held).evaluate(exit_all).approved  # exits are never blocked by the cap


def test_daily_loss_limit_blocks_new_exposure_only():
    acct = account(equity=95_500.0, last_equity=100_000.0)  # −4.5% today
    b = book(acct, positions={"MSFT": position("MSFT", 50)})
    assert b.daily_loss_hit
    assert "daily_loss" in failed(b.evaluate(buy(qty=10)))
    assert b.evaluate(sell("MSFT", qty=50, closes_position=True, kind="exit")).approved


def test_kill_switch_blocks_everything_but_explicit_flattening():
    held = {"AAPL": position("AAPL", 100)}
    b = book(positions=held, kill=True)
    assert "kill_switch" in failed(b.evaluate(buy("MSFT", qty=10)))
    assert "kill_switch" in failed(b.evaluate(sell(qty=100, closes_position=True, kind="exit")))
    assert b.evaluate(sell(qty=100, closes_position=True, kind="flatten", intent="flatten")).approved


def test_market_closed():
    assert failed(book(open_=False).evaluate(buy(qty=10))) == {"market_open"}


def test_missing_stale_or_synthetic_data_is_refused():
    assert "live_data" in failed(book(quotes={}).evaluate(buy(qty=10)))
    synth = book(quotes={"AAPL": quote(status=DataStatus.SYNTHETIC)})
    assert "synthetic" in synth.evaluate(buy(qty=10)).summary
    stale = book(quotes={"AAPL": quote(status=DataStatus.STALE)})
    assert "live data is required" in stale.evaluate(buy(qty=10)).summary
    old = book(quotes={"AAPL": quote(age=3600)})
    assert "old" in old.evaluate(buy(qty=10)).summary
    relaxed = book(
        quotes={"AAPL": quote(status=DataStatus.STALE)}, limits=RiskLimits(require_live_data=False)
    )
    assert relaxed.evaluate(buy(qty=10)).approved
    never = book(
        quotes={"AAPL": quote(status=DataStatus.SYNTHETIC)}, limits=RiskLimits(require_live_data=False)
    )
    assert not never.evaluate(buy(qty=10)).approved  # synthetic prices are never traded


def test_insufficient_buying_power_respects_the_cash_buffer():
    b = book(
        account(cash=10_000.0, long_mv=90_000.0),
        positions={"MSFT": position("MSFT", 900)},
        limits=RiskLimits(max_total_exposure_pct=1.0),
    )
    d = b.evaluate(buy(qty=90))  # $9,000 vs $10,000 cash − $2,000 reserve
    assert failed(d) == {"buying_power"}


def test_shorting_is_disabled():
    b = book(positions={"AAPL": position("AAPL", 10)})
    assert failed(b.evaluate(sell(qty=11))) == {"no_short"}
    assert "no_short" in failed(book().evaluate(sell("NVDA", qty=1, price=200)))


def test_liquidity_and_spread():
    thin = book(quotes={"X": quote(adv=1e6)})
    assert "liquidity" in failed(thin.evaluate(buy("X", qty=10)))
    wide = book(quotes={"X": quote(spread=80.0)})
    assert "spread" in wide.evaluate(buy("X", qty=10)).summary
    penny = book(quotes={"X": quote(price=2.0)})
    assert "liquidity" in failed(penny.evaluate(buy("X", qty=100, price=2.0)))


def test_working_orders_and_blocked_accounts():
    b = book(orders=[open_order("AAPL", "buy", 10)])
    assert failed(b.evaluate(buy(qty=10))) == {"no_working_order"}
    assert "account" in failed(book(account(blocked=True)).evaluate(buy(qty=10)))


def test_approvals_accumulate_within_a_cycle():
    b = book()
    for sym in ("AAPL", "MSFT"):
        d = b.evaluate(buy(sym, qty=140))
        assert d.approved
        b.commit(d.intent)
    assert "no_working_order" in failed(b.evaluate(buy("AAPL", qty=10)))  # one order per symbol per cycle
    assert b.exposure == pytest.approx(28_000.0) and b.cash_left == pytest.approx(72_000.0)
    held = book(positions={"AAPL": position("AAPL", 100)})
    held.commit(held.evaluate(sell(qty=60)).intent)
    assert held.held["AAPL"] == pytest.approx(40.0)


def test_open_orders_count_towards_limits():
    b = book(orders=[open_order("MSFT", "buy", 250, limit=100.0)])
    assert b.exposure == pytest.approx(25_000.0) and "MSFT" in b.pending_new
    assert b.cash_left == pytest.approx(75_000.0)


def test_losing_positions_are_flagged():
    flagged = losing_positions({"A": position("A", 10, price=90.0), "B": position("B", 10, price=99.0)}, 0.08)
    assert set(flagged) == {"A"} and flagged["A"] == pytest.approx(-0.10)


# ----------------------------------------------------------------------------- performance
def test_round_trips_are_fifo():
    t0 = NOW
    fills = [
        perf.Fill("A", "buy", 10, 100.0, t0),
        perf.Fill("A", "buy", 10, 110.0, t0 + timedelta(hours=1)),
        perf.Fill("A", "sell", 15, 120.0, t0 + timedelta(days=1), "exit"),
        perf.Fill("B", "buy", 5, 50.0, t0),
        perf.Fill("B", "sell", 5, 45.0, t0 + timedelta(days=2), "stop_loss"),
    ]
    trips = perf.round_trips(fills)
    assert [(t.symbol, t.qty, t.entry_price) for t in trips] == [
        ("A", 10, 100.0),
        ("A", 5, 110.0),
        ("B", 5, 50.0),
    ]
    assert sum(t.pnl for t in trips) == pytest.approx(200 + 50 - 25)


def test_performance_uses_only_recorded_data():
    few = perf.summarize([perf.EquityPoint(NOW, NOW.date(), 100_000.0, 50_000.0)], [])
    assert few.sharpe is None and few.total_return is None and few.win_rate is None
    assert any("Sharpe" in n for n in few.notes) and any("round trips" in n for n in few.notes)
    points = []
    equity = 100_000.0
    for i, r in enumerate([0.01, -0.005, 0.004, 0.012, -0.02, 0.006, 0.003]):
        equity *= 1 + r
        day = NOW + timedelta(days=i)
        points.append(perf.EquityPoint(day, day.date(), equity, equity * 0.9))
        points.append(perf.EquityPoint(day - timedelta(hours=2), day.date(), equity * 0.999, equity * 0.9))
    fills = [
        perf.Fill("A", "buy", 10, 100.0, NOW),
        perf.Fill("A", "sell", 10, 110.0, NOW + timedelta(days=1), "exit"),
    ]
    p = perf.summarize(points, fills)
    assert p.days == 7 and p.sharpe is not None and p.sortino is not None
    assert (
        p.max_drawdown == pytest.approx(-0.02) and p.win_rate == 1.0 and p.realized_pl == pytest.approx(100.0)
    )
    assert p.avg_exposure == pytest.approx(0.9, abs=0.01) and p.by_exit == {"exit": pytest.approx(100.0)}
    assert p.daily[1]["return"] == pytest.approx(-0.005) and len(p.monthly) >= 1
