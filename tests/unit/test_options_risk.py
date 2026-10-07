"""Option orders in the existing risk engine: the same book, the same kill switch and cash, plus checks that
recompute every number from the legs — never a naked short, never undefined risk, never 0DTE, never a
stale or model quote, never worse than the natural price, and the book's loss, delta and vega capped."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantpulse.providers.alpaca_trading import BrokerAccount, BrokerOrder, BrokerOrderLeg, BrokerPosition
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_risk import (
    OptionLegIntent,
    OptionLegQuote,
    OptionLimits,
    OptionOrderIntent,
    OrderIntent,
    QuoteCheck,
    RiskBook,
    RiskLimits,
)

NOW = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)  # 21 days before the 16 Oct 2026 expiration
C200, C210, C220 = "AAPL261016C00200000", "AAPL261016C00210000", "AAPL261016C00220000"
P190, P180 = "AAPL261016P00190000", "AAPL261016P00180000"
TODAY = "AAPL260925C00200000"  # expires today: 0DTE


def account(equity=100_000.0, cash=None, level=3, **kw) -> BrokerAccount:
    cash = equity if cash is None else cash
    return BrokerAccount("…1234", "ACTIVE", "USD", equity, kw.get("last_equity", equity), cash, cash * 2,
                         equity - cash, 0.0, equity, False, False, False, False, 0, 2.0,
                         options_trading_level=level, options_buying_power=kw.get("obp", cash))  # fmt: skip


def opt_position(symbol, qty, price, avg=None) -> BrokerPosition:
    avg = price if avg is None else avg
    return BrokerPosition(symbol, qty, qty, "long" if qty > 0 else "short", avg, price, qty * price * 100,
                          qty * avg * 100, qty * (price - avg) * 100, 0.0, 0.0, price, "us_option")  # fmt: skip


def stock_position(symbol, qty, price) -> BrokerPosition:
    return BrokerPosition(
        symbol, qty, qty, "long", price, price, qty * price, qty * price, 0.0, 0.0, 0.0, price
    )


def quote(bid, ask, *, age=5.0, feed="indicative", oi=5000.0, delta=0.5, vega=0.2) -> OptionLegQuote:
    return OptionLegQuote(bid, ask, age, feed, oi, 0.3, delta, vega)


QUOTES = {
    C200: quote(5.9, 6.1, delta=0.55, vega=0.25),
    C210: quote(2.4, 2.6, delta=0.35, vega=0.22),
    C220: quote(0.9, 1.0, delta=0.18, vega=0.15),
    P190: quote(2.9, 3.1, delta=-0.35, vega=0.22),
    P180: quote(1.1, 1.25, delta=-0.2, vega=0.16),
    TODAY: quote(1.0, 1.1),
}


def book(*, positions=(), orders=(), quotes=None, kill=False, market_open=True, limits=None, acct=None, **kw):
    return RiskBook(
        limits or RiskLimits(),
        acct or account(),
        {p.symbol: p for p in positions},
        list(orders),
        market_open,
        kill,
        {"AAPL": QuoteCheck(200.0, DataStatus.LIVE, "alpaca", 1.0, 2.0, 5e9)},
        option_quotes=QUOTES if quotes is None else quotes,
        now=NOW,
        **kw,
    )


def leg(symbol, side, intent, ratio=1):
    return OptionLegIntent(symbol, side, ratio, intent)


def spread(qty=1, limit=3.6, family="bull_call_spread", **kw) -> OptionOrderIntent:
    legs = (leg(C200, "buy", "buy_to_open"), leg(C210, "sell", "sell_to_open"))
    return OptionOrderIntent("AAPL", family, legs, qty, limit, "entry", "test", True, 200.0, **kw)


def failed(decision) -> set[str]:
    return {c.name for c in decision.failures}


def test_a_defined_risk_spread_passes_every_check():
    d = book().evaluate_option(spread())
    assert d.approved, d.summary
    names = {c.name for c in d.checks}
    assert {"no_naked_short", "defined_risk", "max_loss", "greeks", "limit_price", "expiration"} <= names
    detail = next(c.detail for c in d.checks if c.name == "max_loss")
    assert "$360" in detail  # the debit at the limit price is the maximum loss


def test_the_order_symbol_side_and_size():
    o = spread(qty=2)
    assert o.symbol == "AAPL:MLEG" and o.side == "buy" and o.notional == pytest.approx(720)
    credit = OptionOrderIntent("AAPL", "bull_put_spread", (leg(P190, "sell", "sell_to_open"),
                               leg(P180, "buy", "buy_to_open")), 1, -1.6, "entry", "t", True, 200.0)  # fmt: skip
    assert credit.side == "sell"
    single = OptionOrderIntent("AAPL", "long_call", (leg(C200, "buy", "buy_to_open"),), 1, 6.1, "entry", "t", True,
                               200.0)  # fmt: skip
    assert single.symbol == C200


def test_naked_shorts_are_refused_whatever_they_are_called():
    naked = OptionOrderIntent("AAPL", "bull_call_spread", (leg(C200, "buy", "buy_to_open"),
                              leg(C210, "sell", "sell_to_open", ratio=2)), 1, 1.2, "entry", "t", True, 200.0)  # fmt: skip
    d = book().evaluate_option(naked)
    assert "no_naked_short" in failed(d) and "defined_risk" in failed(d)
    lone = OptionOrderIntent("AAPL", "naked_call", (leg(C210, "sell", "sell_to_open"),), 1, 2.4, "entry", "t",
                             True, 200.0)  # fmt: skip
    assert {"no_naked_short", "structure_allowed"} <= failed(book().evaluate_option(lone))


def test_a_covered_call_needs_the_shares_and_they_cover_only_once():
    cc = OptionOrderIntent("AAPL", "covered_call", (leg(C210, "sell", "sell_to_open"),), 1, 2.4, "entry", "t",
                           True, 200.0)  # fmt: skip
    assert "no_naked_short" in failed(book().evaluate_option(cc))
    shares = stock_position("AAPL", 100, 200.0)
    assert book(positions=[shares]).evaluate_option(cc).approved
    already = opt_position(C220, -1, 1.0)  # the shares already cover a short call
    assert "no_naked_short" in failed(book(positions=[shares, already]).evaluate_option(cc))


def test_loss_limits_per_trade_underlying_and_book():
    assert "max_loss" not in failed(book().evaluate_option(spread(qty=4)))  # $1,440 ≤ $1,500
    assert "max_loss" in failed(book().evaluate_option(spread(qty=5)))  # $1,800 > $1,500
    tight = RiskLimits(options=OptionLimits(max_underlying_risk_pct=0.003))
    assert "underlying_risk" in failed(book(limits=tight).evaluate_option(spread()))
    held = [opt_position(f"SPY2610{d}C00600000", 1, 20.0) for d in (16, 23)]  # $4,000 at risk already
    small_book = RiskLimits(options=OptionLimits(max_total_risk_pct=0.04))
    d = book(positions=held, limits=small_book).evaluate_option(spread())
    assert "total_risk" in failed(d)
    explore = spread(exploration=True)
    assert "max_loss" not in failed(book().evaluate_option(explore))  # $360 ≤ the $1,000 exploration cap
    capped = RiskLimits(options=OptionLimits(exploration_max_loss=250.0))
    assert "max_loss" in failed(book(limits=capped).evaluate_option(explore))  # $360 > a $250 exploration cap
    assert "max_loss" in failed(book().evaluate_option(spread(qty=3, exploration=True)))  # never above $1,000
    assert book().evaluate_option(replace(explore, limit_price=2.4, legs=(leg(C210, "buy", "buy_to_open"),
                                  leg(C220, "sell", "sell_to_open")))).approved is False  # fmt: skip


def test_quotes_must_be_fresh_two_sided_and_never_modelled():
    stale = {**QUOTES, C210: quote(2.4, 2.6, age=600)}
    assert "option_quotes" in failed(book(quotes=stale).evaluate_option(spread()))
    model = {**QUOTES, C200: quote(5.9, 6.1, feed="model")}
    assert "option_quotes" in failed(book(quotes=model).evaluate_option(spread()))
    one_sided = {**QUOTES, C210: quote(None, 2.6)}
    assert "option_quotes" in failed(book(quotes=one_sided).evaluate_option(spread()))
    missing = {k: v for k, v in QUOTES.items() if k != C210}
    assert "option_quotes" in failed(book(quotes=missing).evaluate_option(spread()))


def test_liquidity_greeks_level_expiration_and_price():
    wide = {**QUOTES, C210: quote(1.5, 3.0)}
    assert "option_liquidity" in failed(book(quotes=wide).evaluate_option(spread(limit=4.5)))
    thin = {**QUOTES, C210: quote(2.4, 2.6, oi=10)}
    assert "option_liquidity" in failed(book(quotes=thin).evaluate_option(spread()))
    blind = {**QUOTES, C210: quote(2.4, 2.6, delta=None)}
    assert "greeks" in failed(book(quotes=blind).evaluate_option(spread()))  # fail closed
    assert "options_level" in failed(book(acct=account(level=2)).evaluate_option(spread()))
    assert "options_level" in failed(book(acct=account(level=None)).evaluate_option(spread()))
    zero = OptionOrderIntent("AAPL", "long_call", (leg(TODAY, "buy", "buy_to_open"),), 1, 1.1, "entry", "t", True,
                             200.0)  # fmt: skip
    no_min = RiskLimits(options=OptionLimits(min_dte=0))
    assert "expiration" in failed(book(limits=no_min).evaluate_option(zero))  # 0DTE: research only, always
    assert "limit_price" in failed(book().evaluate_option(spread(limit=3.9)))  # natural is 6.10 − 2.40 = 3.70


def test_the_same_switches_and_book_as_stocks():
    assert "kill_switch" in failed(book(kill=True).evaluate_option(spread()))
    assert "market_open" in failed(book(market_open=False).evaluate_option(spread()))
    off = RiskLimits(options=OptionLimits(enabled=False))
    assert "options_enabled" in failed(book(limits=off).evaluate_option(spread()))
    loss_day = book(acct=account(last_equity=110_000.0))
    assert "daily_loss" in failed(loss_day.evaluate_option(spread()))
    poor = book(acct=account(cash=2_000.0, obp=2_000.0))
    assert "buying_power" in failed(poor.evaluate_option(spread()))
    b = book()
    first = spread()
    assert b.evaluate_option(first).approved
    b.commit(first)
    assert "no_working_order" in failed(b.evaluate_option(first))  # never stacked in one cycle
    assert b.cash_left == pytest.approx(100_000 - 360)
    working = BrokerOrder("id", "qp-x", "AAPL:MLEG", "buy", "limit", "day", "new", 1.0, None, 0.0, None, 3.6,
                          None, None, None, None, None, None, None, "us_option", "mleg", None,
                          (BrokerOrderLeg(C200, "buy", 1, "buy_to_open", "new", 1, 0, None),
                           BrokerOrderLeg(C210, "sell", 1, "sell_to_open", "new", 1, 0, None)))  # fmt: skip
    w = book(orders=[working])
    assert "no_working_order" in failed(w.evaluate_option(spread()))
    assert w.cash_left == pytest.approx(100_000 - 360)


def test_closing_is_always_possible_but_never_more_than_held():
    held = [opt_position(C200, 2, 6.0), opt_position(C210, -2, 2.5)]
    close = OptionOrderIntent("AAPL", "bull_call_spread", (leg(C200, "sell", "sell_to_close"),
                              leg(C210, "buy", "buy_to_close")), 2, -3.3, "stop_loss", "t", False, 200.0,
                              intent="flatten")  # fmt: skip
    d = book(
        positions=held, kill=True, limits=RiskLimits(options=OptionLimits(enabled=False))
    ).evaluate_option(close)
    assert d.approved, d.summary  # the kill switch and the options switch never trap a position
    too_many = replace(close, qty=3)
    assert "closes_held" in failed(book(positions=held).evaluate_option(too_many))
    mixed = replace(close, legs=(leg(C200, "sell", "sell_to_close"), leg(C220, "buy", "buy_to_open")))
    assert "structure" in failed(book(positions=held).evaluate_option(mixed))


def test_stock_orders_never_carry_an_option_symbol():
    o = OrderIntent(C200, "buy", 1, 6.0, "entry", "t")
    d = book().evaluate(o)
    assert "asset_class" in failed(d)


@settings(max_examples=150, deadline=None)
@given(
    lo=st.integers(150, 240),
    width=st.integers(1, 20),
    p_lo=st.floats(0.3, 20),
    p_hi_frac=st.floats(0.05, 0.95),
    qty=st.integers(1, 4),
    credit=st.booleans(),
)
def test_property_an_approved_vertical_never_risks_more_than_the_cap(lo, width, p_lo, p_hi_frac, qty, credit):
    hi = lo + width
    low, high = f"AAPL261016P00{lo}000", f"AAPL261016P00{hi}000"
    p_hi = p_lo * (1 + p_hi_frac)  # the higher put is worth more
    assume(p_hi - p_lo < width)
    quotes = {
        low: quote(p_lo * 0.98, p_lo * 1.02, delta=-0.2),
        high: quote(p_hi * 0.98, p_hi * 1.02, delta=-0.4),
    }
    if credit:  # bull put spread: sell the higher put, buy the lower
        legs = (leg(high, "sell", "sell_to_open"), leg(low, "buy", "buy_to_open"))
        limit = -(p_hi * 0.98 - p_lo * 1.02)
        family = "bull_put_spread"
        true_loss = (width + limit) * 100 * qty
    else:  # bear put spread: buy the higher, sell the lower
        legs = (leg(high, "buy", "buy_to_open"), leg(low, "sell", "sell_to_open"))
        limit = p_hi * 1.02 - p_lo * 0.98
        family = "bear_put_spread"
        true_loss = limit * 100 * qty
    o = OptionOrderIntent("AAPL", family, legs, qty, round(limit, 2), "entry", "t", True, 200.0)
    d = book(quotes=quotes).evaluate_option(o)
    if d.approved:
        cap = min(OptionLimits().max_loss_per_trade, OptionLimits().max_loss_pct_per_trade * 100_000)
        assert true_loss <= cap + 1.0 + 2 * qty  # cents of rounding per unit
        assert not any(c.name == "no_naked_short" and not c.passed for c in d.checks)


def test_single_leg_limits_are_prices_and_a_sale_is_a_credit():
    held = [opt_position(C200, 2, 6.0)]
    sell = OptionOrderIntent("AAPL", "close", (leg(C200, "sell", "sell_to_close"),), 2, 5.9, "take_profit", "t", False,
                             200.0)  # fmt: skip
    assert sell.net_debit == -5.9 and sell.side == "sell"
    assert book(positions=held).evaluate_option(sell).approved  # selling at the bid is the natural price
    patient = replace(sell, limit_price=6.0)  # asking the mid: better than natural, allowed
    assert book(positions=held).evaluate_option(patient).approved
    giveaway = replace(sell, limit_price=5.8)  # below the bid: worse than crossing the spread
    assert "limit_price" in failed(book(positions=held).evaluate_option(giveaway))
    buy = OptionOrderIntent(
        "AAPL", "long_call", (leg(C200, "buy", "buy_to_open"),), 1, 6.5, "entry", "t", True, 200.0
    )
    assert "limit_price" in failed(book().evaluate_option(buy))  # paying above the ask: refused
