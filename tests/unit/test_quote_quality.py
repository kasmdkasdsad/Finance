"""Quote validation before a spread is believed (the 1,000bp "spreads" seen on Alpaca's IEX feed).

The spread limit itself (``QP_TRADING_MAX_SPREAD_BPS``) is never loosened: a quote that cannot be
trusted yields *no* spread, and a buy without a measurable spread fails the liquidity check."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.providers.alpaca_trading import BrokerAccount
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_data import LiveQuote, assess_quote, spread_of
from quantpulse.services.trading_risk import OrderIntent, QuoteCheck, RiskBook, RiskLimits

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
MAX_AGE = 600.0


def quote(**kw) -> LiveQuote:
    base = dict(
        symbol="DELL",
        price=120.00,
        bid=119.98,
        ask=120.02,
        vwap=None,
        volume=None,
        day_high=None,
        day_low=None,
        day_open=None,
        timestamp=NOW - timedelta(seconds=3),
        provider="alpaca",
        age_seconds=3.0,
        quote_time=NOW - timedelta(seconds=2),
        feed="iex",
        previous_close=119.0,
        as_of=NOW,
        history_close=119.0,
    )
    base.update(kw)
    return LiveQuote(**base)


def book(q: LiveQuote, require_live_data: bool = True) -> RiskBook:
    qq = assess_quote(q, MAX_AGE)
    account = BrokerAccount(
        "…0001", "ACTIVE", "USD", 100_000, 100_000, 100_000, 200_000, 0, 0, 100_000,
        False, False, False, False, 0, 2.0,
    )  # fmt: skip
    check = QuoteCheck(
        q.price, DataStatus.LIVE, q.provider, q.age_seconds, qq.spread_bps, 5e8, qq.spread_source,
        qq.problems, qq.entry_blocks,
    )  # fmt: skip
    return RiskBook(
        RiskLimits(max_spread_bps=30.0, require_live_data=require_live_data),
        account, {}, [], True, False, {q.symbol: check},
    )  # fmt: skip


def buy(q: LiveQuote) -> OrderIntent:
    return OrderIntent(q.symbol, "buy", 10, q.price, "entry", "test")


def failed(b: RiskBook, o: OrderIntent) -> dict[str, str]:
    return {c.name: c.detail for c in b.evaluate(o).failures}


def test_a_tight_iex_quote_is_used_and_labelled():
    qq = assess_quote(quote(), MAX_AGE)
    assert qq.spread_bps == pytest.approx(spread_of(119.98, 120.02)) and qq.spread_source == "IEX only"
    assert qq.problems == () and qq.entry_blocks == ()
    assert failed(book(quote()), buy(quote())) == {}


def test_a_wide_iex_book_is_refused_at_the_unchanged_limit():
    wide = quote(bid=114.0, ask=126.0)  # ~1000bp: IEX's own thin book
    qq = assess_quote(wide, MAX_AGE)
    assert qq.spread_bps == pytest.approx(1000.0, rel=0.01) and qq.spread_source == "IEX only"
    assert "spread 1000bp (IEX only) > 30bp" in failed(book(wide), buy(wide))["liquidity"]


def test_the_consolidated_quote_measures_the_spread_when_available():
    q = quote(
        bid=114.0,
        ask=126.0,
        nbbo_bid=119.99,
        nbbo_ask=120.01,
        nbbo_time=NOW - timedelta(seconds=1),
        nbbo_feed="sip",
    )
    qq = assess_quote(q, MAX_AGE)
    assert qq.spread_source == "SIP" and qq.spread_bps == pytest.approx(spread_of(119.99, 120.01))
    assert failed(book(q), buy(q)) == {}


def test_a_delayed_consolidated_quote_counts_for_the_spread_but_not_forever():
    q = quote(
        bid=None,
        nbbo_bid=119.95,
        nbbo_ask=120.05,
        nbbo_time=NOW - timedelta(minutes=16),
        nbbo_feed="delayed_sip",
    )
    qq = assess_quote(q, MAX_AGE)
    assert qq.spread_source == "SIP (15-min delayed)" and qq.spread_bps == pytest.approx(8.33, abs=0.01)
    old = replace(q, nbbo_time=NOW - timedelta(minutes=40))
    qq = assess_quote(old, MAX_AGE)
    assert qq.spread_bps is None and any("consolidated" in p and "old" in p for p in qq.problems)


@pytest.mark.parametrize(
    ("kw", "why"),
    [
        (dict(bid=None), "one-sided (no bid)"),
        (dict(ask=0.0), "one-sided (no ask)"),
        (dict(bid=121.0, ask=120.0), "crossed"),
        (dict(quote_time=NOW - timedelta(hours=3)), "bid/ask is 10,800s old"),
        (dict(bid=129.9, ask=130.1), "midpoint is +8.3% from the last trade"),
    ],
)
def test_an_untrustworthy_bid_ask_is_not_a_spread(kw, why):
    q = quote(**kw)
    qq = assess_quote(q, MAX_AGE)
    assert qq.spread_bps is None and qq.spread_source == "unavailable" and not qq.usable_bid_ask
    assert any(why in p for p in qq.problems), qq.problems
    # an unmeasurable spread fails the buy (it used to pass silently)
    assert "spread cannot be measured" in failed(book(q), buy(q))["liquidity"]
    # ... unless live data is explicitly not required
    assert "liquidity" not in failed(book(q, require_live_data=False), buy(q))


def test_prices_inconsistent_with_history_block_entries_but_never_exits():
    jump = quote(price=60.0, bid=59.99, ask=60.01)  # -50% vs the last close: a split or a bad tick
    qq = assess_quote(jump, MAX_AGE)
    assert any("-50% from the last close" in b for b in qq.entry_blocks)
    b = book(jump)
    assert "quote_quality" in failed(b, buy(jump))
    b.held["DELL"] = 10.0
    sell = OrderIntent("DELL", "sell", 10, 60.0, "exit", "test", closes_position=True)
    assert b.evaluate(sell).approved

    mismatch = quote(previous_close=59.5, history_close=119.0, price=60.0, bid=59.99, ask=60.01)
    assert any("disagrees with" in b for b in assess_quote(mismatch, MAX_AGE).entry_blocks)


def test_quotes_without_a_bid_ask_timestamp_are_still_checked():
    q = quote(quote_time=None, feed=None, provider="yahoo")
    qq = assess_quote(q, MAX_AGE)
    assert qq.spread_bps is not None and qq.spread_source == "yahoo only"
