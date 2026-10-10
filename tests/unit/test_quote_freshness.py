"""How fresh a trading price is, without I/O.

On Alpaca's IEX feed the last trade is IEX's alone: it can be an hour old while IEX's own book quotes the
stock second by second. A price is as fresh as its most recent *reliable* real-time observation — the
last trade, or a two-sided, uncrossed bid/ask no wider than the spread limit — and the quote-age limit
applies to that. Nothing here raises the limit or accepts a bid/ask that could not stand for the price.
"""

from datetime import UTC, datetime, timedelta

import pytest

from quantpulse.schemas.market import Quote
from quantpulse.services.trading_data import _live_quote, assess_quote, reliable_midpoint

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)  # 11:00 New York
LIMIT = 600.0
MAX_SPREAD = 30.0


def quote(trade_age=5886.0, bidask_age: float | None = 2.0, bid=99.99, ask=100.01, price=100.0) -> Quote:
    return Quote(symbol="SPY", price=price, bid=bid, ask=ask, timestamp=NOW - timedelta(seconds=trade_age),
                 quote_timestamp=None if bidask_age is None else NOW - timedelta(seconds=bidask_age),
                 feed="iex")  # fmt: skip


def live(q: Quote, spread: float | None = MAX_SPREAD):
    return _live_quote("SPY", q, "alpaca", NOW, max_spread_bps=spread)


def test_a_live_book_makes_a_live_price_even_when_iex_has_not_printed_for_an_hour():
    lq = live(quote())
    assert lq.price_basis == "quote" and lq.price == pytest.approx(100.0)
    assert lq.age_seconds == pytest.approx(2.0) and lq.trade_age_seconds == pytest.approx(5886.0)
    assert lq.price_source == "IEX bid/ask midpoint (alpaca)"
    assert lq.trade_price == 100.0 and lq.timestamp == NOW - timedelta(seconds=2)
    qq = assess_quote(lq, LIMIT)
    assert qq.usable_bid_ask and qq.spread_bps == pytest.approx(2.0) and qq.spread_source == "IEX only"


@pytest.mark.parametrize(
    ("why", "q"),
    [
        ("wider than the spread limit", quote(bid=99.8, ask=100.2)),  # 40bp
        ("one-sided", quote(bid=None)),
        ("crossed", quote(bid=100.02, ask=100.0)),
        ("no vendor timestamp", quote(bidask_age=None)),
        ("stamped in the future", quote(bidask_age=-60.0)),
        ("older than the trade", quote(trade_age=100.0, bidask_age=300.0)),
    ],
)
def test_a_bid_ask_that_cannot_stand_for_the_price_leaves_the_trade_as_the_price(why, q):
    lq = live(q)
    assert lq.price_basis == "trade", why
    assert lq.age_seconds == pytest.approx(NOW.timestamp() - q.timestamp.timestamp()), why
    assert lq.price_source == "last IEX trade (alpaca)"


def test_the_age_limit_still_applies_to_the_freshest_observation():
    stale = live(quote(trade_age=5886.0, bidask_age=900.0))  # the book has not moved for 15 minutes either
    assert stale.price_basis == "quote" and stale.age_seconds == pytest.approx(900.0)
    assert stale.age_seconds > LIMIT  # still stale: the limit is unchanged
    old_and_wide = live(quote(trade_age=5886.0, bid=99.0, ask=101.0))  # a 200bp book is not a price
    assert old_and_wide.price_basis == "trade" and old_and_wide.age_seconds == pytest.approx(5886.0)
    assert live(quote(), spread=None).price_basis == "trade"  # asked for prints only


def test_a_fresh_book_far_from_a_recent_print_is_not_believed():
    lq = live(
        quote(trade_age=60.0, bidask_age=1.0, bid=104.99, ask=105.01, price=100.0)
    )  # 5% off a 1-minute-old print
    assert lq.price_basis == "quote"
    qq = assess_quote(lq, LIMIT)
    assert not qq.usable_bid_ask and qq.spread_bps is None  # so no buy: the spread cannot be measured
    assert any("from the last trade $100.00" in p for p in qq.problems)
    # an hour-old print is no reference for where the market is now
    assert assess_quote(live(quote(bid=104.99, ask=105.01)), LIMIT).usable_bid_ask


def test_reliable_midpoint_rules():
    assert reliable_midpoint(quote(), NOW, MAX_SPREAD) == (pytest.approx(100.0), NOW - timedelta(seconds=2))
    assert reliable_midpoint(quote(bid=99.0, ask=101.0), NOW, MAX_SPREAD) is None
    assert reliable_midpoint(quote(bid=None), NOW, MAX_SPREAD) is None
    assert reliable_midpoint(quote(bidask_age=None), NOW, MAX_SPREAD) is None
