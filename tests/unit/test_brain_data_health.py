"""What the Brain knows about its market data: every quote gets one precise status and the reasons, and
the cycle report names the cause (the IEX feed, a refused subscription, the clock, the calendar) instead
of just saying "stale". No limit is loosened: the coarse state decides executability as before."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from quantpulse.brain.data_health import (
    QuoteStatus,
    closed_reason,
    diagnose,
    feed_report,
)
from quantpulse.brain.types import EXECUTABLE_STATES, DataState
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_data import LiveQuote, assess_quote

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)  # Friday 11:00 New York
MAX_AGE, FRESH = 600.0, 60.0


def quote(age: float = 3.0, **kw) -> LiveQuote:
    base = dict(
        symbol="DELL",
        price=120.0,
        bid=119.98,
        ask=120.02,
        vwap=None,
        volume=None,
        day_high=None,
        day_low=None,
        day_open=None,
        timestamp=NOW - timedelta(seconds=age),
        provider="alpaca",
        age_seconds=max(age, 0.0),
        quote_time=NOW - timedelta(seconds=min(age, 2.0)),
        feed="iex",
        previous_close=119.0,
        as_of=NOW,
        history_close=119.0,
    )
    base.update(kw)
    return LiveQuote(**base)


def diag(q: LiveQuote | None, *, market_open=True, closed=None, missing=None, refused=None, synthetic=False):
    return diagnose(
        "DELL",
        quote=q,
        quality=assess_quote(q, MAX_AGE) if q is not None else None,
        missing_reason=missing,
        market_open=market_open,
        closed=closed,
        price_status=DataStatus.SYNTHETIC if synthetic else DataStatus.LIVE,
        fresh_seconds=FRESH,
        max_age_seconds=MAX_AGE,
        refused_feed=refused,
    )


def test_each_kind_of_data_gets_its_own_status():
    cases = {
        QuoteStatus.FRESH: diag(quote(3)),
        QuoteStatus.LIVE: diag(quote(200)),
        QuoteStatus.STALE: diag(quote(900)),
        QuoteStatus.NO_TRADE_TODAY: diag(quote(age=20 * 3600)),  # yesterday's print
        QuoteStatus.DELAYED: diag(quote(900, feed="delayed_sip")),
        QuoteStatus.MISSING: diag(None, missing="not returned by the batch quote source"),
        QuoteStatus.PROVIDER_ERROR: diag(None, missing="batch request failed: timeout"),
        QuoteStatus.SUBSCRIPTION: diag(None, missing="no live quote", refused="sip"),
        QuoteStatus.INVALID_TIMESTAMP: diag(quote(age=-120)),  # stamped two minutes ahead
        QuoteStatus.MARKET_CLOSED: diag(quote(3), market_open=False, closed="weekend"),
        QuoteStatus.HOLIDAY: diag(quote(3), market_open=False, closed="holiday"),
        QuoteStatus.SYNTHETIC: diag(quote(3), synthetic=True),
    }
    for status, d in cases.items():
        assert d.status is status, (status, d.status, d.reasons)
        assert d.reasons or status in (QuoteStatus.FRESH, QuoteStatus.LIVE), status
    executable = {s for s, d in cases.items() if d.state in EXECUTABLE_STATES}
    assert executable == {QuoteStatus.FRESH, QuoteStatus.LIVE}  # nothing else can ever be traded on
    assert cases[QuoteStatus.INVALID_TIMESTAMP].state is DataState.INVALID
    assert "in the future" in cases[QuoteStatus.INVALID_TIMESTAMP].reasons[0]
    assert "since the open" in cases[QuoteStatus.NO_TRADE_TODAY].reasons[0]
    assert "IEX only" in cases[QuoteStatus.FRESH].coverage


def test_the_coarse_state_is_never_looser_than_the_age_limit():
    for age in (0, 30, 59, 61, 599, 601, 3000):
        d = diag(quote(age))
        assert (d.state in EXECUTABLE_STATES) == (age <= MAX_AGE), age


def test_a_believed_bid_ask_older_than_the_print_counts():
    q = replace(quote(10), quote_time=NOW - timedelta(seconds=700))
    assert diag(q).status is QuoteStatus.FRESH  # a stale bid/ask is not believed at all (no spread from it)
    q = replace(quote(10), quote_time=NOW - timedelta(seconds=300))
    assert diag(q).status is QuoteStatus.LIVE  # a believed one ages the data to 300s


def test_the_calendar_says_why_the_market_is_closed():
    assert closed_reason(datetime(2026, 9, 26, 15, 0, tzinfo=UTC)) == "weekend"
    assert closed_reason(datetime(2026, 11, 26, 16, 0, tzinfo=UTC)) == "holiday"  # Thanksgiving
    assert closed_reason(datetime(2026, 9, 25, 12, 0, tzinfo=UTC)) == "pre-market"
    assert closed_reason(NOW) is None


def test_the_report_names_the_cause():
    iex_quiet = {f"S{i}": diag(quote(900)) for i in range(6)} | {"OK": diag(quote(3))}
    report = feed_report(
        iex_quiet,
        market_open=True,
        closed=None,
        feeds=[
            {
                "provider": "alpaca",
                "stock_feed": "iex",
                "stock_feed_error": None,
                "refused_feeds": {"sip": "2026-09-25T14:00:00+00:00"},
                "history_feed": "iex",
                "history_feed_refused": 403,
            }
        ],
        skew_seconds=0.3,
        max_age_seconds=MAX_AGE,
    )
    assert not report["healthy"] and report["usable"] == 1 and report["counts"]["stale"] == 6
    causes = " ".join(report["causes"])
    assert "Real-time SIP is not in this subscription" in causes
    assert "refused all-exchange (SIP) price history (HTTP 403)" in causes
    assert "6 of 7 IEX-priced symbols" in causes and "not a longer quote-age limit" in causes
    assert "clock" not in report["headline"]  # 0.3s is within tolerance

    skewed = feed_report(
        {"OK": diag(quote(3))},
        market_open=True,
        closed=None,
        feeds=[],
        skew_seconds=-45.0,
        max_age_seconds=MAX_AGE,
    )
    assert "45.0s behind Alpaca's" in skewed["headline"]

    closed = feed_report(
        {"A": diag(quote(3), market_open=False, closed="holiday")},
        market_open=False,
        closed="holiday",
        feeds=[],
        skew_seconds=None,
        max_age_seconds=MAX_AGE,
    )
    assert "market is closed (holiday)" in closed["headline"] and "not stale data" in closed["headline"]

    healthy = feed_report(
        {"A": diag(quote(3))},
        market_open=True,
        closed=None,
        feeds=[],
        skew_seconds=0.0,
        max_age_seconds=MAX_AGE,
    )
    assert healthy["healthy"] and healthy["headline"].startswith("Market data is healthy")
