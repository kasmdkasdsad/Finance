"""What kind of market-data problem stopped trading, without I/O. The limits themselves are never touched."""

import pytest

from quantpulse.brain.data_blockage import classify_decision, classify_market, classify_symbol
from quantpulse.db.models import BrainDecisionRow

LIMIT = 30.0


def diag(status, spread=5.0, source="last IEX trade (alpaca)", spread_source="IEX only"):
    return {"status": status, "spread_bps": spread, "price_source": source, "spread_source": spread_source}


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (diag("fresh"), []),
        (diag("live"), []),
        (diag("stale"), ["stale_trade"]),
        (diag("stale", source="IEX bid/ask midpoint (alpaca)"), ["stale_quote"]),
        (diag("no_trade_today"), ["stale_trade"]),
        (diag("fresh", spread=45.0), ["wide_spread"]),
        (diag("live", spread=None, spread_source="unavailable"), ["wide_spread"]),
        (diag("stale", spread=80.0), ["stale_trade", "wide_spread"]),
        (diag("missing", spread=None), ["missing_quote"]),
        (diag("provider_error", spread=None), ["provider_failure"]),
        (diag("subscription_unavailable", spread=None), ["provider_failure"]),
        (diag("delayed"), ["delayed_vendor"]),
        (diag("market_closed", spread=None), ["market_closed"]),
        (diag("holiday", spread=None), ["market_closed"]),
        (diag("invalid_timestamp"), ["invalid_timestamp"]),
    ],
)
def test_each_quote_problem_has_one_precise_name(d, expected):
    assert classify_symbol(d, LIMIT) == expected


def test_cycle_wide_causes_from_the_market_veto():
    assert classify_market({"veto": "only 32% of the universe has usable live quotes"}) == [
        "insufficient_coverage"
    ]
    assert classify_market({"veto": "system clock is +14s off Alpaca's: quote ages cannot be trusted"}) == [
        "clock_skew"
    ]
    assert classify_market({"veto": "broker unavailable (timeout)"}) == ["broker_unavailable"]
    assert classify_market({"veto": None}) == [] and classify_market(None) == []


def test_why_a_decision_was_stopped_by_data():
    halted = BrainDecisionRow(
        execution={"reason": "entries halted: TRADING BLOCKED — DATA QUALITY INSUFFICIENT"}
    )
    assert classify_decision(halted, None, ["insufficient_coverage", "stale_trade"], LIMIT) == [
        "insufficient_coverage",
        "stale_trade",
    ]
    too_old = BrainDecisionRow(
        risk={"checks": [{"name": "live_data", "passed": False, "detail": "quote is 900s old"}]}
    )
    assert classify_decision(too_old, diag("stale", source="IEX bid/ask midpoint (alpaca)"), [], LIMIT) == [
        "stale_quote"
    ]
    wide = BrainDecisionRow(
        risk={"checks": [{"name": "liquidity", "passed": False, "detail": "spread 45bp > 30bp"}]}
    )
    assert classify_decision(wide, diag("fresh", spread=45.0), [], LIMIT) == ["wide_spread"]
    nothing = BrainDecisionRow(
        risk={"checks": [{"name": "live_data", "passed": False, "detail": "no live quote for this symbol"}]}
    )
    assert classify_decision(nothing, None, [], LIMIT) == ["missing_quote"]
    fine = BrainDecisionRow(
        risk={"checks": [{"name": "position_limit", "passed": False, "detail": "too big"}]}
    )
    assert classify_decision(fine, diag("fresh"), [], LIMIT) == []
