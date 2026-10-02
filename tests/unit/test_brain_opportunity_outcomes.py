"""Rejected (and taken) ideas without I/O: why an idea was not taken, and how its outcome is judged."""

import pytest

from quantpulse.brain.decisions import Proposal
from quantpulse.brain.opportunities import Opportunity
from quantpulse.brain.opportunity_outcomes import _group, classify, verdict
from quantpulse.brain.types import Action
from quantpulse.db.models import BrainOpportunityOutcomeRow


def idea(direction=1, status="detected", stages=None) -> Opportunity:
    o = Opportunity("breakout", "AAA", ["AAA"], direction, 0.7, "AAA broke out")
    o.status = status
    o.stages = stages or [{"stage": "detection", "result": "AAA broke out"}]
    return o


def proposal(action, reasons=("r",), status="no_trade", risk_approved=None, qty=None, blocked=()) -> Proposal:
    return Proposal(subject="AAA", action=action, confidence=0.5, reasons=list(reasons), quantity=qty,
                    status=status, risk_approved=risk_approved, blocked_by=list(blocked), est_price=10.0)  # fmt: skip


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (proposal(Action.BUY, status="filled", risk_approved=True, qty=10), (True, "taken")),
        (proposal(Action.BUY, status="risk_rejected", risk_approved=False, qty=10), (False, "risk_engine")),
        (proposal(Action.BUY, status="halted", risk_approved=True, qty=10), (False, "entry_halt")),
        (proposal(Action.NO_ACTION, ["I do not know: one source only"]), (False, "unknown_consensus")),
        (proposal(Action.NO_ACTION, ["consensus bearish"]), (False, "consensus_against")),
        (proposal(Action.WATCH, ["bullish but confidence 0.31 below 0.45"]), (False, "low_confidence")),
        (proposal(Action.WATCH, ["bullish but earnings in 2 day(s): no new risk"]), (False, "earnings")),
        (
            proposal(Action.WATCH, ["bullish, but a poor portfolio fit: nearly the same bet"]),
            (False, "portfolio_fit"),
        ),
        (proposal(Action.WATCH, ["bullish, ranked 3: beyond the 2 new position(s)"]), (False, "slot_limit")),
        (proposal(Action.WATCH, ["bullish but devil's advocate: crowded"]), (False, "challenged")),
        (proposal(Action.WATCH, ["bullish but risk-off market: no new positions"]), (False, "risk_off")),
    ],
)
def test_why_an_idea_was_not_taken(p, expected):
    taken, reason, detail = classify(idea(), p, market_open=True, held=False)
    assert (taken, reason) == expected and detail


def test_ideas_stopped_before_a_decision_and_the_special_cases():
    assert classify(idea(status="not_analysed"), None, True, False)[1] == "focus_budget"
    assert classify(idea(status="rejected_data"), None, True, False)[1] == "data_quality"
    assert classify(idea(status="no_view"), None, True, False)[1] == "no_view"
    assert classify(idea(), proposal(Action.WATCH, ["bullish but x"]), False, False)[1] == "market_closed"
    assert classify(idea(), proposal(Action.HOLD, ["hold"]), True, True)[1] == "already_held"
    assert classify(idea(direction=-1), proposal(Action.NO_ACTION, ["consensus bearish"]), True, False)[
        1
    ] == ("nothing_to_sell")
    sold = proposal(Action.REDUCE, status="filled", risk_approved=True, qty=5)
    assert classify(idea(direction=-1), sold, True, True)[:2] == (True, "taken")


def test_verdicts_keep_noise_apart_from_right_and_wrong():
    assert verdict(False, "low_confidence", 0.04, 1.2) == "missed"
    assert verdict(False, "low_confidence", -0.04, -1.2) == "avoided"
    assert verdict(False, "low_confidence", 0.01, 0.2) == "noise"  # within half a standard deviation
    assert verdict(True, "taken", -0.03, -0.9) == "failed" and verdict(True, "taken", 0.03, 0.9) == "worked"
    assert verdict(False, "nothing_to_sell", 0.02, 0.8) == "signal_right"
    assert verdict(False, "low_confidence", 0.04, None) == "missed"  # no volatility: judged by sign only


def rows(verdicts):
    return [
        BrainOpportunityOutcomeRow(verdict=v, favourable=0.01 if v == "missed" else -0.01) for v in verdicts
    ]


def test_a_rejection_reason_is_unproven_until_the_sample_is_there():
    small = _group(rows(["avoided"] * 8 + ["missed"] * 2), 30, rejection=True)
    assert small["status"] == "unproven" and small["needs"] == 20 and small["decisive"] == 10
    saves = _group(rows(["avoided"] * 45 + ["missed"] * 15 + ["noise"] * 30), 30, rejection=True)
    assert saves["status"] == "right more often than not" and saves["avoided_share"] == 0.75
    assert saves["decisive"] == 60 and saves["ci95"][0] > 0.5  # noise is not counted either way
    costs = _group(rows(["avoided"] * 12 + ["missed"] * 40), 30, rejection=True)
    assert costs["status"] == "costing opportunities" and costs["ci95"][1] < 0.5
    even = _group(rows(["avoided"] * 20 + ["missed"] * 19), 30, rejection=True)
    assert even["status"] == "no evidence either way"
