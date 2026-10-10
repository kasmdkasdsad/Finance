"""The learning report without I/O: regime buckets, calibration error, an agent against the consensus on
the same calls, consensus patterns and data mistakes — each unproven until the sample is there."""

from datetime import UTC, datetime, timedelta

import numpy as np

from quantpulse.brain.learning_report import (
    buckets,
    calibration,
    consensus_patterns,
    data_mistakes,
    regime_matrix,
    versus_consensus,
)
from quantpulse.brain.performance import Graded

T0 = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)


def call(i, hit, *, source="consensus", direction=1, score=0.6, conf=0.8, regime="bullish", vol=0.15, event=False,
         subject=None, cycle=None, data_ok=True, context=None) -> Graded:  # fmt: skip
    """Independent calls: a different subject every time, a week apart (no overlapping horizons)."""
    rel = 0.02 if hit else -0.02
    return Graded(source=source, version="1", regime=regime, horizon=5, score=score * direction, confidence=conf,
                  relative=rel * direction, hit=hit, made_at=T0 + timedelta(days=7 * i), direction=direction,
                  subject=subject or f"S{i}", data_ok=data_ok, market_vol=vol, cycle_id=cycle if cycle is not None else i,
                  market_event=event, context=context or {})  # fmt: skip


def test_a_call_can_sit_in_several_regime_buckets():
    assert buckets("bullish", 0.15, False) == ["bullish"]
    assert buckets("neutral", 0.10, False) == ["sideways", "low_volatility"]
    assert buckets("risk_off", 0.35, True) == ["bearish", "high_volatility", "event"]
    assert buckets("high_volatility", None, None) == ["high_volatility"]


def test_calibration_measures_how_far_confidence_is_from_the_hit_rate():
    rng = np.random.default_rng(1)
    # implied probability of being right = 0.5 + 0.5 × 0.6 × 0.8 = 0.74; they are right 50% of the time
    over = [call(i, bool(rng.random() < 0.5)) for i in range(80)]
    cal = calibration(over, 30)
    assert cal["n_effective"] == 80 and cal["status"] == "overconfident" and cal["bias"] > 0.15
    assert 0.15 < cal["ece"] < 0.35
    good = [call(i, bool(rng.random() < 0.74)) for i in range(400)]
    assert calibration(good, 30)["ece"] < 0.08
    assert calibration(over[:10], 30)["status"] == "unproven" and calibration(over[:10], 30)["needs"] == 20


def test_an_agent_is_compared_with_the_consensus_on_the_same_calls():
    consensus, agent = [], []
    for i in range(60):  # they disagree on every call; the agent is right 45 times out of 60
        consensus.append(call(i, i >= 45, direction=1))
        agent.append(call(i, i < 45, source="momentum", direction=-1))
    vs = versus_consensus(agent, consensus, 30)
    assert vs["pairs"] == 60 and vs["agreement"] == 0 and vs["disagreements"] == 60
    assert vs["agent_right_when_disagreeing"] == 0.75 and vs["status"].startswith("adds information")
    few = versus_consensus(agent[:8], consensus[:8], 30)
    assert few["status"] == "unproven" and few["needs"] > 0
    # the same call repeated in several cycles of one day is one pair
    same_day = [call(0, True, source="momentum", cycle=c, subject="AAA") for c in (1, 2, 3)]
    cons = [call(0, True, cycle=c, subject="AAA") for c in (1, 2, 3)]
    assert versus_consensus(same_day, cons, 30)["pairs"] == 1


def test_regime_cells_are_adjusted_together_and_unproven_when_thin():
    rng = np.random.default_rng(2)
    rows = {
        "consensus": [call(i, bool(rng.random() < 0.8), regime="bullish") for i in range(60)]
        + [call(100 + i, bool(rng.random() < 0.5), regime="neutral", vol=0.1) for i in range(12)],
    }
    m = regime_matrix(rows, 30)["consensus"]
    assert m["bullish"]["verdict"] == "evidence of skill" and m["bullish"]["q_value"] is not None
    assert m["sideways"]["verdict"] == "unproven" and m["sideways"]["needs"] == 18
    assert m["low_volatility"]["n_effective"] == 12


def test_consensus_patterns_and_data_mistakes():
    rows = [call(i, i % 3 != 0, context={"independent_sources": 2, "disagreement": 0.1, "debate": "passed"})
            for i in range(40)]  # fmt: skip
    rows += [call(100 + i, i % 2 == 0, context={"independent_sources": 1, "disagreement": 0.5, "debate": "challenged"})
             for i in range(10)]  # fmt: skip
    p = consensus_patterns(rows, 30)
    assert p["two_or_more_sources"]["n_effective"] == 40 and p["two_or_more_sources"]["status"] == "measured"
    assert p["one_source"]["status"] == "unproven" and p["contested"]["n_effective"] == 10
    assert p["challenged_by_devils_advocate"]["n_effective"] == 10 and "by_regime" in p["not_challenged"]
    bad = [call(200 + i, False, data_ok=False, context={"data_status": "stale"}) for i in range(5)]
    d = data_mistakes(rows + bad, 30)
    assert d["on_unusable_data"]["calls"] == 5 and d["on_unusable_data"]["hit_rate"] == 0
    assert d["by_data_status"]["stale"]["calls"] == 5 and d["status"] == "unproven"
