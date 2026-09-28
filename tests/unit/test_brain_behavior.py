"""Pathological behaviour, without I/O: each check on constructed records, with and without enough data."""

from datetime import UTC, date, datetime, timedelta

import numpy as np

from quantpulse.brain import behavior as b
from quantpulse.db.models import (
    BrainConsensusRow,
    BrainExecutionRow,
    BrainOpinionRow,
    BrainOpportunityOutcomeRow,
    BrainSessionRow,
    BrainThesisRow,
)

T0 = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)


def ex(symbol, side, day, qty=10, price=100.0):
    return BrainExecutionRow(symbol=symbol, side=side, filled_qty=qty, qty=qty, expected_price=price,
                             decided_at=T0 + timedelta(days=day), client_order_id=f"{symbol}{side}{day}")  # fmt: skip


def thesis(
    symbol, opened, closed, pnl, supporting=("technical", "momentum"), reason="bearish consensus", **kw
):
    return BrainThesisRow(symbol=symbol, origin="brain", status="closed", opened_at=T0 + timedelta(days=opened),
                          closed_at=T0 + timedelta(days=closed), realized_pnl=pnl, supporting=list(supporting),
                          exit_reason=reason, **kw)  # fmt: skip


def codes(findings, severity=None):
    return [f["code"] for f in findings if severity is None or f["severity"] == severity]


def test_repeated_buying_and_selling_is_an_alert():
    churn = [ex("AAA", side, d) for d, side in enumerate(["buy", "sell"] * 4)]  # four round trips
    assert codes(b.round_trips(churn, []), "alert") == ["round_trips"]
    assert b.round_trips([ex("AAA", "buy", 0), ex("AAA", "sell", 10)], []) == []  # one round trip: normal
    quick = [thesis(s, 0, 1, -5.0) for s in ("A", "B", "C")]  # closed a session after opening, not at a stop
    found = b.round_trips([], quick)
    assert found and found[0]["severity"] == "warning" and found[0]["evidence"]["symbols"] == ["A", "B", "C"]
    assert b.round_trips([], [thesis("A", 0, 1, -5.0, reason="at its stop (-8.1%)")]) == []  # stops are fine


def test_turnover_against_equity_and_the_shadow():
    days = [BrainSessionRow(day=date(2026, 9, i + 1), equity_close=100_000.0, traded_notional=80_000.0,
                            close={"strategy_shadow": {"turnover": 100_000.0}}) for i in range(5)]  # fmt: skip
    [f] = b.turnover(days)
    assert f["severity"] == "warning" and f["evidence"]["daily_turnover"] == 0.8
    assert f["evidence"]["shadow_daily_turnover"] == 0.2
    calm = [BrainSessionRow(day=date(2026, 9, 1), equity_close=100_000.0, traded_notional=5_000.0, close={})]
    assert b.turnover(calm)[0]["severity"] == "info"


def test_concentration_and_correlation():
    heavy = [
        BrainThesisRow(symbol="A", weight=0.28, sector="Tech"),
        BrainThesisRow(symbol="B", weight=0.25, sector="Tech"),
    ]
    [f] = b.concentration(heavy, 0.30)
    assert f["severity"] == "warning" and f["evidence"]["heavy_sectors"] == {"Tech": 0.53}
    spread = [BrainThesisRow(symbol=s, weight=0.1, sector=s) for s in "ABCDE"]
    assert b.concentration(spread, 0.30)[0]["severity"] == "info"
    pf = {"constraints": {"avg_correlation": 0.82}, "positions": {"A": {}, "B": {}, "C": {}}}
    assert b.correlated_positions(pf)[0]["severity"] == "warning"
    assert b.correlated_positions({"constraints": {}, "positions": {}}) == []


def test_repeated_losses_from_the_same_symbol_or_the_same_agents():
    closed = [thesis("AAA", 0, 3, -50.0), thesis("BBB", 4, 6, -20.0), thesis("AAA", 7, 9, -30.0)]
    found = b.repeated_losses(closed)
    assert "repeated_thesis_losses" in codes(found, "alert")  # three losers in a row, same agents
    assert any(f["evidence"].get("symbols") == {"AAA": 2} for f in found)
    mixed = [thesis("A", 0, 1, -5.0), thesis("B", 2, 3, 8.0, supporting=("factor",)), thesis("C", 4, 5, -5.0)]
    assert b.repeated_losses(mixed) == []


def test_a_kind_of_idea_never_taken_is_reported_with_what_it_did():
    ideas = [BrainOpportunityOutcomeRow(kind="breakout", market_open=True, reason="low_confidence", taken=False,
                                        verdict="missed" if i % 4 else "avoided") for i in range(24)]  # fmt: skip
    [f] = b.ignored_opportunities(ideas)
    assert f["severity"] == "warning" and f["evidence"]["missed"] == 18 and "none taken" in f["finding"]
    assert b.ignored_opportunities(ideas[:5]) == []  # too few to say anything
    taken_once = [
        *ideas,
        BrainOpportunityOutcomeRow(kind="breakout", market_open=True, reason="taken", taken=True),
    ]
    assert b.ignored_opportunities(taken_once) == []


def test_herding_and_lockstep_agents():
    cons = [BrainConsensusRow(supporting=4, opposing=0, unknown=False, subject=f"S{i}", stance="bullish",
                              created_at=T0) for i in range(60)]  # fmt: skip
    rng = np.random.default_rng(0)
    ops = []
    for i in range(60):
        x = float(rng.normal())
        ops += [BrainOpinionRow(cycle_id=i, subject="S", agent_id="technical", stance="bullish", score=x),
                BrainOpinionRow(cycle_id=i, subject="S", agent_id="momentum", stance="bullish", score=x * 0.9 + 0.01),
                BrainOpinionRow(cycle_id=i, subject="S", agent_id="factor", stance="bullish", score=float(rng.normal()))]  # fmt: skip
    found = b.herding(cons, ops)
    assert [f["severity"] for f in found] == ["warning", "warning"]
    pairs = found[1]["evidence"]["pairs"]
    assert [p["agents"] for p in pairs] == [["momentum", "technical"]]
    assert b.herding(cons[:10], ops[:30]) == []  # too few calls to judge


def test_consensus_flips_within_a_day():
    def c(stance, minute):
        return BrainConsensusRow(
            subject="AAA", stance=stance, unknown=False, created_at=T0 + timedelta(minutes=minute)
        )

    flippy = [c("bullish", 0), c("bearish", 30), c("bullish", 60)]
    [f] = b.instability(flippy)
    assert f["severity"] == "warning" and f["evidence"]["flips"]
    steady = [c("bullish", 0), c("bullish", 30), c("bullish", 60)]
    assert b.instability(steady)[0]["severity"] == "info"


def test_behaviour_after_a_losing_streak():
    closed = [thesis(s, i, 10 + i, -10.0) for i, s in enumerate("ABC")]  # three losers closed by day 12
    before = [ex(f"B{d}", "buy", d) for d in range(0, 12, 2)]  # one buy every other day
    after = [ex(f"A{d}{k}", "buy", d) for d in range(13, 23, 2) for k in range(3)]  # three a day after
    [f] = b.losing_streak(closed, before + after)
    assert f["severity"] == "warning" and f["evidence"]["buys_per_session"] == [1.0, 3.0]
    [thin] = b.losing_streak(closed, before[:2])
    assert thin["severity"] == "info" and "too few sessions" in thin["finding"]
