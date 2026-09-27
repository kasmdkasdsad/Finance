"""The learning loop against a real database: grading against fixed closing prices, decision outcomes,
decision-vs-outcome reflection, measured track records (only after enough graded calls), failure analysis,
and the consensus using measured reliability."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select

from quantpulse.brain import performance as perf
from quantpulse.brain.consensus import ReliabilityBook, build_consensus
from quantpulse.brain.evaluation import evaluate_due, grade
from quantpulse.brain.learning import Learner
from quantpulse.brain.memory import AGENT, LONG_TERM, MemoryStore
from quantpulse.brain.reflection import category, decision_quality, failure_analysis, outcome_quality
from quantpulse.brain.store import BrainStore
from quantpulse.brain.types import DataState, Opinion, stance_of
from quantpulse.core.clock import FakeClock
from quantpulse.db.models import (
    BrainConsensusRow,
    BrainDebateRow,
    BrainDecisionRow,
    BrainPredictionRow,
    BrainReflectionRow,
)

MADE = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)  # Tuesday
NOW = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)  # Friday after the close


class FixedPrices:
    def __init__(self, table: dict[str, dict[date, float]]) -> None:
        self.table = table
        self.asked: list[str] = []

    async def closes(self, symbols, since):
        self.asked.extend(symbols)
        return {s: self.table[s] for s in symbols if s in self.table}


def prediction(subject="AAA", *, source="technical", direction=1, score=0.6, confidence=0.7, due=date(2026, 9, 10),
               entry=100.0, bench=500.0, regime="bullish", cycle_id=None, source_type="agent", benchmark="SPY"):  # fmt: skip
    return BrainPredictionRow(
        cycle_id=cycle_id, source_type=source_type, source_id=source, source_version="1.0.0", subject=subject,
        direction=direction, score=score * direction, confidence=confidence, horizon_days=5, benchmark=benchmark,
        regime=regime, made_at=MADE, due_date=due, entry_price=entry, entry_benchmark=bench, status="open", context={},
    )  # fmt: skip


def test_grade_relative_and_absolute():
    assert grade(1, 100, 110, 500, 505, absolute=False) == pytest.approx((0.10, 0.09, True))
    assert grade(-1, 100, 101, 500, 520, absolute=False)[2] is True  # lagged the benchmark: a bearish hit
    assert grade(1, 100, 99, None, None, absolute=True) == pytest.approx((-0.01, -0.01, False))
    assert grade(1, 100, 110, None, None, absolute=False) is None  # no benchmark leg, no grade


async def test_evaluation_grades_due_calls_voids_missing_ones_and_leaves_the_rest_open(database):
    async with database.session() as s:
        s.add_all(
            [
                prediction("AAA"),  # due, priced
                prediction(
                    "@market", source="market_regime", direction=-1, benchmark="absolute", entry=500.0
                ),
                prediction("GONE", due=date(2026, 9, 3)),  # due long ago, never priced -> void
                prediction("LATE", due=date(2026, 9, 24)),  # due yesterday, no price yet -> pending
                prediction("AAA", due=date(2026, 10, 30)),  # not due
            ]
        )
    prices = FixedPrices({"AAA": {date(2026, 9, 10): 108.0}, "SPY": {date(2026, 9, 10): 510.0}})
    result = await evaluate_due(database, prices, FakeClock(NOW), "SPY")
    assert (result.evaluated, result.voided, result.pending) == (2, 1, 1)
    async with database.session() as s:
        rows = {(r.subject, r.due_date): r for r in (await s.scalars(select(BrainPredictionRow))).all()}
    aaa = rows[("AAA", date(2026, 9, 10))]
    assert aaa.status == "evaluated" and aaa.realized_return == pytest.approx(0.08)
    assert aaa.realized_relative == pytest.approx(0.08 - 0.02) and aaa.hit is True
    market = rows[("@market", date(2026, 9, 10))]
    assert market.realized_relative == pytest.approx(0.02) and market.hit is False  # bearish, SPY rose
    assert rows[("GONE", date(2026, 9, 3))].status == "void"
    assert (
        rows[("LATE", date(2026, 9, 24))].status == "open"
        and rows[("AAA", date(2026, 10, 30))].status == "open"
    )


# ---------------------------------------------------------------------------------------------- reflection
def decision(action="buy", approved=True, relative=0.03):
    return BrainDecisionRow(cycle_id=1, subject="AAA", action=action, mode="paper_recommendation", status="recommended",
                            confidence=0.6, risk_approved=approved, rationale={"reasons": ["x"]}, risk={}, execution={},
                            outcome={"relative": relative}, created_at=MADE)  # fmt: skip


def consensus_row(
    confidence=0.62, disagreement=0.1, data="fresh", voters=("technical", "factor", "valuation")
):
    votes = [{"agent_id": a, "score": 0.5, "reliability": {"status": "unproven"}} for a in voters]
    return BrainConsensusRow(cycle_id=1, subject="AAA", stance="bullish", score=0.5, confidence=confidence,
                             unknown=False, disagreement=disagreement, detail={"votes": votes}, vetoes=[],
                             data_quality=data, reasons=[], created_at=MADE)  # fmt: skip


def debate_row(verdict="stands", objections=()):
    return BrainDebateRow(cycle_id=1, subject="AAA", stance_before="bullish", confidence_before=0.6,
                          confidence_after=0.6, verdict=verdict, bull=[], bear=[], objections=list(objections),
                          change_our_mind=[], created_at=MADE)  # fmt: skip


def test_decision_quality_is_judged_without_the_outcome():
    good, checks = decision_quality(decision(), consensus_row(), debate_row())
    assert good == "good" and all(checks["hard"].values())
    stale, _ = decision_quality(decision(), consensus_row(data="stale"), debate_row())
    challenged, _ = decision_quality(decision(), consensus_row(), debate_row("challenged"))
    unapproved, _ = decision_quality(decision(approved=False), consensus_row(), debate_row())
    assert stale == challenged == unapproved == "poor"
    shaky, checks = decision_quality(
        decision(), consensus_row(confidence=0.46, disagreement=0.45, voters=("technical", "momentum")),
        debate_row("weakened", [{"code": "one_idea", "severity": "medium", "text": "one idea"}]),
    )  # fmt: skip
    assert shaky == "fair" and checks["soft_score"] < 0.6


def test_the_four_quadrants_and_blocked_ideas():
    assert outcome_quality(1, 0.02) == "good" and outcome_quality(-1, 0.02) == "bad"
    assert outcome_quality(1, 0.002) == "neutral"
    assert category("good", "good", False) == "earned"
    assert category("good", "bad", False) == "unlucky"  # sound process, bad luck: do not punish the process
    assert category("fair", "good", False) == "lucky"  # weak process, good luck: do not reward it
    assert category("poor", "bad", False) == "process_failure"
    assert category("good", "bad", True) == "block_saved_money"
    assert category("good", "good", True) == "block_cost_opportunity"
    assert category("good", "neutral", False) == "inconclusive"


async def test_learning_pass_reflects_and_remembers(database):
    clock = FakeClock(NOW)
    store, memory = BrainStore(database), MemoryStore(database)
    cycle_id = await store.start_cycle(
        kind="full", trigger="test", session="market_open", mode="paper_recommendation", now=MADE
    )
    async with database.session() as s:
        c = consensus_row()
        c.cycle_id = cycle_id
        s.add(c)
        await s.flush()
        d = decision()
        d.cycle_id, d.consensus_id, d.outcome = cycle_id, c.id, {}
        s.add(d)
        deb = debate_row("weakened", [{"code": "extended", "severity": "medium", "text": "chasing"}])
        deb.cycle_id = cycle_id
        s.add(deb)
        s.add(prediction(source="consensus", source_type="consensus", cycle_id=cycle_id))
        s.add(prediction(source="technical", cycle_id=cycle_id))
    prices = FixedPrices({"AAA": {date(2026, 9, 10): 95.0}, "SPY": {date(2026, 9, 10): 505.0}})  # AAA fell
    learner = Learner(database, prices, clock, memory, "SPY", min_observations=30)
    summary = await learner.learn()
    assert summary["evaluated"] == 2 and summary["decision_outcomes"] == 1 and summary["reflections"] == 1
    async with database.session() as s:
        refl = (await s.scalars(select(BrainReflectionRow))).one()
        dec = (await s.scalars(select(BrainDecisionRow))).one()
    assert dec.outcome["relative"] == pytest.approx(-0.06) and dec.evaluated_at is not None
    assert refl.outcome_quality == "bad" and refl.decision_quality in ("good", "fair")
    assert refl.category in ("unlucky", "process_failure")
    assert any("objection 'extended' was borne out" in lesson for lesson in refl.lessons)
    assert refl.questions["Which agents were wrong?"] == [] or isinstance(
        refl.questions["Which agents were wrong?"], list
    )
    rows = await store.performance("all")
    assert {r["agent_id"] for r in rows} == {"technical", "consensus"}
    assert all(r["reliability"] is None for r in rows)  # one graded call is not a track record
    again = await learner.learn()
    assert again["evaluated"] == 0 and again["reflections"] == 0  # nothing is graded or reflected twice
    if refl.category == "process_failure":
        assert await memory.recall(tier=LONG_TERM, kind="lesson", now=NOW)


# ---------------------------------------------------------------------------------------------- performance
def graded_rows(n, hit_rate, *, source="technical", regime="bullish", confidence=0.5):
    hits = round(n * hit_rate)
    return [
        perf.Graded(
            source, "1.0.0", regime, 5, 0.5, confidence, 0.01 if i < hits else -0.01, i < hits, MADE, 1
        )
        for i in range(n)
    ]


def test_reliability_needs_enough_graded_calls_and_is_shrunk():
    assert perf.metrics(graded_rows(29, 0.9), 30)["reliability"] is None
    m = perf.metrics(graded_rows(100, 0.6), 30)
    assert m["hit_rate"] == 0.6 and m["reliability"] == pytest.approx(
        1 + 4 * ((60 + 10) / 120 - 0.5), abs=1e-4
    )
    low = perf.metrics(graded_rows(100, 0.3), 30)["reliability"]
    assert low == pytest.approx(1 + 4 * ((30 + 10) / 120 - 0.5), abs=1e-4)
    assert perf.metrics(graded_rows(200, 0.0), 30)["reliability"] == 0.25  # bounded
    assert m["brier"] is not None and m["calibration"][-1]["kind"] == "summary"


async def test_measured_reliability_reaches_the_consensus(database):
    async with database.session() as s:
        for i in range(40):
            p = prediction(source="technical", confidence=0.6)
            p.status, p.hit, p.realized_relative, p.realized_return = (
                "evaluated",
                i < 32,
                0.01 if i < 32 else -0.01,
                0.0,
            )
            s.add(p)
            q = prediction(source="momentum", confidence=0.6)
            q.status, q.hit, q.realized_relative, q.realized_return = (
                "evaluated",
                i < 12,
                0.01 if i < 12 else -0.01,
                0.0,
            )
            s.add(q)
    await perf.recompute(database, NOW, min_observations=30)
    book = ReliabilityBook(await BrainStore(database).reliability_rows(), 30)
    good, bad = book.get("technical", "1.0.0", "bullish"), book.get("momentum", "1.0.0", "bullish")
    assert good.status == bad.status == "measured" and good.weight > 1.0 > bad.weight
    assert book.get("technical", "2.0.0").status == "unproven"

    def op(agent, score):
        return Opinion(
            agent, "1.0.0", "AAA", stance_of(score), score, 0.6, 5, "", data_quality=DataState.FRESH
        )

    c = build_consensus(
        "AAA", [op("technical", 0.5), op("momentum", -0.5)], reliability=book, regime="bullish"
    )
    assert c.score > 0  # the agent with the better record now carries more weight


def test_failure_analysis_names_where_an_agent_fails():
    rows = graded_rows(40, 0.3, regime="high_volatility", confidence=0.7) + graded_rows(
        40, 0.6, regime="bullish", confidence=0.3
    )
    findings = {f["agent_id"]: f for f in failure_analysis(rows, 30)}
    notes = " ".join(findings["technical"]["notes"])
    assert "weak in high_volatility markets" in notes
    assert "confidently wrong" in notes and "miscalibrated" in notes
    assert failure_analysis(graded_rows(10, 0.2), 30) == []  # too few calls to say anything


async def test_agent_memory_holds_measured_performance(database):
    async with database.session() as s:
        for i in range(35):
            p = prediction(source="technical", confidence=0.7, regime="high_volatility")
            p.status, p.hit, p.realized_relative, p.realized_return = (
                "evaluated",
                i < 10,
                0.01 if i < 10 else -0.01,
                0.0,
            )
            s.add(p)
    memory = MemoryStore(database)
    learner = Learner(database, FixedPrices({}), FakeClock(NOW + timedelta(days=1)), memory, "SPY", 30)
    summary = await learner.learn()
    assert summary["agents_with_findings"] == 1
    remembered = await memory.recall(tier=AGENT, kind="performance", now=NOW + timedelta(days=1))
    assert remembered and "technical" in remembered[0]["summary"] and "29%" in remembered[0]["summary"]
