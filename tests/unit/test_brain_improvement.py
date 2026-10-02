"""Self-improvement proposals come from the record — graded performance, run history, events, lab results —
and are never applied automatically or aimed at risk controls."""

from datetime import UTC, datetime, timedelta

from quantpulse.brain.improvement import PIPELINE, ImprovementEngine
from quantpulse.brain.store import BrainStore
from quantpulse.db.models import (
    BrainAgentPerformanceRow,
    BrainAgentRunRow,
    BrainEventRow,
    BrainStrategyRow,
)

NOW = datetime(2026, 9, 26, 15, 0, tzinfo=UTC)


def perf(agent, regime, n, hit, reliability, verdict=None, n_effective=None):
    return BrainAgentPerformanceRow(agent_id=agent, agent_version="1.0.0", regime=regime, horizon_days=5, window="all",
                                    n=n, hits=round(n * hit), hit_rate=hit, brier=0.26, ic=-0.05, calibration=[],
                                    reliability=reliability, computed_at=NOW, verdict=verdict,
                                    n_effective=n_effective)  # fmt: skip


async def seed(database):
    store = BrainStore(database)
    cycle = await store.start_cycle(kind="full", trigger="t", session="market_open", mode="dry_run", now=NOW)
    async with database.session() as s:
        s.add_all([
            perf("technical", "all", 80, 0.41, 0.72),  # weak overall
            perf("momentum", "all", 80, 0.56, 1.2),
            perf("momentum", "high_volatility", 30, 0.37, 0.6),  # fails in one regime
            perf("statistical", "all", 12, 0.25, None),  # too few calls to judge
            # 44% on 400 predictions but only 60 independent ones and no significant evidence: no claim
            perf("valuation", "all", 400, 0.44, 1.0, verdict="no evidence either way", n_effective=60),
            perf("factor", "all", 400, 0.40, 0.7, verdict="evidence of harm", n_effective=200),
        ])  # fmt: skip
        for i in range(12):
            s.add(BrainAgentRunRow(cycle_id=cycle, agent_id="options", agent_version="1.0.0", status="skipped",
                                   reason="no live option chains this cycle", started_at=NOW - timedelta(hours=i),
                                   duration_ms=0.0, subjects=0, opinions=0, model_tier="deterministic", cost=0.0))  # fmt: skip
        for i in range(6):
            s.add(BrainEventRow(type="QuoteBecameStale", subject="XYZ", payload={}, cycle_id=None,
                                created_at=NOW - timedelta(days=i)))  # fmt: skip
        s.add(BrainStrategyRow(strategy_id="short_term_reversal", version=1, name="Short-term reversal", spec={},
                               status="rejected", source="template",
                               validation={"verdict": "rejected", "gates": [{"gate": "deflated Sharpe", "passed": False}]},
                               paper={}, created_at=NOW, updated_at=NOW))  # fmt: skip


async def test_proposals_are_structured_evidence_based_and_deduplicated(database):
    await seed(database)
    engine = ImprovementEngine(database, min_observations=30, min_confidence=0.45)
    found = await engine.review(NOW)
    by = {(f["kind"], f["target"]): f for f in found}
    assert ("agent", "technical@1.0.0") in by
    assert ("routing", "momentum@1.0.0") in by and "high_volatility" in by[("routing", "momentum@1.0.0")][
        "title"
    ]
    assert ("capability", "options") in by and ("data", "XYZ") in by
    assert ("strategy", "short_term_reversal@v1") in by
    assert not any(f["target"].startswith("statistical") for f in found)  # 12 calls: no judgement yet
    assert not any(f["target"].startswith("valuation") for f in found)  # not significant: no finding
    assert by[("agent", "factor@1.0.0")]["evidence"]["independent_calls"] == 200
    for f in found:
        p = f["proposal"]
        assert (
            f["title"]
            and f["evidence"]
            and p["change"]
            and p["expected_improvement"]
            and p["validation_plan"]
        )
        assert p["pipeline"] == PIPELINE
        assert "QP_TRADING" not in str(f) and f["kind"] != "risk"  # risk controls are never a subject
    tech = by[("agent", "technical@1.0.0")]["evidence"]
    assert tech["graded_calls"] == 80 and tech["hit_rate"] == 0.41

    stored = await engine.proposals()
    assert len(stored) == len(found) and all(p["status"] == "proposed" for p in stored)
    await engine.review(NOW + timedelta(hours=1))
    assert len(await engine.proposals()) == len(stored)  # refreshed, not duplicated

    decided = await engine.decide(stored[0]["id"], "rejected", "user", "not convinced", NOW)
    assert decided["status"] == "rejected" and decided["decided_by"] == "user"
    assert decided["test_result"]["note"] == "not convinced"


async def test_an_empty_record_proposes_nothing(database):
    assert await ImprovementEngine(database, 30, 0.45).review(NOW) == []


def test_a_proposal_that_names_a_protected_control_is_withheld():
    from quantpulse.brain.improvement import _p, withheld

    loosen = _p("stale_data", "AAPL", "quotes keep going stale", {}, "raise QP_TRADING_MAX_QUOTE_AGE_SECONDS to 3600",
                "more trades", [])  # fmt: skip
    assert withheld(loosen) == "QP_TRADING_MAX_"
    for name in (
        "qp_trading_kill_switch",
        "QP_ALPACA_PAPER",
        "QP_TRADING_DAILY_LOSS_ACTION",
        "QP_BRAIN_KILL_SWITCH",
    ):
        assert withheld(_p("k", "t", "x", {}, f"change {name}", "", [])) is not None
    data = _p("stale_data", "@market", "stale", {}, "a SIP subscription (QP_ALPACA_STOCK_FEED=sip)", "", [])
    assert withheld(data) is None  # saying data is the problem is allowed; moving a limit is not
    assert (
        withheld(_p("calibration", "consensus", "x", {}, "revisit QP_BRAIN_MIN_CONFIDENCE", "", [])) is None
    )


async def test_a_proposal_touching_a_protected_control_is_recorded_for_review_never_suggested(
    database, monkeypatch
):
    from quantpulse.brain.improvement import _p

    engine = ImprovementEngine(database, 30, 0.45)

    async def unsafe():
        return [
            _p("decision", "risk", "blocks cost trades", {}, "set QP_TRADING_REQUIRE_LIVE_DATA=false", "", [])
        ]

    monkeypatch.setattr(engine, "_assumptions", unsafe)
    [written] = await engine.review(NOW)
    assert written["status"] == "protected_review"
    assert written["evidence"]["protected_control"] == "QP_TRADING_REQUIRE_LIVE_DATA"
    assert await engine.proposals("proposed") == []  # never offered as a suggestion
    [kept] = await engine.proposals("protected_review")
    assert "never changes a protected control" in kept["proposal"]["protected"]
    await engine.review(NOW)  # reviewed again: updated in place, not duplicated
    assert len(await engine.proposals()) == 1
