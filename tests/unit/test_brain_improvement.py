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


def perf(agent, regime, n, hit, reliability):
    return BrainAgentPerformanceRow(agent_id=agent, agent_version="1.0.0", regime=regime, horizon_days=5, window="all",
                                    n=n, hits=round(n * hit), hit_rate=hit, brier=0.26, ic=-0.05, calibration=[],
                                    reliability=reliability, computed_at=NOW)  # fmt: skip


async def seed(database):
    store = BrainStore(database)
    cycle = await store.start_cycle(kind="full", trigger="t", session="market_open", mode="dry_run", now=NOW)
    async with database.session() as s:
        s.add_all([
            perf("technical", "all", 80, 0.41, 0.72),  # weak overall
            perf("momentum", "all", 80, 0.56, 1.2),
            perf("momentum", "high_volatility", 30, 0.37, 0.6),  # fails in one regime
            perf("statistical", "all", 12, 0.25, None),  # too few calls to judge
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
