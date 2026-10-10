"""Learning beyond hit rates: calls graded at every horizon and in each volatility environment, recurring
patterns consolidated into memory (tentative until the evidence is established) and recalled into
decisions, and improvement proposals only where the record supports them."""

from datetime import UTC, date, datetime, timedelta

from quantpulse.brain import performance as perf
from quantpulse.brain.evaluation import by_horizon
from quantpulse.brain.improvement import ImprovementEngine
from quantpulse.brain.memory import LONG_TERM, MemoryStore
from quantpulse.brain.patterns import MIN_PATTERN, consolidate, find, recall
from quantpulse.db.models import (
    BrainAgentPerformanceRow,
    BrainCycleRow,
    BrainOpportunityRow,
    BrainPredictionRow,
    BrainReflectionRow,
)

NOW = datetime(2026, 9, 26, 15, 0, tzinfo=UTC)
MADE = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)


def test_every_call_is_also_graded_at_the_standard_horizons_up_to_its_own():
    row = BrainPredictionRow(subject="AAA", direction=1, horizon_days=10, benchmark="SPY", made_at=MADE,
                             due_date=date(2026, 9, 16), entry_price=100.0, entry_benchmark=500.0)  # fmt: skip
    days = [date(2026, 9, d) for d in (2, 3, 4, 8, 9, 10, 11, 14, 15, 16)]
    series = {d: 100.0 + i for i, d in enumerate(days, start=1)}
    bench = dict.fromkeys(days, 500.0)
    out = by_horizon(row, series, bench)
    assert set(out) == {"1", "5", "10"}  # 21 is beyond its own horizon
    assert out["1"] == 0.01 and out["5"] == 0.05 and out["10"] == 0.10


def test_slices_by_horizon_and_volatility_environment():
    rows = [
        perf.Graded("technical", "1.0.0", "bullish", 21, 0.5, 0.6, -0.01, False, MADE + timedelta(days=30 * i), 1,
                    subject=f"S{i}", market_vol=0.25 if i % 2 else 0.10, by_horizon={"5": 0.02, "21": -0.01})
        for i in range(40)
    ]  # fmt: skip
    assert perf.vol_environment(0.10) == "low" and perf.vol_environment(0.15) == "normal"
    assert perf.vol_environment(0.25) == "high" and perf.vol_environment(None) is None
    at5 = [perf.at_horizon(r, "5") for r in rows]
    assert all(g is not None and g.hit and g.horizon == 5 for g in at5)  # right at 5 sessions, wrong at 21
    assert perf.metrics([g for g in at5 if g], 30)["verdict"] == "evidence of skill"
    assert perf.metrics(rows, 30)["verdict"] == "evidence of harm"


def reflection(i, *, objections=(), kinds=(), outcome="bad", action="buy", category="process_failure"):
    return BrainReflectionRow(subject_type="decision", subject_id=i, category=category, decision_quality="fair",
                              outcome_quality=outcome, questions={}, lessons=["x"], created_at=NOW,
                              evidence={"action": action, "subject": f"S{i}", "kinds": list(kinds),
                                        "objections": [{"code": c, "severity": "medium", "borne_out": outcome == "bad"}
                                                       for c in objections]})  # fmt: skip


async def seed_reflections(database):
    async with database.session() as s:
        for i in range(40):  # 'extended' raised 40 times, borne out 34
            s.add(
                reflection(
                    i, objections=["extended"], kinds=["breakout"], outcome="bad" if i < 34 else "good"
                )
            )
        for i in range(40, 45):  # too few to report
            s.add(reflection(i, objections=["fighting_the_regime"]))


async def test_patterns_are_counted_tentative_or_established_and_updated_in_place(database):
    await seed_reflections(database)
    found = {(p["kind"], p["name"]): p for p in await find(database, 30)}
    assert ("objection", "fighting_the_regime") not in found  # fewer than MIN_PATTERN observations
    extended = found[("objection", "extended")]
    assert extended["n"] == 40 and extended["k"] == 34 and extended["status"] == "established"
    assert "34 of 40" in extended["summary"] and "interval" in extended["summary"]
    breakout = found[("hypothesis", "breakout")]
    assert breakout["rate"] == 6 / 40 and "failed hypothesis" in breakout["summary"]
    assert found[("process", "outcomes")]["n"] == 45

    memory = MemoryStore(database)
    assert await consolidate(database, memory, NOW, 30) == len(found)
    assert await consolidate(database, memory, NOW + timedelta(days=1), 30) == len(found)
    stored = await memory.recall(tier=LONG_TERM, kind="pattern", limit=50)
    assert len(stored) == len(found)  # one memory per pattern, refreshed, never duplicated

    lessons = [{"subject": "AAA", "summary": "process failure: bought an extended move", "created_at": NOW}]
    said = recall(
        "AAA", patterns=stored, lessons=lessons, objections=["extended"], kinds=["breakout"], regime=None
    )
    assert said[0].startswith("pattern:") and any("extended" in x for x in said)
    assert any("breakout" in x for x in said) and any(x.startswith("lesson") for x in said)
    assert recall("ZZZ", patterns=stored, lessons=lessons, objections=[], kinds=[], regime=None) == []


def perf_row(agent, regime, horizon, verdict, n_eff=60, hit=0.6):
    return BrainAgentPerformanceRow(agent_id=agent, agent_version="1.0.0", regime=regime, horizon_days=horizon,
                                    window="all", n=n_eff * 3, hits=0, hit_rate=hit, brier=0.24, ic=0.05,
                                    calibration=[], reliability=1.0, computed_at=NOW, verdict=verdict,
                                    n_effective=n_eff, ci_low=0.52, ci_high=0.7, q_value=0.02)  # fmt: skip


async def test_new_improvement_discoveries(database):
    await seed_reflections(database)
    async with database.session() as s:
        s.add_all([
            perf_row("momentum", "all", 21, "no evidence either way", hit=0.51),
            perf_row("momentum", "at:5d", 5, "evidence of skill"),  # works at 5 sessions, not at 21
            perf_row("technical", "vol:high", 5, "evidence of harm", hit=0.38),
        ])  # fmt: skip
        cycle = BrainCycleRow(kind="full", trigger="t", session="market_open", mode="dry_run", status="completed",
                              started_at=NOW - timedelta(days=40))  # fmt: skip
        s.add(cycle)
        await s.flush()
        for i in range(12):
            fit = "poor fit" if i < 9 else "fits"
            s.add(BrainOpportunityRow(cycle_id=cycle.id, kind="relative_value", subject=f"P{i}", symbols=[f"P{i}"],
                                      direction=1, strength=0.5, headline="h", evidence=[], status="watch",
                                      stages=[{"stage": "portfolio_fit", "result": fit}], created_at=NOW))  # fmt: skip
        for i in range(12):
            causes = ["3 of 20 IEX-priced symbols have no IEX print within 600s"]
            s.add(BrainCycleRow(kind="full", trigger="t", session="market_open", mode="dry_run", status="completed",
                                started_at=NOW - timedelta(hours=i), data_quality={"feed": {"market_open": True,
                                "causes": causes, "clock_skew_s": 3.5 if i < 4 else 0.1}}))  # fmt: skip
    found = await ImprovementEngine(database, 30, 0.45).review(NOW)
    by = {(f["kind"], f["target"]): f for f in found}
    assert "5 sessions" in by[("horizon", "momentum@1.0.0")]["title"]
    assert "high-volatility markets" in by[("routing", "technical@1.0.0")]["title"]
    assert by[("opportunity", "relative_value")]["evidence"]["failed_fit"] == 9
    assert ("data", "stock_feed") in by and "person's decision" in by[("data", "stock_feed")]["proposal"][
        "change"
    ]
    assert by[("data", "system_clock")]["evidence"]["cycles_with_skew_over_2s"] == 4
    assert "usually right" in by[("assumption", "objection:extended")]["title"]
    assert "keep losing" in by[("assumption", "hypothesis:breakout")]["title"]
    assert not any("fighting_the_regime" in f["target"] for f in found)  # not enough evidence
    assert MIN_PATTERN == 10
