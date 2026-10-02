"""The 24/7 research subsystem's pure rules: how a conclusion is judged from its evidence, how the queue orders
questions, which operating mode each moment of the week is in, and when research must wait or stop for memory."""

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from quantpulse.brain.research import operating
from quantpulse.brain.research.catalog import CATALOG, PHASES
from quantpulse.brain.research.ledger import Finding, binomial_vs_half, check, judge, p_two_sided, t_test_mean
from quantpulse.brain.research.lifecycle import STAGES, promotion_refusal
from quantpulse.brain.research.queue import job_key, priority
from quantpulse.brain.research.resources import ResourceGovernor, Snapshot, measure
from quantpulse.config import Settings


def finding(**kw) -> Finding:
    base = dict(
        topic="agent:momentum",
        claim="momentum's calls beat a coin flip",
        sample_size=60,
        min_sample=30,
        regime="all",
        benchmark="a coin flip",
        method="block binomial test",
        limitations=["one market regime"],
        p_value=0.01,
        effect=0.08,
    )
    base.update(kw)
    return Finding(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ no fake learning
def test_a_small_sample_is_unproven_however_striking():
    status, conf = judge(finding(sample_size=12, p_value=0.0001, effect=0.4))
    assert (status, conf) == ("UNPROVEN", 0.0)


@pytest.mark.parametrize(("p", "effect"), [(None, 0.1), (0.01, None), (math.nan, 0.1), (math.inf, 0.1)])
def test_no_test_means_unproven(p, effect):
    assert judge(finding(p_value=p, effect=effect)) == ("UNPROVEN", 0.0)


def test_supported_refuted_inconclusive_follow_the_test():
    assert judge(finding())[0] == "SUPPORTED"
    assert judge(finding(effect=-0.08))[0] == "REFUTED"
    assert judge(finding(p_value=0.2)) == ("INCONCLUSIVE", 0.0)
    assert judge(finding(effect=0.0)) == ("INCONCLUSIVE", 0.0)


def test_confidence_grows_with_the_sample_and_never_reaches_certainty():
    small = judge(finding(sample_size=30, p_value=0.001))[1]
    large = judge(finding(sample_size=60, p_value=0.001))[1]
    assert 0 < small < large <= 0.99
    assert judge(finding(sample_size=10_000, p_value=0.0))[1] == 0.99


@pytest.mark.parametrize(
    ("change", "needs"),
    [
        ({"limitations": []}, "limitations"),
        ({"benchmark": ""}, "benchmark"),
        ({"method": ""}, "method"),
        ({"regime": ""}, "regime"),
        ({"min_sample": 0}, "counts"),
        ({"period_start": date(2026, 9, 2), "period_end": date(2026, 9, 1)}, "period"),
    ],
)
def test_a_finding_without_its_evidence_is_refused(change, needs):
    with pytest.raises(ValueError, match=needs):
        check(finding(**change))


def test_statistics_helpers():
    assert p_two_sided(0.0) == pytest.approx(1.0)
    assert p_two_sided(1.96) == pytest.approx(0.05, abs=1e-3)
    mean, t, p = t_test_mean([0.01] * 20 + [0.03] * 20)
    assert mean == pytest.approx(0.02) and t is not None and t > 5 and p is not None and p < 1e-6
    assert t_test_mean([0.1, 0.2]) == (pytest.approx(0.15), None, None)  # too few to test
    assert t_test_mean([0.1] * 5)[2] is None  # no variance, no test
    rate, p = binomial_vs_half(70, 100)
    assert rate == 0.7 and p is not None and p < 0.001


# ------------------------------------------------------------------ the queue's order
def test_priority_is_expected_information_value():
    never, _ = priority(7, "medium", last_status=None, age=None, refresh=timedelta(days=1), source="system")
    settled, detail = priority(
        7,
        "medium",
        last_status="SUPPORTED",
        age=timedelta(hours=1),
        refresh=timedelta(days=1),
        source="system",
    )
    stale, _ = priority(
        7,
        "medium",
        last_status="SUPPORTED",
        age=timedelta(days=5),
        refresh=timedelta(days=1),
        source="system",
    )
    heavy, _ = priority(7, "heavy", last_status=None, age=None, refresh=timedelta(days=1), source="system")
    asked, _ = priority(7, "medium", last_status=None, age=None, refresh=timedelta(days=1), source="person")
    assert never > stale > settled  # unanswered first; a settled answer only when it has gone stale
    assert heavy < never  # cost counts
    assert asked == pytest.approx(never + 5)  # a person's question jumps the queue
    assert detail["uncertainty"] == 0.5 and detail["last_conclusion"] == "SUPPORTED"
    assert {"value", "staleness", "cost_weight", "bonus"} <= set(detail)  # every order can be explained


def test_job_keys_identify_a_question_by_its_parameters():
    assert job_key("trade_review", {}) == "trade_review:-"
    a = job_key("trade_post_mortem", {"symbol": "AAA", "thesis_id": 3})
    assert a == job_key("trade_post_mortem", {"thesis_id": 3, "symbol": "AAA"})
    assert a != job_key("trade_post_mortem", {"symbol": "AAA", "thesis_id": 4})


def test_the_catalog_covers_the_closed_market_loop():
    assert {s.phase for s in CATALOG.values()} == set(PHASES)
    assert all(s.cost in ("light", "medium", "heavy") and 1 <= s.value <= 10 for s in CATALOG.values())
    assert CATALOG["reconcile_state"].owner_only  # the only job that reads the Alpaca account
    assert CATALOG["trade_post_mortem"].refresh is None  # asked once, as a follow-up
    assert {"grade_predictions", "agent_calibration", "decision_quality", "trade_review", "rejected_opportunities",
            "account_vs_benchmark", "feature_research", "agent_combinations", "strategy_research",
            "watchlist_prep", "data_quality", "system_integrity"} <= set(CATALOG)  # fmt: skip


# ------------------------------------------------------------------ the operating modes
def ny(y, m, d, hh, mm) -> datetime:
    from quantpulse.core.market_calendar import NEW_YORK

    return datetime(y, m, d, hh, mm, tzinfo=NEW_YORK).astimezone(UTC)


@pytest.mark.parametrize(
    ("moment", "mode", "costs"),
    [
        (ny(2026, 9, 25, 10, 0), "EXECUTION", None),  # Friday, in session
        (ny(2026, 9, 25, 15, 59), "EXECUTION", None),
        (ny(2026, 9, 25, 16, 30), "RESEARCH", ("light", "medium", "heavy")),  # after the close
        (ny(2026, 9, 26, 3, 0), "RESEARCH", ("light", "medium", "heavy")),  # Saturday night
        (ny(2026, 9, 27, 12, 0), "RESEARCH", ("light", "medium", "heavy")),  # Sunday
        (ny(2026, 9, 28, 6, 0), "RESEARCH", ("light", "medium", "heavy")),  # Monday, before preparation
        (ny(2026, 9, 28, 8, 30), "PRE_MARKET", ("light",)),  # preparation: light jobs only
        (ny(2026, 9, 28, 9, 20), "PRE_MARKET", None),  # nothing starts in the last 15 minutes
        (ny(2026, 11, 26, 11, 0), "RESEARCH", ("light", "medium", "heavy")),  # Thanksgiving: a holiday
    ],
)
def test_modes_and_what_research_may_run(moment, mode, costs):
    assert operating.mode_at(moment) == mode
    assert operating.research_costs(moment) == costs
    assert operating.LOOPS[mode]


def test_the_loops_are_the_operating_model():
    assert operating.LOOPS["EXECUTION"] == ("EXECUTE", "MONITOR", "RECONCILE", "LEARN")
    assert operating.LOOPS["RESEARCH"] == ("GRADE", "ANALYZE", "RESEARCH", "TEST", "LEARN", "PREPARE")
    assert operating.LOOPS["PRE_MARKET"][-1] == "EXECUTION READINESS"


# ------------------------------------------------------------------ resource limits
def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_research_waits_for_memory_and_cpu_and_heavy_jobs_need_more_room():
    s = settings(research_max_memory_pct=70, research_abort_memory_pct=85, research_max_load=0.85)
    gov = ResourceGovernor(s, lambda: Snapshot(50, "cgroup", 0.2, 300))
    assert gov.may_start("heavy")[0] and gov.may_start("light")[0] and not gov.must_stop()[0]
    gov = ResourceGovernor(s, lambda: Snapshot(65, "host", 0.2, 300))
    assert gov.may_start("medium")[0]
    ok, why = gov.may_start("heavy")
    assert not ok and "60%" in why  # a heavy job keeps a 10-point margin
    gov = ResourceGovernor(s, lambda: Snapshot(72, "cgroup", 0.2, 300))
    assert not gov.may_start("light")[0] and not gov.must_stop()[0]  # waits, but nothing is stopped
    gov = ResourceGovernor(s, lambda: Snapshot(30, "host", 1.4, 300))
    ok, why = gov.may_start("light")
    assert not ok and "CPU" in why
    gov = ResourceGovernor(s, lambda: Snapshot(86, "cgroup", 0.1, 300))
    stop, why = gov.must_stop()
    assert stop and "protect execution" in why


def test_research_limits_are_bounded():
    with pytest.raises(ValueError):
        settings(research_max_concurrent=3)  # never more than 2 at once on a small server
    with pytest.raises(ValueError):
        settings(research_abort_memory_pct=99)
    s = settings()
    assert s.research_enabled and s.research_max_concurrent == 1
    assert s.research_max_memory_pct < s.research_abort_memory_pct


def test_measure_reads_this_machine():
    snap = measure()
    assert 0 <= snap.memory_pct <= 100 and snap.memory_source in ("cgroup", "host")
    assert snap.load_per_cpu >= 0 and snap.rss_mb > 0


# ------------------------------------------------------------------ promotion is a person's
@pytest.mark.parametrize("who", ["brain", "Research", " system ", "supervisor", "lab", "automatic", ""])
def test_the_brain_never_promotes(who):
    history = [{"to": "EVALUATION", "passed": True}]
    assert "only a person" in promotion_refusal("EVALUATION", history, "k", who, "enough evidence")


def test_promotion_needs_a_passed_evaluation_and_a_note():
    ok = [{"to": "EVALUATION", "passed": True}]
    assert promotion_refusal("EVALUATION", ok, "k", "Kim", "the forward record held") is None
    assert "note" in promotion_refusal("EVALUATION", ok, "k", "Kim", "  ")
    for stage in STAGES[:-2]:
        assert "only an evaluated" in promotion_refusal(stage, ok, "k", "Kim", "note")
    assert "did not pass" in promotion_refusal(
        "EVALUATION", [{"to": "EVALUATION", "passed": None}], "k", "Kim", "n"
    )
