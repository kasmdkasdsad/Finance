"""The options research lab as a service, on the database: the research library and generation 0 seeded once,
a budgeted research run that evaluates strategies on real-shaped prices with MODEL-PRICED chains (labelled),
promotes one gate at a time with every step recorded, and never lets anything reach paper execution without
live shadow evidence."""

import math
from datetime import date, timedelta

import numpy as np
from sqlalchemy import select

from quantpulse.core.market_calendar import is_trading_day
from quantpulse.db.options_models import (
    OptionsStrategyBacktestRow,
    OptionsStrategyClaimRow,
    OptionsStrategySourceRow,
    OptionsStrategyVersionRow,
)
from quantpulse.options.lab.promotion import Stage
from quantpulse.options.lab.research import SEED


def gbm(seed: int, n: int = 700, mu: float = 0.10, vol: float = 0.25) -> dict[date, float]:
    rng = np.random.default_rng(seed)
    days, d = [], date(2023, 6, 1)
    while len(days) < n:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    r = rng.normal(mu / 252 - 0.5 * vol * vol / 252, vol / math.sqrt(252), n)
    return dict(zip(days, (100 * np.exp(np.cumsum(r))).tolist(), strict=True))


CLOSES = {"AAA": gbm(1), "BBB": gbm(2, mu=0.02, vol=0.3)}


async def test_the_library_and_generation_zero_are_seeded_once(api):
    lab = api.container.options_lab
    first = await lab.seed()
    assert first["sources"] == len(SEED) and first["versions"] >= 15
    assert await lab.seed() == {"sources": 0, "versions": 0}
    library = await lab.research_library()
    assert len(library) == len(SEED) and all(x["claim_status"] == "UNTESTED" for x in library)
    assert all("hypothesis" in x["note"] for x in library)
    strategies = await lab.strategies()
    assert strategies and all(s["stage"] in ("EXTRACTED", "RESEARCH") for s in strategies)
    assert all(s["stage_history"][0]["stage"] == "RESEARCH" for s in strategies)
    async with api.container.db.session() as s:
        assert len((await s.scalars(select(OptionsStrategySourceRow))).all()) == len(SEED)
        assert len((await s.scalars(select(OptionsStrategyClaimRow))).all()) == len(SEED)


async def test_a_research_run_evaluates_labels_and_promotes_one_gate_at_a_time(api):
    lab = api.container.options_lab
    report = await lab.research(closes=CLOSES, budget_seconds=240, max_evaluations=3)
    assert report["data"]["grade"] == "model" and "model-priced" in report["label"]
    assert len(report["evaluated"]) == 3
    async with api.container.db.session() as s:
        rows = (await s.scalars(select(OptionsStrategyBacktestRow))).all()
        versions = (await s.scalars(select(OptionsStrategyVersionRow))).all()
    assert rows and {r.data_source for r in rows} == {"model"}
    assert {r.execution_model for r in rows} == {
        "OPTIMISTIC",
        "MIDPOINT",
        "REALISTIC",
        "PESSIMISTIC",
        "STRESS",
    }
    order = [s.value for s in Stage]
    for v in versions:
        stages = [h["stage"] for h in v.stage_history]
        assert [order.index(x) for x in stages if x != "RETIRED"] == sorted(
            order.index(x) for x in stages if x != "RETIRED"
        )
        assert v.stage not in ("PAPER_ACTIVE", "PROVEN")  # never without live shadow and paper evidence
    evaluated = report["evaluated"][0]["version_id"]
    detail = await lab.strategy(evaluated)
    assert detail["backtests"] and detail["label"].startswith("evidence below PAPER_SHADOW is model-priced")
    assert detail["stage"] != "EXTRACTED" or detail["next_gate"]
    # a second run the same day re-evaluates nothing it just evaluated
    again = await lab.research(closes=CLOSES, budget_seconds=60, max_evaluations=3)
    assert not {e["version_id"] for e in again["evaluated"]} & {e["version_id"] for e in report["evaluated"]}


async def test_shadow_is_the_only_way_to_paper(api):
    lab = api.container.options_lab
    await lab.seed()
    async with api.container.db.session() as s:
        v = (
            await s.scalars(select(OptionsStrategyVersionRow).order_by(OptionsStrategyVersionRow.id))
        ).first()
        v.stage = Stage.PAPER_SHADOW.value
        vid = v.id
    await lab._promote_all(api.container.clock.now())
    detail = await lab.strategy(vid)
    assert detail["stage"] == "PAPER_SHADOW"
    assert any("shadow trades" in r for r in detail["next_gate"])


async def test_live_evidence_carries_a_strategy_to_paper_active_and_proven(api):
    """The rest of the ladder, through the lab's own promotion: a winning shadow record over enough sessions
    earns PAPER_ACTIVE (full-size paper trades); 50 winning paper trades with t ≥ 2 earn PROVEN. A losing
    shadow record earns nothing."""
    from quantpulse.db.options_models import OptionsPositionRow
    from quantpulse.options.lab.promotion import stage_record

    lab = api.container.options_lab
    await lab.seed()
    now = api.container.clock.now()
    evaluated = {"latest": {"validation_ror": 0.06, "walkforward_passed": True}, "fdr": {"discovery": True}}

    def closed(vid, mode, i, pnl):
        at = now - timedelta(days=60 - i)
        return OptionsPositionRow(version_id=vid, underlying="SPY", family="bull_put_spread", direction="bullish",
                                  mode=mode, structure={}, quantity=1, status="closed", opened_at=at - timedelta(days=5),
                                  entry_value=-100.0, entry_underlying=500.0, max_loss=400.0, closed_at=at,
                                  realized_pnl=pnl)  # fmt: skip

    async with api.container.db.session() as s:
        versions = (
            await s.scalars(select(OptionsStrategyVersionRow).order_by(OptionsStrategyVersionRow.id))
        ).all()
        win, lose = versions[0], versions[1]
        for v in (win, lose):
            v.stage = Stage.PAPER_SHADOW.value
            v.stage_history = [
                *v.stage_history,
                stage_record(Stage.PAPER_SHADOW, now.isoformat(), "test", evaluated),
            ]
        for i in range(12):  # 12 shadow trades on 12 sessions
            s.add(closed(win.id, "shadow", i, 60.0 if i % 4 else -40.0))
            s.add(closed(lose.id, "shadow", i, -50.0 if i % 4 else 30.0))
        ids = win.id, lose.id
    await lab._promote_all(now)
    stages = {x["id"]: x["stage"] for x in await lab.strategies()}
    assert stages[ids[0]] == "PAPER_ACTIVE" and stages[ids[1]] != "PAPER_ACTIVE"
    async with api.container.db.session() as s:
        for i in range(55):  # winning paper trades (realistic noise)
            s.add(closed(ids[0], "paper", i % 50, 50.0 if i % 3 else -30.0))
    await lab._promote_all(now)
    detail = await lab.strategy(ids[0])
    assert detail["stage"] == "PROVEN", detail["next_gate"]
    assert [h["stage"] for h in detail["stage_history"]][-2:] == ["PAPER_ACTIVE", "PROVEN"]


async def test_an_exploring_strategy_that_breaks_live_is_retired(api):
    """Exploration trades are watched like any live trades: a walk-forward strategy whose exploration contracts
    keep losing far beyond what it was validated to earn is retired. One without live trades is left alone."""
    from quantpulse.db.options_models import OptionsPositionRow, OptionsStrategyDecayRow
    from quantpulse.options.lab.promotion import stage_record

    lab = api.container.options_lab
    await lab.seed()
    now = api.container.clock.now()
    evaluated = {"latest": {"validation_ror": 0.06}, "fdr": {"discovery": True}}
    async with api.container.db.session() as s:
        versions = (
            await s.scalars(select(OptionsStrategyVersionRow).order_by(OptionsStrategyVersionRow.id))
        ).all()
        losing, quiet = versions[0], versions[1]
        for v in (losing, quiet):
            v.stage = Stage.WALK_FORWARD.value
            v.stage_history = [
                *v.stage_history,
                stage_record(Stage.WALK_FORWARD, now.isoformat(), "test", evaluated),
            ]
        for i in range(16):  # sixteen exploration contracts, nearly all lost
            at = now - timedelta(days=30 - i)
            s.add(OptionsPositionRow(version_id=losing.id, underlying="SPY", family="long_call", direction="bullish",
                                     mode="paper", structure={"exploration": True}, quantity=1, status="closed",
                                     opened_at=at - timedelta(days=3), entry_value=400.0, entry_underlying=500.0,
                                     max_loss=400.0, closed_at=at, realized_pnl=-350.0 if i % 3 else -250.0))  # fmt: skip
        ids = losing.id, quiet.id
    _promoted, demoted = await lab._promote_all(now)
    assert [(d["version_id"], d["to"]) for d in demoted] == [(ids[0], "RETIRED")]
    async with api.container.db.session() as s:
        assert (await s.get(OptionsStrategyVersionRow, ids[1])).stage == "WALK_FORWARD"
        watched = {r.version_id for r in (await s.scalars(select(OptionsStrategyDecayRow))).all()}
    assert ids[0] in watched and ids[1] not in watched  # no live trades: nothing to judge, no row written


async def test_with_nothing_passed_yet_a_run_still_breeds_new_ideas_and_runs_alone(api):
    """The search never stalls waiting for a first success: a run with time left explores (wider variations of
    the best-scoring candidates and brand-new random strategies). One run at a time."""
    import asyncio

    lab = api.container.options_lab
    report = await lab.research(closes=CLOSES, budget_seconds=240, max_evaluations=2)
    gen = report["generation"]
    assert gen is not None and gen["children"] > 0 and "immigrant" in gen["origins"]
    async with api.container.db.session() as s:
        versions = (await s.scalars(select(OptionsStrategyVersionRow))).all()
    new = [v for v in versions if v.origin == "immigrant"]
    assert new and all(v.strategy_key.startswith("new-") and v.stage == "EXTRACTED" for v in new)
    if gen["kind"] == "exploration":  # nothing reached VALIDATION in this short run
        assert any(v.origin == "exploration" for v in versions) or len(new) == gen["children"]
    first, second = await asyncio.gather(
        lab.research(closes=CLOSES, budget_seconds=30, max_evaluations=1),
        lab.research(closes=CLOSES, budget_seconds=30, max_evaluations=1),
    )
    assert [r.get("skipped") is not None for r in (first, second)].count(True) == 1


async def test_options_research_runs_in_the_closed_market_research_queue(api):
    from quantpulse.brain.research import handlers
    from quantpulse.brain.research.catalog import CATALOG

    spec = CATALOG["options_research"]
    assert (
        spec.cost == "heavy" and spec.refresh == timedelta(hours=2) and spec.timeout == timedelta(minutes=40)
    )
    brain = api.container.brain
    assert brain.options_lab is api.container.options_lab
    calls = []

    async def research(**kw):
        calls.append(kw)
        return {"evaluated": [{"version_id": 1}], "promoted": [], "demoted": [], "experiments": 0,
                "generation": {"generation": 0, "children": 12, "kind": "exploration"}, "data": {}, "seconds": 1.0,
                "label": "model-priced chains over real underlying prices"}  # fmt: skip

    brain.options_lab.research = research
    ctx = handlers.JobContext(job={"id": None, "kind": "options_research"}, brain=brain,
                              settings=api.container.settings, clock=api.container.clock,
                              ledger=brain.research.ledger, lifecycle=brain.research.lifecycle)  # fmt: skip
    out = await handlers.options_research(ctx)
    assert calls == [{"budget_seconds": handlers.OPTIONS_BUDGET.total_seconds()}]  # 25 minutes, not 3
    assert out["evaluated"] == 1 and out["generation"]["kind"] == "exploration"
