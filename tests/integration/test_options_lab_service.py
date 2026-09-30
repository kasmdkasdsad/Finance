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
