"""The options edge model as a service, on the database: a budgeted training run on model-priced chains that is
registered as the rule's challenger (never authoritative from offline evidence alone), saved and restored in a new
process, a live assessment of a candidate with the full feature vector recorded, a live record graded from closed
positions it scored, and the read-only status page."""

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select

from quantpulse.brain.options.perception import UnderlyingView
from quantpulse.core.market_calendar import is_trading_day
from quantpulse.db.models import BrainStateRow
from quantpulse.db.options_models import OptionsPositionRow, OptionsTradeCandidateRow
from quantpulse.options.lab.chains import ModelChains, close_time
from quantpulse.options.lab.features import history
from quantpulse.options.ml.features import FEATURES
from quantpulse.options.ml.service import STATE_KEY, OptionsMLService
from quantpulse.options.selection import Spec, build, evaluate
from tests.integration.conftest import _client, make_settings


def gbm(seed: int, n: int = 650, vol: float = 0.25) -> dict[date, float]:
    rng = np.random.default_rng(seed)
    days, d = [], date(2023, 6, 1)
    while len(days) < n:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    r = rng.normal(-0.5 * vol * vol / 252, vol / math.sqrt(252), n)
    return dict(zip(days, (100 * np.exp(np.cumsum(r))).tolist(), strict=True))


CLOSES = {"AAA": gbm(1), "BBB": gbm(2, vol=0.35)}


@pytest.fixture
async def ml_api(tmp_path, clock):
    settings = make_settings(tmp_path, options_universe=["AAA", "BBB"], options_ml_min_rows=150,
                             options_ml_step_days=6, options_ml_trees=25, options_ml_budget_seconds=900)  # fmt: skip
    async for client in _client(settings, clock):
        yield client


async def test_training_registers_a_challenger_saves_it_and_a_new_process_restores_it(ml_api):
    svc: OptionsMLService = ml_api.container.options_ml
    out = await svc.train(closes=CLOSES)
    assert out["rows"]["model"] >= 150 and out["model_id"] is not None
    assert out["stage"] != "AUTHORITATIVE"  # offline evidence alone never makes it decide
    assert {"ic", "rule_ic", "n"} <= set(out["oos"]) and "passed" in out["walk_forward"]
    models = await ml_api.container.registry.models("options_candidate_score")
    mine = next(m for m in models if m["id"] == out["model_id"])
    assert (
        mine["kind"] == "ml"
        and mine["role"] == "challenger"
        and mine["evidence"]["oos"]["n"] == out["oos"]["n"]
    )
    async with ml_api.container.db.session() as s:
        row = await s.get(BrainStateRow, STATE_KEY)
        assert row is not None and row.value["blob"] and row.value["model_id"] == out["model_id"]
    again = OptionsMLService(ml_api.container.settings, ml_api.container.db, ml_api.container.clock,
                             registry=ml_api.container.registry)  # fmt: skip
    assert await again.load() and again.model is not None and again.stage == mine["stage"]
    # a second run while one is running is refused, not queued
    async with svc._running:
        assert "already in progress" in (await svc.train(closes=CLOSES))["skipped"]

    # the live assessment: the same vector the training rows have, recorded with the prediction
    days = sorted(CLOSES["AAA"])
    day = days[-1]
    src = ModelChains(CLOSES)
    chain = src.chain("AAA", day, (14, 90))
    assert chain is not None
    feats = history(CLOSES["AAA"], {d: src.atm_iv("AAA", d) for d in days})
    view = UnderlyingView("AAA", close_time(day), spot=chain.underlying_price, chain=chain, features=feats[day],
                          closes=dict(CLOSES["AAA"]))  # fmt: skip
    cand = build(Spec("long_call", 20, 45, 0.5), chain.quotes, chain.underlying_price, close_time(day))[0]
    evaluate(cand, chain.underlying_price, close_time(day))
    a = svc.assess(view, cand, {"score_signed": 0.3})
    assert a is not None and len(a["x"]) == len(FEATURES) and a["authoritative"] is False
    p = a["prediction"]
    assert p["lower"] <= p["median"] <= p["upper"] and 0 <= p["p_win"] <= 1

    page = await ml_api.get("/api/v1/options/ml")
    assert page.status_code == 200
    body = page.json()
    assert body["model"]["model_id"] == out["model_id"] and "AUTHORITATIVE" in body["how_it_decides"]


async def test_the_live_record_grades_the_model_against_the_rule_on_closed_positions(ml_api):
    svc: OptionsMLService = ml_api.container.options_ml
    now = datetime(2026, 9, 1, 15, 0, tzinfo=UTC)
    rng = np.random.default_rng(5)
    async with ml_api.container.db.session() as s:
        for i in range(14):
            outcome = float(rng.normal())
            audit = {"metrics": {"expected_on_risk": float(rng.normal())},  # the rule: uninformative here
                     "verdict": {"opinions": [{"agent": "OptionsMLAgent", "verdict": "abstain", "score": 0.0,
                                               "reasons": [], "data": {"prediction": {"expected": outcome + 0.1 * float(rng.normal())},
                                                                       "x": [0.0] * len(FEATURES)}}]}}  # fmt: skip
            cand = OptionsTradeCandidateRow(cycle_key=f"brain-{i}", created_at=now, underlying="AAA", family="long_call",
                                            structure_key=f"k{i}", mode="shadow", status="shadow_opened", audit=audit)  # fmt: skip
            s.add(cand)
            await s.flush()
            s.add(OptionsPositionRow(candidate_id=cand.id, underlying="AAA", family="long_call", direction="bullish",
                                     mode="shadow", quantity=1, status="closed", opened_at=now + timedelta(days=i),
                                     closed_at=now + timedelta(days=i + 5), entry_value=500.0, entry_underlying=100.0,
                                     max_loss=500.0, realized_pnl=500.0 * outcome / 4))  # fmt: skip
    rows, record = await svc._live_rows()
    assert len(rows) == 14 and {r.grade for r in rows} == {"shadow"}
    assert record["n"] == 14 and record["score"] > 0.8 and abs(record["champion_score"]) < record["score"]
    async with ml_api.container.db.session() as s:
        assert len((await s.scalars(select(OptionsPositionRow))).all()) == 14  # read, never changed
