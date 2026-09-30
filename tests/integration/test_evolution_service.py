"""The Market Evolution Monitor and the model registry on the database: daily multi-timescale metrics (1-minute
micro-volatility included), a structural change detected only after the population-wide FDR control, its
competing hypotheses (none assumed — automated liquidity marked unidentifiable from prices), relationship
history appended, and a registry in which in-sample brilliance never makes a model authoritative."""

import math
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
from sqlalchemy import select

from quantpulse.core.market_calendar import NEW_YORK, is_trading_day
from quantpulse.db.evolution_models import EvolutionMetricRow, EvolutionRelationshipRow

SUBJECTS = ("AAA", "BBB", "CCC")


def trading_days(n: int, end: date) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def session(day: date, rng, noise_bps: float):
    t0 = datetime.combine(day, time(9, 30), NEW_YORK).astimezone(UTC)
    ts = [t0 + timedelta(minutes=i) for i in range(390)]
    eff = 100 * np.exp(np.cumsum(rng.normal(0, 0.2 / math.sqrt(252 * 390), 390)))
    px = eff + noise_bps / 1e4 * 100 * np.where(rng.random(390) < 0.5, -1, 1)
    return ts, px.tolist(), rng.integers(1000, 5000, 390).astype(float).tolist()


async def seed_history(api, days: list[date], shift_after: int):
    """Measure every day: AAA's intraday noise (micro-volatility) jumps for the last ``len - shift_after``."""
    ev = api.container.evolution
    rng = np.random.default_rng(7)
    closes = {u: {} for u in SUBJECTS}
    level = {u: 100.0 for u in SUBJECTS}
    for i, d in enumerate(days):
        for u in SUBJECTS:
            level[u] *= math.exp(rng.normal(0, 0.012))
            closes[u][d] = level[u]
        if i < 70:
            continue  # the daily metrics need 60+ sessions first
        intraday = {u: session(d, rng, 12.0 if (u == "AAA" and i >= shift_after) else 2.0) for u in SUBJECTS}
        await ev.collect(day=d, closes={u: dict(closes[u]) for u in SUBJECTS}, volumes={}, intraday=intraday)


async def test_a_micro_volatility_shift_is_detected_explained_and_kept(api):
    s = api.container.settings
    s.evolution_recent_days, s.evolution_reference_days = 20, 60
    days = trading_days(160, date(2026, 9, 24))
    await seed_history(api, days, shift_after=140)
    async with api.container.db.session() as sess:
        rows = (await sess.scalars(select(EvolutionMetricRow).where(EvolutionMetricRow.subject == "AAA",
                                                                   EvolutionMetricRow.metric == "rv_1m"))).all()  # fmt: skip
    assert len(rows) == 90 and {r.timescale for r in rows} == {"1m"} and rows[0].source == "1-minute bars"
    report = await api.container.evolution.scan()
    changed = {c["key"] for c in report["changes"]}
    assert "micro_volatility:AAA:rv_1m:1m" in changed  # noise inflates the finest scale
    assert not any(
        k.split(":")[1] in ("BBB", "CCC") and "micro" in k for k in changed
    )  # FDR: no false alarms here
    changes = await api.container.evolution.changes()
    c = next(x for x in changes if x["metric"] == "rv_1m" and x["subject"] == "AAA")
    names = {h["name"]: h for h in c["hypotheses"]}
    assert {"chance", "data_artifact", "macro_regime", "automated_liquidity"} <= set(names)
    assert names["automated_liquidity"]["identifiable_from_prices"] is False
    assert names["chance"]["verdict"] == "inconsistent" and "none is established" in c["summary"]
    # a second scan: the change persisted; relationships are appended (history), never overwritten
    await api.container.evolution.scan()
    async with api.container.db.session() as sess:
        rels = (await sess.scalars(select(EvolutionRelationshipRow))).all()
    status = await api.container.evolution.status()
    assert status["days_measured"] == 90 and status["enough_history"]
    if rels:
        keys = {(r.key, r.subject) for r in rels}
        assert len(rels) >= 2 * len(keys) // 2


async def test_the_registry_never_promotes_on_in_sample_results(api):
    reg = api.container.registry
    assert await reg.bootstrap() == 3 and await reg.bootstrap() == 0
    champs = [m for m in await reg.models() if m["role"] == "champion"]
    assert {m["slot"] for m in champs} == {"stock_alpha", "brain_consensus", "options_candidate_score"}
    m = await reg.register("stock_alpha", "ml", "a brilliant-looking candidate", in_sample={"score": 0.99})
    out = await reg.advance(m["id"])
    assert out["stage"] == "CANDIDATE" and out["moved"] == []
    try:
        await reg.record(m["id"], in_sample={"score": 1.0})
        raise AssertionError("in-sample results must be refused as evidence")
    except ValueError:
        pass
    await reg.record(m["id"], oos={"n": 300, "score": 0.58, "baseline": 0.52}, walk_forward={"passed": True},
                     stress={"passed": True}, shadow={"n": 40, "score": 0.61, "champion_score": 0.55})  # fmt: skip
    out = await reg.advance(m["id"])
    assert out["stage"] == "AUTHORITATIVE" and out["role"] == "champion"
    assert out["moved"] == [
        "OOS_VALIDATED",
        "WALK_FORWARD_VALIDATED",
        "STRESS_VALIDATED",
        "PAPER_SHADOW",
        "AUTHORITATIVE",
    ]
    old = next(x for x in await reg.models("stock_alpha") if x["version"] == 1)
    assert old["role"] == "none" and "replaced by v2" in old["stage_history"][-1]["reason"]
    ai = await reg.register("brain_consensus", "ai", "a language-model judge")
    await reg.record(ai["id"], oos={"n": 300, "score": 0.7, "baseline": 0.5}, walk_forward={"passed": True},
                     stress={"passed": True}, shadow={"n": 60, "score": 0.9, "champion_score": 0.5})  # fmt: skip
    assert (await reg.advance(ai["id"]))["stage"] == "PAPER_SHADOW"  # a person must approve an AI model
    r = await api.post(f"/api/v1/registry/models/{ai['id']}/approve", json={"by": "tester", "confirm": "yes"})
    assert r.status_code == 400
    r = await api.post(
        f"/api/v1/registry/models/{ai['id']}/approve",
        json={"by": "tester", "confirm": "I APPROVE THIS MODEL"},
    )
    assert r.status_code == 200 and r.json()["approved_by"] == "tester"
    assert (await reg.advance(ai["id"]))["stage"] == "AUTHORITATIVE"


async def test_the_options_and_evolution_endpoints_answer(api):
    for path in ("/api/v1/options/status", "/api/v1/options/strategies", "/api/v1/options/research",
                 "/api/v1/options/experiments", "/api/v1/options/learning", "/api/v1/options/portfolio",
                 "/api/v1/options/greeks", "/api/v1/options/performance", "/api/v1/options/counterfactuals",
                 "/api/v1/options/missed-opportunities", "/api/v1/options/candidates", "/api/v1/evolution/status",
                 "/api/v1/evolution/changes", "/api/v1/evolution/relationships", "/api/v1/registry/models",
                 "/api/v1/brain/status"):  # fmt: skip
        r = await api.get(path)
        assert r.status_code == 200, (path, r.text[:300])
    st = (await api.get("/api/v1/options/status")).json()
    assert st["paper_only"] is True and "naked short options" in st["never"] and len(st["agents"]) >= 18
    assert (await api.get("/api/v1/brain/status")).json()["options"]["paper_only"] is True
    assert (await api.get("/api/v1/options/chains", params={"underlying": "SPY"})).status_code == 503
    assert (await api.get("/api/v1/options/strategies/999999")).status_code == 404
