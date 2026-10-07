"""The strategy generator keeps the lab trying new ideas: it climbs from what worked out of sample, combines what
research found, explores a little, never tries the same rule twice, and every idea tried raises the bar."""

import random

import pytest

from quantpulse.brain.lab import generator as g
from quantpulse.brain.lab.spec import TEMPLATES, StrategySpec, from_template
from quantpulse.brain.lab.validation import Population, validate, walk_forward
from quantpulse.domain.features import FEATURES, compute_features
from tests.unit.test_brain_lab import universe

TREND = from_template("trend_momentum")  # 3 features, the uptrend filter


def test_a_rule_is_known_by_what_it_does_not_by_its_name():
    a = StrategySpec("a", 1, "A", "", {"mom_12_1": 1.0, "vol_63": -0.5})
    b = StrategySpec(
        "b", 7, "B", "other words", {"vol_63": -0.5, "mom_12_1": 1.0}, top_n=20, grid={"top_n": [5]}
    )
    assert g.fingerprint(a) == g.fingerprint(b)  # names, versions and the searched grid do not matter
    assert g.fingerprint(a) != g.fingerprint(StrategySpec("a", 1, "A", "", {"mom_12_1": 1.0, "vol_63": -1.0}))
    assert g.fingerprint(a) != g.fingerprint(StrategySpec("a", 1, "A", "", a.signal, rebalance_days=5))


def test_mutations_change_one_thing_at_a_time():
    out = g.mutations(TREND, {"vol_63": -0.03, "mom_12_1": 0.05})
    origins = [c.origin for c in out]
    assert all(o.startswith("mutation of trend_momentum@v1: ") for o in origins)
    assert "mutation of trend_momentum@v1: mom_12_1 ×0.5" in origins
    assert "mutation of trend_momentum@v1: without mom_3m" in origins  # the weakest weight
    assert "mutation of trend_momentum@v1: adds −vol_63 (research-backed)" in origins  # the IC's sign
    assert not any("adds +mom_12_1" in o for o in origins)  # already in the signal
    assert "mutation of trend_momentum@v1: without the uptrend filter" in origins
    assert "mutation of trend_momentum@v1: inverse_vol weights" in origins
    added = next(c for c in out if "adds −vol_63" in c.origin).spec
    assert added.signal["vol_63"] == -0.5 and added.filters == TREND.filters
    assert added.id.startswith("gen-") and added.version == 1 and added.grid  # a full spec, its grid searched
    assert len({g.fingerprint(c.spec) for c in out}) == len(out)


def test_combinations_pair_research_backed_features_with_the_sign_of_their_ic():
    out = g.combinations({"mom_12_1": 0.04, "vol_63": -0.03, "rsi_14": -0.01})
    assert len(out) == 3
    first = out[0].spec  # the two strongest
    assert first.signal == {"mom_12_1": 1.0, "vol_63": -0.5}
    assert g.combinations({}) == []


def test_a_batch_is_new_unique_and_reproducible():
    templates = [from_template(t) for t in TEMPLATES]
    ev = g.Evidence(
        tried={g.fingerprint(s) for s in templates},
        leaders=[TREND, from_template("low_volatility")],
        features={"mom_12_1": 0.04, "vol_63": -0.03, "sharpe_126": 0.02},
    )
    batch = g.propose(ev, 6, seed="2026-10-07:6")
    assert len(batch) == 6
    prints = [g.fingerprint(c.spec) for c in batch]
    assert len(set(prints)) == 6 and not set(prints) & ev.tried  # nothing tried twice
    kinds = {c.origin.split(":")[0].split(" of ")[0] for c in batch}
    assert {"mutation", "combination", "exploration"} <= kinds  # climbs, combines and explores
    assert all(len(c.spec.signal) <= g.MAX_SIGNALS and set(c.spec.signal) <= set(FEATURES) for c in batch)
    assert [c.spec for c in g.propose(ev, 6, seed="2026-10-07:6")] == [c.spec for c in batch]
    ev.tried |= set(prints)
    again = g.propose(ev, 6, seed="2026-10-08:12")
    assert not {g.fingerprint(c.spec) for c in again} & ev.tried


def test_with_no_evidence_yet_it_explores():
    batch = g.propose(g.Evidence(), 4, seed="x")
    assert len(batch) == 4 and all(c.origin.startswith("exploration") for c in batch)
    assert g.propose(g.Evidence(), 0, seed="x") == []


def test_leaders_are_the_positive_out_of_sample_strategies_best_first():
    rows = [
        {"spec": from_template("momentum_12_1").to_dict(), "status": "rejected",
         "validation": {"walk_forward": {"oos_active_sharpe": 0.4}}},
        {"spec": TREND.to_dict(), "status": "paper", "validation": {"walk_forward": {"oos_active_sharpe": 0.9}}},
        {"spec": from_template("low_volatility").to_dict(), "status": "validated",
         "validation": {"walk_forward": {"oos_active_sharpe": -0.2}}},
        {"spec": from_template("dip_in_uptrend").to_dict(), "status": "proposed", "validation": {}},
    ]  # fmt: skip
    assert [s.id for s in g.leaders(rows)] == ["trend_momentum", "momentum_12_1"]


def test_only_significant_research_findings_feed_the_generator():
    learnings = [
        {"topic": "feature:mom_12_1", "status": "SUPPORTED", "statistics": {"p_value": 0.01, "effect": 0.04}},
        {"topic": "feature:vol_63", "status": "UNPROVEN", "statistics": {"p_value": 0.08, "effect": -0.03}},
        {"topic": "feature:rsi_14", "status": "INCONCLUSIVE", "statistics": {"p_value": 0.4, "effect": -0.01}},
        {"topic": "feature:mom_12_1:high_vol", "status": "SUPPORTED", "statistics": {"p_value": 0.01, "effect": 0.1}},
        {"topic": "feature:beta_252", "status": "REFUTED", "statistics": {"p_value": 0.01, "effect": 0.02}},
        {"topic": "agent:technical", "status": "SUPPORTED", "statistics": {"p_value": 0.01, "effect": 0.2}},
    ]  # fmt: skip
    assert g.research_features(learnings) == {"mom_12_1": 0.04, "vol_63": -0.03}


def test_every_strategy_tried_raises_the_bar():
    """The same out-of-sample record is less convincing after many tries: the deflated Sharpe counts every
    strategy the lab has tested, and the spread of their results."""
    real = universe(momentum=True)
    features = compute_features(real)
    spec = from_template("momentum_12_1")
    alone = walk_forward(spec, features, real.close, real.benchmark)
    crowd = walk_forward(
        spec, features, real.close, real.benchmark, population=Population(tried=200, sharpe_variance=0.5)
    )
    assert alone["trials"] == 6 and crowd["trials"] == 1200 and crowd["strategies_tried"] == 200
    assert crowd["oos_active_sharpe"] == alone["oos_active_sharpe"]  # the same record ...
    assert crowd["dsr"] < alone["dsr"]  # ... judged against far more luck
    report = validate(spec, features, real.close, real.benchmark, volume=real.volume,
                      population=Population(tried=200, sharpe_variance=0.5))  # fmt: skip
    gate = next(x for x in report["gates"] if x["gate"].startswith("not explained by trying many"))
    assert "200 strategies tried" in gate["detail"]


@pytest.mark.parametrize("n", [1, 3, 10])
def test_the_batch_size_is_respected(n):
    ev = g.Evidence(leaders=[TREND], features={"mom_12_1": 0.04, "vol_63": -0.03})
    assert len(g.propose(ev, n, seed=str(n))) == n


def test_exploration_draws_real_features():
    c = g.exploration(random.Random(1), {})
    assert set(c.spec.signal) <= set(FEATURES) and len(c.spec.signal) == 2
