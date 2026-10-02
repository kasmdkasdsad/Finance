"""The options research lab on synthetic, MODEL-PRICED data (labelled as such): genomes, the backtester and its
five execution models, walk-forward, Monte Carlo, the tail lab, overfitting defences, baselines, promotion,
the research library and extraction, the population, learning, decay, counterfactuals, lessons, experiments
and the research-only bandit."""

import math
import random
from datetime import date, timedelta

import numpy as np
import pytest

from quantpulse.core.market_calendar import is_trading_day
from quantpulse.options.contracts import OptionContract
from quantpulse.options.fills import ASSUMPTIONS, ExecutionModel, leg_fill, limit_price
from quantpulse.options.lab import (
    bandit,
    baselines,
    critic,
    decay,
    discovery,
    experiments,
    learning,
    lessons,
    overfit,
    population,
    promotion,
    scoring,
    stress,
    walkforward,
)
from quantpulse.options.lab import montecarlo as mc
from quantpulse.options.lab.backtest import BacktestConfig, run
from quantpulse.options.lab.chains import ModelChains, close_time, standard_expirations
from quantpulse.options.lab.counterfactuals import evaluate as cf_evaluate
from quantpulse.options.lab.counterfactuals import verdict as cf_verdict
from quantpulse.options.lab.extraction import extract
from quantpulse.options.lab.features import history, signal, trend_regime
from quantpulse.options.lab.genome import Genome, from_dict
from quantpulse.options.lab.research import SEED, claim_test, sign_test
from quantpulse.options.selection import Spec, build, evaluate, rank
from quantpulse.options.structures import bull_put_spread


def gbm(
    seed: int, n: int = 700, start: date = date(2022, 1, 3), mu: float = 0.10, vol: float = 0.25
) -> dict[date, float]:
    rng = np.random.default_rng(seed)
    days, d = [], start
    while len(days) < n:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    r = rng.normal(mu / 252 - 0.5 * vol * vol / 252, vol / math.sqrt(252), n)
    return dict(zip(days, (100 * np.exp(np.cumsum(r))).tolist(), strict=True))


@pytest.fixture(scope="module")
def world():
    closes = {"AAA": gbm(1), "BBB": gbm(2, mu=0.02, vol=0.35)}
    src = ModelChains(closes)
    feats = {u: history(c, {d: src.atm_iv(u, d) for d in c}) for u, c in closes.items()}
    days = sorted(closes["AAA"])
    return closes, src, feats, days


PUT_SPREAD = Genome("bull_put_spread", "bullish", entry_signal="trend_up", dte_min=30, dte_max=45, delta_target=0.25,
                    width_pct=0.05, take_profit=0.5, stop_loss=2.0)  # fmt: skip
LONG_CALL = Genome("long_call", "bullish", entry_signal="always", dte_min=30, dte_max=60, delta_target=0.5)


# --------------------------------------------------------------------------- genome
def test_genomes_are_explicit_hashed_and_immutable():
    assert PUT_SPREAD.valid and len(PUT_SPREAD.hash) == 32 and PUT_SPREAD.parameter_count >= 5
    assert from_dict(PUT_SPREAD.canonical()).hash == PUT_SPREAD.hash
    assert Genome("bull_put_spread", "bullish").problems() == ["bull_put_spread needs an explicit width_pct"]
    assert "DTE below 1" in " ".join(
        Genome("long_call", "bullish", dte_min=0, dte_max=1, exit_dte=0).problems()
    )
    assert Genome(
        "long_call", "bullish", dte_min=0, dte_max=1, exit_dte=0, zero_dte=True
    ).valid  # 0DTE: its own family
    assert "unknown structure family" in " ".join(Genome("stock", "bullish").problems())
    rng = random.Random(1)
    kids = {PUT_SPREAD.mutate(rng).hash for _ in range(20)}
    assert PUT_SPREAD.hash not in kids and all(len(k) == 32 for k in kids)
    child = PUT_SPREAD.crossover(LONG_CALL, random.Random(2))
    assert child.valid
    assert "entry: trend up" in PUT_SPREAD.describe()


# --------------------------------------------------------------------------- fills and selection
def test_fill_models_are_ordered_and_limit_prices_are_realistic():
    buys = [leg_fill(1, 1.0, 1.2, m) for m in ExecutionModel]
    assert buys == sorted(buys) and buys[0] == pytest.approx(
        1.1
    )  # optimistic = mid … stress = beyond the ask
    assert leg_fill(-1, 1.0, 1.2, ExecutionModel.PESSIMISTIC) == pytest.approx(1.0)
    assert ASSUMPTIONS[ExecutionModel.MIDPOINT].label.startswith("mid fills")
    from quantpulse.options.quotes import Greeks, OptionQuote

    c1, c2 = (
        OptionContract("AAA", date(2026, 10, 16), "put", 95),
        OptionContract("AAA", date(2026, 10, 16), "put", 90),
    )
    q1 = OptionQuote(c1, 2.0, 2.2, None, "opra", "t", greeks=Greeks(delta=-0.3))
    q2 = OptionQuote(c2, 0.9, 1.1, None, "opra", "t", greeks=Greeks(delta=-0.15))
    assert limit_price(-1, [(-1, 1, q1), (1, 1, q2)]) == pytest.approx(
        -2.05 + 1.05
    )  # a credit, a quarter-spread worse


def test_selection_builds_every_family_and_ranks_on_the_whole_payoff(world):
    _closes, src, _, days = world
    d = days[300]
    chain = src.chain("AAA", d)
    spot = chain.underlying_price
    now = close_time(d)
    for fam, extra in (("long_call", {}), ("long_put", {}), ("bull_call_spread", {"width_pct": 0.05}),
                       ("bear_put_spread", {"width_pct": 0.05}), ("bull_put_spread", {"width_pct": 0.05}),
                       ("bear_call_spread", {"width_pct": 0.05}), ("long_straddle", {}), ("long_strangle", {"wing_pct": 0.05}),
                       ("iron_condor", {"width_pct": 0.05}), ("call_butterfly", {"width_pct": 0.05}),
                       ("covered_call", {}), ("cash_secured_put", {})):  # fmt: skip
        cands = build(Spec(fam, 20, 60, 0.3, **extra), chain.quotes, spot, now)
        assert cands, fam
        for c in cands:
            assert c.structure.family == fam and 20 <= c.dte <= 60
    cands = [
        evaluate(c, spot, now)
        for c in build(Spec("bull_put_spread", 20, 60, 0.25, 0.05), chain.quotes, spot, now)
    ]
    m = cands[0].metrics
    assert m["max_loss"] > 0 and m["capital"] > 0 and 0 < m["pop"] < 1 and m["distribution"] == "market"
    assert m["fill_debit"] >= m["mid_debit"]  # paying something to trade
    ranked = rank(cands)
    assert all(r.score is not None for r in ranked)


# --------------------------------------------------------------------------- the backtester
def test_the_backtester_runs_every_execution_model_and_costs_only_hurt(world):
    _closes, src, feats, days = world
    results = {}
    for m in ExecutionModel:
        res = run(PUT_SPREAD, src, feats, BacktestConfig(days[260], days[-1], ("AAA", "BBB"), model=m))
        results[m] = res
        assert res.grade == "model" and "MODEL-PRICED" in res.label
        assert m.value in res.summary()["execution_model"]
    opt = results[ExecutionModel.OPTIMISTIC].metrics["expectancy"]
    pes = results[ExecutionModel.PESSIMISTIC].metrics["expectancy"]
    assert opt is not None and pes is not None and pes <= opt
    t = results[ExecutionModel.REALISTIC].trades[0]
    assert t["pnl"] >= -t["max_loss"] - 1  # a defined-risk trade never loses more than its maximum
    assert {"regime", "iv_regime", "features", "attribution", "exit_reason", "dte_entry"} <= set(t)
    assert all(isinstance(v, float | int | str | list | dict | type(None)) for v in t.values())
    assert res.metrics["pnl_attribution"]["delta"] != 0


def test_entries_are_point_in_time_and_never_hold_through_expiration(world):
    _closes, src, feats, days = world
    res = run(LONG_CALL, src, feats, BacktestConfig(days[260], days[-1], ("AAA",)))
    for t in res.trades:
        assert t["dte_exit"] >= 0 and t["exit_reason"] in (
            "exit_dte",
            "max_hold",
            "take_profit",
            "stop_loss",
            "expiration",
        )
        f = t["features"]
        assert f["day"] == t["entry_date"]  # decided on the entry day's own features
    assert not [t for t in res.trades if t["exit_reason"] == "expiration"]  # exit_dte 7 closes them first


# --------------------------------------------------------------------------- validation
def test_walk_forward_rolls_and_reports_out_of_sample(world):
    _closes, src, feats, days = world
    wins = walkforward.windows(
        days[0], days[-1], train_days=240, validate_days=90, test_days=90, step_days=120
    )
    assert len(wins) >= 2 and wins[0].test[1] < wins[1].test[1]
    for w in wins:
        assert w.train[1] < w.validate[0] <= w.validate[1] < w.test[0]
    wf = walkforward.walk_forward(PUT_SPREAD, src, feats, ("AAA", "BBB"), wins, n_variants=2, min_trades=10)
    assert wf["variants_tried"] == 3 and len(wf["windows"]) == len(wins)
    assert "oos" in wf and isinstance(wf["passed"], bool) and 0 <= wf["parameter_stability"] <= 1


def test_monte_carlo_distributions_and_shocks(world):
    _closes, src, feats, days = world
    res = run(LONG_CALL, src, feats, BacktestConfig(days[260], days[-1], ("AAA", "BBB")))
    sim = mc.simulate(res.trades, paths=500)
    assert set(sim["scenarios"]) == set(mc.SCENARIOS)
    boot = sim["scenarios"]["bootstrap"]
    assert boot["final_pnl"]["p5"] <= boot["final_pnl"]["p50"] <= boot["final_pnl"]["p95"]
    assert sim["scenarios"]["gap_shock"]["final_pnl"]["p50"] <= boot["final_pnl"]["p50"]
    assert 0 <= sim["worst_risk_of_ruin"] <= 1
    assert mc.simulate(res.trades[:3])["scenarios"] == {}


def test_the_tail_lab_never_breaches_a_defined_risk_maximum(world):
    _closes, src, feats, days = world
    d = days[300]
    chain = src.chain("AAA", d)
    c = build(
        Spec("bull_put_spread", 20, 60, 0.25, 0.05), chain.quotes, chain.underlying_price, close_time(d)
    )[0]
    t = stress.tail(c.structure, chain.underlying_price, close_time(d), 0.3)
    assert set(t["scenarios"]) >= {
        "crash",
        "vol_spike",
        "iv_crush",
        "liquidity_collapse",
        "gap_through_strike",
    }
    assert not t["breaches_max_loss"] and t["scenarios"]["crash"] < 0
    book = stress.correlated_failure([(c.structure, 2, chain.underlying_price, 0.3)] * 3, close_time(d))
    assert book["total"] < 0 and len(book["positions"]) == 3
    res = run(PUT_SPREAD, src, feats, BacktestConfig(days[260], days[-1], ("AAA", "BBB")))
    assert stress.strategy_tail(res.trades, 100_000)["passed"]


def test_overfitting_defences():
    assert overfit.benjamini_hochberg([0.001, 0.01, 0.2, 0.9], 0.10) == [True, True, False, False]
    assert overfit.expected_max_sharpe(1000) > overfit.expected_max_sharpe(10) > 0
    many = overfit.deflated_sharpe(0.1, 250, 1000)
    few = overfit.deflated_sharpe(0.1, 250, 1)
    assert many is not None and few is not None and many < few
    bad = overfit.overfit_risk(parameter_count=12, trades=15, train_ror=0.3, test_ror=-0.05, parameter_stability=0.2,
                               pnl_by_symbol={"A": 900, "B": 10}, pnl_by_regime={"X": 1000}, sharpe_annual=4.0,
                               win_rate=0.95, ror_by_model={"OPTIMISTIC": 0.1, "REALISTIC": -0.02, "PESSIMISTIC": -0.05},
                               variants_tried=200, deflated=0.3)  # fmt: skip
    assert bad["promote_blocked"] and bad["score"] > 0.9 and len(bad["warnings"]) >= 7
    good = overfit.overfit_risk(parameter_count=5, trades=200, train_ror=0.05, test_ror=0.04, parameter_stability=0.9,
                                pnl_by_symbol={"A": 500, "B": 450, "C": 400}, pnl_by_regime={"X": 500, "Y": 400},
                                sharpe_annual=1.2, win_rate=0.6, ror_by_model={"REALISTIC": 0.04, "PESSIMISTIC": 0.02},
                                variants_tried=5, deflated=0.97)  # fmt: skip
    assert not good["promote_blocked"] and good["score"] == 0


def test_baselines_and_the_critic(world):
    closes, src, feats, days = world
    cfg = BacktestConfig(days[260], days[-1], ("AAA", "BBB"))
    res = run(PUT_SPREAD, src, feats, cfg)
    cmp = baselines.compare(
        PUT_SPREAD, res.metrics.get("expectancy_on_risk"), src, feats, closes, cfg, random_seeds=(1, 2, 3)
    )
    assert cmp["buy_and_hold_return"] is not None and cmp["no_trade"] == 0.0
    assert cmp["random"]["seeds"] >= 1 and isinstance(cmp["beats_baselines"], bool)
    assert baselines.random_control(PUT_SPREAD, 3).entry_signal == "random"
    report = critic.critique(PUT_SPREAD, res, src, feats, tail_breach=False)
    assert set(report["attacks"]) >= {"costs", "wider_spreads", "look_ahead", "one_ticker", "one_period", "small_sample",
                                      "stress_periods", "information"}  # fmt: skip
    assert "not covered by the data" in report["attacks"]["stress_periods"]["detail"]["2008 crisis"]
    assert "survivorship" in report["attacks"]["information"]["detail"]


# --------------------------------------------------------------------------- promotion and scoring
def test_no_stage_can_be_skipped_and_proven_needs_paper_evidence():
    S = promotion.Stage
    ev = promotion.Evidence(family="bull_put_spread", backtests=1, backtest_trades=40,
                            ror_by_model={"REALISTIC": 0.05, "PESSIMISTIC": 0.02})  # fmt: skip
    stage, _ = promotion.advance(S.RESEARCH, ev)
    assert stage == S.EXTRACTED  # one step, even though later gates would pass too
    stage, _ = promotion.advance(stage, ev)
    assert stage == S.BACKTESTING
    stage, _ = promotion.advance(stage, ev)
    assert stage == S.VALIDATION
    stage, why = promotion.advance(stage, ev)
    assert stage == S.VALIDATION and any("validation period" in w for w in why)
    only_flattering = promotion.Evidence(
        backtest_trades=40, ror_by_model={"OPTIMISTIC": 0.2, "REALISTIC": -0.01}
    )
    assert promotion.gate(S.VALIDATION, only_flattering)
    shadow = promotion.Evidence(family="long_straddle", shadow_trades=30, shadow_sessions=30, shadow_ror=0.1)
    assert any("person's approval" in r for r in promotion.gate(S.PAPER_ACTIVE, shadow))
    naked = promotion.Evidence(family="naked_put", shadow_trades=30, shadow_sessions=30, shadow_ror=0.1)
    assert any("never executable" in r for r in promotion.gate(S.PAPER_ACTIVE, naked))
    assert promotion.gate(S.PROVEN, promotion.Evidence(paper_trades=10, paper_ror=0.2, paper_t=3))
    assert promotion.demote_for(S.PAPER_ACTIVE, promotion.Evidence(decay_status="BROKEN"))[0] == S.RETIRED
    assert (
        promotion.demote_for(S.PAPER_ACTIVE, promotion.Evidence(decay_status="DEGRADING"))[0]
        == S.PAPER_SHADOW
    )
    sc = scoring.score(metrics={"expectancy_on_risk": 0.06, "sharpe": 1.2, "max_drawdown": -0.05, "trades": 150},
                       walkforward={"oos": {"expectancy_on_risk": 0.04}},
                       montecarlo={"worst_risk_of_ruin": 0.0, "scenarios": {"bootstrap": {"final_pnl": {"p5": 100}}}},
                       overfit={"score": 0.1}, regimes={"A": 0.1, "B": 0.05}, grade="model")  # fmt: skip
    assert (
        sc["eligible"]
        and sc["dimensions"]["data_quality"]["grade"] == "ok"
        and sc["note"] == "model-priced evidence"
    )


# --------------------------------------------------------------------------- research, extraction, population
def test_every_seed_source_extracts_with_its_assumptions_recorded():
    for s in SEED:
        e = extract(s.rules_text)
        assert e.status == "EXTRACTED" and e.genome is not None, s.key
        assert e.genome.valid and (e.assumed or e.stated)
        assert s.quality in ("ACADEMIC", "EXCHANGE", "PROFESSIONAL_RESEARCH", "SECONDARY")
        assert 0 < s.evidence_score() <= 1
    tasty = next(s for s in SEED if s.key.startswith("tasty"))
    academic = next(s for s in SEED if s.key.startswith("jegadeesh"))
    assert tasty.evidence_score() < academic.evidence_score()  # popularity is not evidence


def test_extraction_refuses_vagueness_and_undefined_risk():
    assert extract("buy stuff when it goes up").status == "NEEDS_SPECIFICATION"
    assert extract("long call").status == "NEEDS_SPECIFICATION"  # directional with no entry condition
    assert extract("sell naked puts every month", allow_substitution=False).status == "REFUSED_UNDEFINED_RISK"
    e = extract("sell short strangles at 16 delta 45 DTE, manage winners at 50%")
    assert e.genome.family == "iron_condor" and "iron condor" in e.substitution
    e = extract(
        "Sell put credit spread when IV rank above 60, 20 delta, 5% wide, 30-45 DTE, take profit at 50%, stop at 2x"
    )
    g = e.genome
    assert (g.family, g.iv_rank_min, g.delta_target, g.width_pct, g.dte_min, g.dte_max, g.take_profit, g.stop_loss) == (
        "bull_put_spread", 60.0, 0.2, 0.05, 30, 45, 0.5, 2.0)  # fmt: skip
    assert e.confidence > 0.7 and "exit_dte" in e.assumed


def test_claims_are_tested_with_effect_size_and_regimes():
    rng = np.random.default_rng(3)
    trades = []
    for i in range(200):
        ivr = float(rng.uniform(0, 100))
        pnl = (20 if ivr > 50 else -10) + float(rng.normal(0, 15))
        trades.append(
            {"pnl": pnl, "max_loss": 100, "features": {"iv_rank": ivr}, "regime": "A" if i % 2 else "B"}
        )
    ct = claim_test(trades, "iv_rank", 50)
    assert (
        ct.verdict == "SUPPORTED" and ct.effect > 0 and ct.ci[0] > 0 and set(ct.within_regime) == {"A", "B"}
    )
    flat = [{**t, "pnl": float(rng.normal(0, 15))} for t in trades]
    assert claim_test(flat, "iv_rank", 50).verdict in ("INCONCLUSIVE", "NOT_REPRODUCED")
    assert sign_test([0.02] * 30 + [0.01] * 10)["verdict"] == "SUPPORTED"


def test_generations_are_tracked_and_never_overwrite_parents():
    g0 = population.generation0()
    assert len(g0) >= 11 and any(c.origin == "extraction" for c in g0) and any(c.origin == "seed" for c in g0)
    assert len({c.genome.hash for c in g0}) == len(g0)
    members = [population.Member(f"s{i}", i, c.genome, "VALIDATION", score=0.1 * i,
                                 regimes={"TRENDING_UP": 0.1, "PANIC": -0.2}, correlation=0.8,
                                 strengths=(("iv",) if i % 2 else ("trend",)))
               for i, c in enumerate(g0[:6])]  # fmt: skip
    before = [m.genome.hash for m in members]
    for gen in (1, 2, 3, 4, 5):
        kids = population.next_generation(members, gen, budget=5, seed=1)
        assert len(kids) <= 5 and all(k.generation == gen and k.parents for k in kids)
        assert not {k.genome.hash for k in kids} & set(before)
    assert [m.genome.hash for m in members] == before
    assert population.novelty(g0[0].genome, [g0[0].genome]) == 0.0


# --------------------------------------------------------------------------- learning
def test_shrinkage_keeps_small_samples_uncertain_and_recency_bounded():
    few = learning.shrunk_weight([0.5, 0.6, 0.4])
    many = learning.shrunk_weight([0.05] * 300)
    assert few.mean < 0.5 and many.mean == pytest.approx(0.05, abs=0.01) and many.sd < few.sd
    w = learning.recency_weights([0, 180, 3650])
    assert w[0] == 1.0 and w[1] == pytest.approx(0.5) and w[2] == 0.25  # old evidence never erased
    cal = learning.calibration([0.7] * 100, [True] * 70 + [False] * 30)
    assert cal["well_calibrated"] and cal["brier"] == pytest.approx(0.21, abs=0.01)
    conf = learning.confusion([{"predicted_profit": True, "profit": False, "family": "x"}], "family")
    assert conf["x"]["FP"] == 1
    trades = [{"iv_regime": "HIGH_IV", "family": f, "pnl": p, "max_loss": 100}
              for f, p in [("bull_put_spread", 10)] * 10 + [("long_call", -10)] * 10]  # fmt: skip
    meta = learning.meta_learn(trades)
    assert "bull_put_spread has done better than long_call" in meta["HIGH_IV"]["learned"]
    bias = learning.structure_bias([{"chosen": "long_call", "best_alternative": "bull_call_spread", "chosen_pnl": -1,
                                     "best_alternative_pnl": 1}] * 12)  # fmt: skip
    assert bias and "structure-selection bias" in bias["lesson"]
    fresh = learning.relevance(created=date(2026, 9, 1), today=date(2026, 9, 29), regime_then="A", regime_now="A",
                               sample_size=100, stability=0.9, agrees_now=True)  # fmt: skip
    old = learning.relevance(created=date(2010, 1, 1), today=date(2026, 9, 29), regime_then="A", regime_now="B",
                             sample_size=100, stability=0.9, agrees_now=False)  # fmt: skip
    assert fresh > old > 0


def test_decay_needs_evidence_before_it_retires_anything():
    exp = decay.Expectation(mean=0.05, sd=0.3)
    assert decay.assess([-0.5, -0.5, -0.5], exp)["status"] == "WATCH"  # three losses are not decay
    rng = np.random.default_rng(1)
    healthy = decay.assess(list(rng.normal(0.05, 0.3, 60)), exp)
    assert healthy["status"] in ("HEALTHY", "WATCH")
    broken = decay.assess(list(rng.normal(0.05, 0.3, 40)) + list(rng.normal(-0.4, 0.2, 30)), exp)
    assert broken["status"] == "BROKEN"


def test_counterfactuals_separate_direction_from_structure(world):
    _closes, src, feats, days = world
    res = run(LONG_CALL, src, feats, BacktestConfig(days[260], days[-1], ("AAA",)))
    t = res.trades[0]
    cfs = cf_evaluate(t, src)
    labels = {c["alternative"] for c in cfs}
    assert {"stock", "no_trade", "longer_expiration", "call_spread"} <= labels
    assert all(c["data_source"] == "model" for c in cfs)
    v = cf_verdict(t, cfs)
    assert set(v) == {
        "direction_correct",
        "structure_correct",
        "best_alternative",
        "best_alternative_on_risk",
        "lesson",
    }


def test_critique_lessons_and_missed_opportunities():
    trade = {"underlying": "AAA", "family": "long_call", "direction": "bullish", "pnl": -120, "underlying_return": 0.04,
             "attribution": {"delta": 150, "gamma": 10, "theta": -90, "vega": -200, "execution": -8, "fees": -1},
             "features": {"iv_rank": 90, "event_days": None}}  # fmt: skip
    c = lessons.critique(
        trade, counterfactual={"structure_correct": False, "best_alternative": "call_spread"}
    )
    assert "direction" in c["assumptions_held"] and "volatility" in c["assumptions_failed"]
    assert c["fault"] == "implementation" and any("call_spread" in x for x in c["should_change"])
    ls = lessons.lessons_from([{**c, "context": {"family": "long_call", "iv_regime": "HIGH_IV", "exit_date": "2026-09-01"}}] * 3,
                              today=date(2026, 9, 29))  # fmt: skip
    assert len(ls) >= 1 and ls[0]["sample_size"] == 3 and ls[0]["status"] == "candidate"
    assert (
        lessons.lessons_from([{**c, "context": {}}], today=date(2026, 9, 29)) == []
    )  # one trade teaches nothing
    assert (
        lessons.classify_missed(50, rejected_for="stale quote", would_have_passed_risk=False)
        == "GOOD_REJECTION"
    )
    assert (
        lessons.classify_missed(50, rejected_for="low strategy confidence", would_have_passed_risk=True)
        == "MISSED_WINNER"
    )
    assert (
        lessons.classify_missed(-50, rejected_for="x", would_have_passed_risk=True)
        == "CORRECTLY_AVOIDED_LOSER"
    )
    assert lessons.classify_missed(None, rejected_for="x", would_have_passed_risk=None) == "INSUFFICIENT_DATA"


def test_experiments_are_competing_hypotheses_ranked_by_information():
    props = experiments.from_failure(LONG_CALL, "low_iv")
    texts = [p.hypothesis for p in props]
    assert len(props) == 5 and texts[0].startswith("H1") and all(p.child.valid for p in props)
    assert all(p.parent_hash == LONG_CALL.hash and p.child.hash != LONG_CALL.hash for p in props)
    uncertain = experiments.information_value(n=10, sd=0.3, mean=0.05, added=20)
    known = experiments.information_value(n=10_000, sd=0.3, mean=0.001, added=20, tested_kinds=("parameter",))
    assert uncertain > known
    assert (
        experiments.decide({"expectancy_on_risk": 0.02}, {"expectancy_on_risk": 0.08, "trades": 30})[0]
        == "PASSED"
    )
    assert (
        experiments.decide({"expectancy_on_risk": 0.02}, {"expectancy_on_risk": 0.08, "trades": 5})[0]
        == "INCONCLUSIVE"
    )


def test_the_bandit_is_research_only():
    assert bandit.EXECUTION_ENABLED is False
    rng = np.random.default_rng(0)
    log = [{"context": [1.0, float(x)], "arm": "a" if rng.random() < 0.5 else "b",
            "reward": 1.0 if x > 0 else -1.0} for x in rng.normal(0, 1, 300)]  # fmt: skip
    out = bandit.replay(log, ["a", "b"], 2)
    assert (
        out["execution_enabled"] is False
        and out["matched"] > 0
        and "never selects a live trade" in out["note"]
    )


def test_analogues_features_and_the_graph():
    rng = np.random.default_rng(2)
    hist = []
    for i in range(120):
        ivr = float(rng.uniform(0, 100))
        hist.append({"family": "bull_put_spread" if i % 2 else "long_call", "entry_date": f"2025-{1 + i % 12:02d}-01",
                     "pnl": (15 if ivr > 50 else -15) * (1 if i % 2 else -1), "max_loss": 100,
                     "features": {"iv_rank": ivr, "ret20": float(rng.normal()), "noise": float(rng.normal())}})  # fmt: skip
    a = discovery.analogues({"iv_rank": 90, "ret20": 0.0}, hist, k=20)
    assert a["neighbours"] == 20 and "iv_rank" in a["dimensions"]
    assert a["by_family"]["bull_put_spread"]["mean_on_risk"] > 0
    imp = discovery.importance(hist)
    assert all(r["note"] == "association, not causation" for r in imp)
    edges = discovery.graph_edges(strategy="s1", regimes={"HIGH_IV": 0.1, "LOW_IV": -0.1}, source="carr_wu2009",
                                  trade="t1", lesson="l1", interactions_found=[("iv_rank", "ret20")])  # fmt: skip
    rels = {e["relation"] for e in edges}
    assert rels == {
        "WORKS_IN",
        "FAILS_IN",
        "DERIVED_FROM",
        "TESTED",
        "PRODUCED",
        "MODIFIES",
        "INTERACTS_WITH",
    }


def test_features_signals_and_regimes_are_past_only(world):
    _closes, _src, feats, days = world
    f = feats["AAA"][days[400]]
    assert f.sma200 is not None and f.iv is not None and f.iv_rank is not None
    early = feats["AAA"][days[10]]
    assert early.sma200 is None and signal("trend_up", early) is None  # unknown, never guessed
    assert trend_regime(f) in ("TRENDING_UP", "TRENDING_DOWN", "MEAN_REVERTING", "CALM", "PANIC")
    exps = standard_expirations(date(2026, 9, 25))
    assert (
        exps[0] == date(2026, 10, 2) and date(2026, 10, 16) in exps and all(is_trading_day(e) for e in exps)
    )


def test_a_defined_risk_structure_from_the_selection_is_never_naked(world):
    _closes, src, _feats, days = world
    d = days[350]
    chain = src.chain("BBB", d)
    for c in build(
        Spec("iron_condor", 20, 60, 0.2, 0.05), chain.quotes, chain.underlying_price, close_time(d)
    ):
        assert c.structure.naked_legs() == [] and math.isfinite(c.structure.max_loss())
    s = bull_put_spread(
        OptionContract("BBB", date(2026, 10, 16), "put", 100),
        3,
        OptionContract("BBB", date(2026, 10, 16), "put", 95),
        1,
    )
    assert s.max_loss() == 300
