"""The options edge model's parts: the SVI surface, the volatility forecast, purged cross-validation, the
triple-barrier labels, the feature vector, the model (out of sample, conformal, persistence), the new structure
families and the research-family search — and the agent, which never votes before it is authoritative."""

import math
import random
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from quantpulse.brain.options import agents as A
from quantpulse.core.market_calendar import is_trading_day
from quantpulse.options.contracts import OptionContract, expiration_time
from quantpulse.options.lab import population
from quantpulse.options.lab.chains import ModelChains, close_time
from quantpulse.options.lab.genome import RANDOM_FAMILIES, RESEARCH_FAMILIES, Genome, research_genome
from quantpulse.options.ml.cv import combinatorial, walk_forward
from quantpulse.options.ml.dataset import Dataset, Row
from quantpulse.options.ml.drift import out_of_range, psi, reference
from quantpulse.options.ml.features import FAMILY_CODES, FEATURES, candidate_features, vector
from quantpulse.options.ml.labels import ExitPolicy, triple_barrier
from quantpulse.options.ml.model import OptionsEdgeModel
from quantpulse.options.ml.surface import SVISlice, fit_slice, fit_surface
from quantpulse.options.ml.volatility import ewma_vol, har_forecast, vrp
from quantpulse.options.pricing import greeks as bs
from quantpulse.options.quotes import OptionQuote
from quantpulse.options.selection import Spec, build, evaluate
from quantpulse.options.structures import FAMILIES

NOW = datetime(2026, 1, 5, 20, 0, tzinfo=UTC)


def _days(n: int, start: date = date(2023, 1, 3)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _path(n: int, seed: int, vol: float = 0.25, drift: float = 0.0, s0: float = 100.0) -> list[float]:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift / 252 - 0.5 * vol * vol / 252, vol / math.sqrt(252), n - 1)
    return [s0, *(s0 * np.exp(np.cumsum(r))).tolist()]


# ------------------------------------------------------------------ the surface
def _svi_quotes(
    spot: float, params: dict[date, tuple[float, float, float, float, float]]
) -> list[OptionQuote]:
    out = []
    for exp, (a, b, rho, m, s) in params.items():
        years = (expiration_time(exp) - NOW).total_seconds() / (365 * 86400)
        fwd = spot * math.exp(0.04 * years)
        sl = SVISlice(exp, years, fwd, a, b, rho, m, s, 0.0, 0)
        for k in np.arange(0.6, 1.41, 0.025) * spot:
            strike = round(float(k), 1)
            for kind in ("call", "put"):
                iv = sl.vol_at_strike(strike)
                px = bs(kind, spot, strike, years, iv).price
                if px < 0.05:
                    continue
                c = OptionContract("SPY", exp, kind, strike)
                out.append(
                    OptionQuote(c, px * 0.99, px * 1.01, NOW, "opra", "test", iv=iv, underlying_price=spot)
                )
    return out


def test_the_svi_surface_recovers_the_smile_it_was_made_from():
    exps = {
        date(2026, 2, 6): (0.004, 0.06, -0.6, 0.02, 0.10),
        date(2026, 3, 20): (0.010, 0.08, -0.5, 0.03, 0.15),
    }
    quotes = _svi_quotes(500.0, exps)
    surf = fit_surface(quotes, 500.0, NOW)
    assert len(surf.slices) == 2 and surf.calendar_violations == 0
    for sl in surf.slices:
        assert sl.rmse_vol < 0.005  # half a volatility point
    f = surf.features(30)
    assert f["iv_skew"] is not None and f["iv_skew"] < 0  # puts richer: the skew slopes down
    assert f["rr"] is not None and f["rr"] < 0 and f["atm_iv"] is not None and 0.05 < f["atm_iv"] < 0.6
    # a quote 3 vol points above the surface reads as 3 points rich
    q = next(q for q in quotes if q.contract.kind == "put" and q.contract.expiration == date(2026, 2, 6))
    rich = OptionQuote(
        q.contract, q.bid, q.ask, NOW, "opra", "test", iv=(q.iv or 0) + 0.03, underlying_price=500.0
    )
    assert surf.residual(rich) == pytest.approx(0.03, abs=0.006)


def test_butterfly_arbitrage_is_detected_not_hidden():
    sane = SVISlice(date(2026, 2, 6), 0.1, 100.0, 0.004, 0.06, -0.5, 0.0, 0.1, 0.0, 0)
    assert sane.density_ok().all()
    broken = SVISlice(date(2026, 2, 6), 0.1, 100.0, -0.02, 2.0, -0.99, 0.0, 0.005, 0.0, 0)
    assert not broken.density_ok().all()


def test_a_slice_with_too_few_points_is_not_fitted():
    assert fit_slice([0.0, 0.1], [0.2, 0.21], [1.0, 1.0], 0.1) is None


# ------------------------------------------------------------------ the volatility forecast
def test_har_forecasts_the_volatility_it_was_fitted_on_and_falls_back_to_ewma():
    closes = _path(900, seed=3, vol=0.30)
    fc = har_forecast(closes, horizon=21)
    assert fc is not None and fc.method == "har" and 0.15 < fc.vol < 0.5
    short = har_forecast(closes[:100], horizon=21)
    assert short is not None and short.method == "ewma"
    assert ewma_vol(closes[:5]) is None and har_forecast(closes[:10], 5) is None
    gap, ratio = vrp(fc.vol + 0.05, fc)
    assert gap == pytest.approx(0.05) and ratio is not None and ratio > 1
    assert vrp(None, fc) == (None, None)


# ------------------------------------------------------------------ purged cross-validation
def test_walk_forward_never_trains_on_a_label_that_reaches_the_test_window():
    rng = np.random.default_rng(0)
    t0 = np.sort(rng.integers(0, 500, 2000))
    t1 = t0 + rng.integers(1, 15, 2000)
    splits = walk_forward(t0, t1, n_splits=5, embargo_days=5)
    assert len(splits) == 5
    for train, test in splits:
        lo = t0[test].min()
        assert (t1[train] < lo).all() and (t0[train] < lo).all()


def test_combinatorial_splits_purge_and_embargo_both_sides():
    rng = np.random.default_rng(1)
    t0 = np.sort(rng.integers(0, 600, 3000))
    t1 = t0 + rng.integers(1, 12, 3000)
    splits = combinatorial(t0, t1, groups=6, k_test=2, embargo_days=5)
    assert len(splits) == 15
    for train, test in splits:
        tested = set(t0[test].tolist())
        lo_hi = []
        for d in sorted(tested):
            if lo_hi and d <= lo_hi[-1][1] + 1:
                lo_hi[-1][1] = d
            else:
                lo_hi.append([d, d])
        for lo, hi in lo_hi:
            assert not ((t0[train] <= hi) & (t1[train] >= lo)).any()  # purged
            assert not ((t0[train] > hi) & (t0[train] <= hi + 5)).any()  # embargoed


# ------------------------------------------------------------------ labels
def _chain_world(path: list[float]) -> tuple[ModelChains, list[date]]:
    days = _days(len(path))
    return ModelChains({"AAA": dict(zip(days, path, strict=True))}), days


def test_the_triple_barrier_takes_profit_stops_and_times_out():
    flat = [100.0] * 80
    up = flat + [100.0 * (1.03**i) for i in range(1, 15)]
    down = flat + [100.0 * (0.97**i) for i in range(1, 15)]
    still = flat + [100.0] * 14
    for path, expected in ((up, "take_profit"), (down, "stop"), (still, "time")):
        src, days = _chain_world(path)
        entry = days[79]
        now = close_time(entry)
        chain = src.chain("AAA", entry, (20, 60))
        assert chain is not None
        cand = build(Spec("long_call", 20, 60, 0.5), chain.quotes, chain.underlying_price, now)[0]
        evaluate(cand, chain.underlying_price, now)
        lab = triple_barrier(
            cand, src, "AAA", entry, days, ExitPolicy(horizon=10, take_profit=0.5, stop_loss=0.5)
        )
        assert lab is not None and lab.barrier == expected, (expected, lab)
        assert lab.t0 == entry and lab.t1 > entry
        assert (lab.ror > 0) == (expected == "take_profit")


def test_a_label_without_a_full_horizon_or_a_barrier_is_not_made():
    src, days = _chain_world([100.0] * 85)
    entry = days[80]
    chain = src.chain("AAA", entry, (20, 60))
    assert chain is not None
    cand = build(Spec("long_call", 20, 60, 0.5), chain.quotes, chain.underlying_price, close_time(entry))[0]
    evaluate(cand, chain.underlying_price, close_time(entry))
    assert triple_barrier(cand, src, "AAA", entry, days, ExitPolicy(horizon=10)) is None


# ------------------------------------------------------------------ the new structures
NEW_FAMILIES = ("put_butterfly", "iron_butterfly", "broken_wing_butterfly", "reverse_iron_condor", "calendar")


@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_the_new_structures_are_built_defined_risk_and_never_naked(family):
    days = _days(300)
    src = ModelChains({"AAA": dict(zip(days, _path(300, seed=5), strict=True))})
    day = days[-1]
    now = close_time(day)
    chain = src.chain("AAA", day, (14, 120))
    assert chain is not None
    spec = Spec(family, 20, 45, 0.35, width_pct=0.04, wing_pct=0.05)
    cands = build(spec, chain.quotes, chain.underlying_price, now)
    assert cands, family
    for c in cands:
        st = c.structure
        assert st.family == family and st.defined_risk and not st.naked_legs()
        assert 0 < st.max_loss() < math.inf
        evaluate(c, chain.underlying_price, now)
        assert c.metrics["max_loss"] > 0 and c.metrics["expected_on_risk"] is not None
    fam = FAMILIES[family]
    assert fam.defined_risk and (family == "calendar" or not fam.default_executable)


def test_a_broken_wing_butterflys_lower_wing_is_wider():
    days = _days(300)
    src = ModelChains({"AAA": dict(zip(days, _path(300, seed=6), strict=True))})
    chain = src.chain("AAA", days[-1], (14, 60))
    assert chain is not None
    c = build(Spec("broken_wing_butterfly", 20, 45, 0.35, width_pct=0.03), chain.quotes, chain.underlying_price,
              close_time(days[-1]))[0]  # fmt: skip
    hi, mid, lo = (leg.contract.strike for leg in c.structure.legs if leg.contract)
    assert mid - lo > hi - mid


def test_research_genomes_are_valid_defined_risk_and_wait_for_a_person_to_trade():
    rng = random.Random(4)
    drawn = [research_genome(rng) for _ in range(300)]
    assert all(g.valid for g in drawn)
    assert {g.family for g in drawn} == set(RESEARCH_FAMILIES)
    assert all(FAMILIES[g.family].defined_risk and not FAMILIES[g.family].default_executable for g in drawn)


def test_one_immigrant_per_generation_comes_from_the_research_families():
    g0 = population.generation0()
    members = [population.Member(f"s{i}", i, c.genome, "BACKTESTING", score=None) for i, c in enumerate(g0)]
    kids = population.immigrants(members, [], random.Random(2), 3, 3, research=1)
    fams = [k.genome.family for k in kids]
    assert len(kids) == 3 and fams[-1] in RESEARCH_FAMILIES and all(f in RANDOM_FAMILIES for f in fams[:-1])
    # at least one immigrant always comes from the families executable by default
    only = population.immigrants(members, [], random.Random(2), 3, 1, research=1)
    assert len(only) == 1 and only[0].genome.family in RANDOM_FAMILIES


# ------------------------------------------------------------------ features
def test_the_feature_vector_is_fixed_and_honest_about_what_it_does_not_know():
    days = _days(400)
    closes = _path(400, seed=8)
    src = ModelChains({"AAA": dict(zip(days, closes, strict=True))})
    day = days[-1]
    now = close_time(day)
    chain = src.chain("AAA", day, (14, 90))
    assert chain is not None
    cand = build(
        Spec("bull_call_spread", 20, 45, 0.5, width_pct=0.04), chain.quotes, chain.underlying_price, now
    )[0]
    evaluate(cand, chain.underlying_price, now)
    surf = fit_surface(chain.quotes, chain.underlying_price, now)
    x = candidate_features(
        cand, chain.underlying_price, now, day=None, closes=closes, surface=surf, grade="model"
    )
    assert set(x) == set(FEATURES) and len(vector(x)) == len(FEATURES)
    assert math.isnan(x["iv_rank"]) and math.isnan(x["stock_score"])  # unknown stays unknown
    assert x["family"] == FAMILY_CODES.index("bull_call_spread") and x["grade_real"] == 0.0
    assert x["har_vol"] > 0 and math.isfinite(x["eor_market"]) and x["spread_cost_on_risk"] >= 0


def test_family_codes_never_move():
    # a saved model's categorical codes: appending is allowed, reordering is not
    assert FAMILY_CODES[:15] == ("long_call", "long_put", "bull_call_spread", "bear_put_spread", "bull_put_spread",
                                 "bear_call_spread", "covered_call", "cash_secured_put", "long_straddle",
                                 "long_strangle", "iron_condor", "call_butterfly", "protective_put", "collar",
                                 "calendar")  # fmt: skip
    executable = {n for n, f in FAMILIES.items() if f.defined_risk and n != "stock"}
    assert executable <= set(FAMILY_CODES)


# ------------------------------------------------------------------ the model
def _planted(n_days: int = 300, per_day: int = 8, seed: int = 0) -> Dataset:
    """Rows whose outcome depends on two features (nonlinearly), with a 'rule' that knows nothing."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for _ in range(per_day):
            x = dict.fromkeys(FEATURES, math.nan)
            a, b, noise = rng.normal(), rng.normal(), rng.normal()
            x.update(vrp=a, iv_skew=b, rv20=abs(rng.normal(0.25, 0.08)), eor_market=rng.normal(),
                     spread_cost_on_risk=abs(rng.normal(0.02, 0.01)), family=float(rng.integers(0, 6)),
                     grade_real=1.0)  # fmt: skip
            y = 0.15 * a - 0.1 * max(b, 0) + 0.05 * noise
            t0 = 738000 + d
            rows.append(Row(x, y, t0, t0 + 5, f"U{rng.integers(0, 4)}", "long_call", "recorded"))
    return Dataset.from_rows(rows)


def test_the_model_finds_a_planted_edge_out_of_sample_and_beats_a_rule_that_knows_nothing():
    ds = _planted()
    model = OptionsEdgeModel(max_iter=40, min_samples_leaf=20)
    rep = model.fit(ds)
    oos = rep["oos"]
    assert oos["n"] > 1000 and oos["ic"] > 0.5 and abs(oos["rule_ic"]) < 0.1
    assert rep["walk_forward"]["passed"] and rep["stress"]["cpcv_share_beating_rule"] >= 0.9
    assert oos["top_quintile"] > oos["all"] > oos["bottom_quintile"]
    cov = rep["conformal"]["holdout_coverage"]
    assert 0.65 <= cov <= 0.97  # about 80% of outcomes it has not seen
    top = [n for n, _ in rep["importance"][:3]]
    assert "vrp" in top
    pred = model.predict(ds.X[-5:])
    assert all(p.lower <= p.median <= p.upper and 0 <= p.p_win <= 1 for p in pred)
    assert any(n == "vrp" for n, _ in pred[0].drivers)


def test_too_little_history_is_refused_not_fitted():
    ds = _planted(n_days=3, per_day=5)
    with pytest.raises(ValueError, match="too little history"):
        OptionsEdgeModel(max_iter=10).fit(ds)


def test_a_saved_model_comes_back_only_for_the_same_features_and_library():
    ds = _planted(n_days=120, per_day=6)
    model = OptionsEdgeModel(max_iter=20, min_samples_leaf=20)
    model.fit(ds)
    again = OptionsEdgeModel.from_bytes(model.to_bytes())
    assert again is not None
    assert np.allclose([p.expected for p in again.predict(ds.X[:3], explain=False)],
                       [p.expected for p in model.predict(ds.X[:3], explain=False)])  # fmt: skip
    assert OptionsEdgeModel.from_bytes(b"not a model") is None
    model.sklearn_version = "0.0"
    assert OptionsEdgeModel.from_bytes(model.to_bytes()) is None


def test_drift_is_measured_against_the_training_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 2))
    refs = reference(X, ("a", "b"))
    assert psi(refs["a"], rng.normal(size=500)) < 0.1
    assert psi(refs["a"], rng.normal(3.0, 1.0, size=500)) > 0.25
    share, outside = out_of_range(refs, np.array([10.0, 0.0]), ("a", "b"))
    assert share == 0.5 and outside == ["a"]


# ------------------------------------------------------------------ the agent
def _ctx(ml):
    g = Genome("long_call", "bullish", dte_min=20, dte_max=45, delta_target=0.5)
    return A.CandidateContext(view=None, version={}, genome=g, cand=None, now=NOW, ml=ml)  # type: ignore[arg-type]


def _ml(stage="PAPER_SHADOW", authoritative=False, **pred):
    p = {"expected": 0.12, "median": 0.1, "lower": 0.02, "upper": 0.4, "p_win": 0.62, "out_of_range": 0.0,
         "drivers": [["vrp", 0.05]], **pred}  # fmt: skip
    return {
        "model_id": 7,
        "stage": stage,
        "authoritative": authoritative,
        "prediction": p,
        "x": [0.0] * len(FEATURES),
    }


def test_the_model_agent_abstains_until_the_registry_makes_it_authoritative():
    none = A.options_ml(_ctx(None))
    assert none.verdict == "abstain" and "no options edge model" in none.reasons[0]
    shadow = A.options_ml(_ctx(_ml()))
    assert shadow.verdict == "abstain" and "shadow model" in shadow.reasons[-1]
    assert shadow.data["x"] and shadow.data["prediction"]["expected"] == 0.12  # recorded, so it can be graded
    strange = A.options_ml(_ctx(_ml(authoritative=True, stage="AUTHORITATIVE", out_of_range=0.4)))
    assert strange.verdict == "abstain" and "outside what it learned from" in strange.reasons[-1]


def test_an_authoritative_model_votes_but_never_vetoes():
    yes = A.options_ml(_ctx(_ml(authoritative=True, stage="AUTHORITATIVE")))
    assert yes.verdict == "support" and yes.score >= 0.6
    no = A.options_ml(
        _ctx(_ml(authoritative=True, stage="AUTHORITATIVE", expected=-0.2, lower=-0.6, upper=-0.05))
    )
    assert no.verdict == "oppose" and no.score <= -0.6
    assert "OptionsMLAgent" in {name for name, *_ in A.AGENTS}
