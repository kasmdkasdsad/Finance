"""The strategy lab without I/O: specs and versions, backtest mechanics (lag, drift, costs), and validation
that promotes a real effect and refuses a lucky one."""

import numpy as np
import pandas as pd
import pytest

from quantpulse.brain.lab.backtest import backtest, equal_weight, metrics, portfolio_weights
from quantpulse.brain.lab.scrutiny import refute
from quantpulse.brain.lab.service import _paper_performance
from quantpulse.brain.lab.spec import TEMPLATES, StrategySpec, from_template
from quantpulse.brain.lab.validation import deflated_sharpe, validate
from quantpulse.core.errors import DomainError
from quantpulse.domain.features import Panel, compute_features


def panel_from(returns: np.ndarray, seed_price: float = 50.0) -> Panel:
    idx = pd.bdate_range(end="2026-09-24", periods=returns.shape[0])
    close = pd.DataFrame(seed_price * np.exp(np.cumsum(returns, axis=0)), index=idx,
                         columns=[f"S{i}" for i in range(returns.shape[1])])  # fmt: skip
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return Panel(close, close * 1.01, close * 0.99, vol, close.mean(axis=1))


def universe(momentum: bool, n: int = 40, t: int = 1300, seed: int = 1) -> Panel:
    rng = np.random.default_rng(seed)
    drift = rng.normal(0, 0.0015, n)
    rets = np.zeros((t, n))
    for d in range(t):
        drift = 0.999 * drift + rng.normal(0, 0.00005, n) if momentum else np.zeros(n)
        rets[d] = drift + rng.normal(0, 0.01, n) + rng.normal(0, 0.008)
    return panel_from(rets)


# ---------------------------------------------------------------------------------------------- specs
def test_specs_are_point_in_time_price_rules_and_versions_are_explicit():
    with pytest.raises(DomainError, match="unknown features"):
        StrategySpec("x", 1, "x", "", {"future_return": 1.0})
    with pytest.raises(DomainError, match="only top_n"):
        StrategySpec("x", 1, "x", "", {"mom_12_1": 1.0}, grid={"cost_bps": [0, 5]})
    spec = from_template("momentum_12_1")
    assert spec.key == "momentum_12_1@v1" and len(spec.variants()) == 6
    v2 = from_template("momentum_12_1", version=2, top_n=15, grid={})
    assert v2.key == "momentum_12_1@v2" and v2.variants() == [v2]
    assert StrategySpec.from_dict(spec.to_dict()) == spec
    for name in TEMPLATES:
        from_template(name)  # every template is a valid spec


# ---------------------------------------------------------------------------------------------- mechanics
def test_backtest_holds_the_ranked_names_with_a_lag_and_pays_costs():
    t = 600
    rng = np.random.default_rng(0)
    drifts = np.array([0.002] * 3 + [-0.001] * 3 + [0.0] * 6)
    rets = drifts + rng.normal(0, 0.005, (t, 12))
    p = panel_from(rets)
    f = compute_features(p)
    spec = StrategySpec("m", 1, "m", "", {"mom_12_1": 1.0}, top_n=3, rebalance_days=21, cost_bps=0)
    res = backtest(spec, f, p.close, p.benchmark)
    assert all(set(names) == {"S0", "S1", "S2"} for _, names in res.holdings[1:])
    winners = p.close[["S0", "S1", "S2"]].pct_change().mean(axis=1).reindex(res.returns.index)
    assert res.returns.mean() == pytest.approx(winners.mean(), rel=1e-3)
    first_signal = res.holdings[0][0]
    # signal at the close of day t, traded at the close of t+1, first return earned on t+2
    assert res.returns.index[0] == p.close.index[p.close.index.get_loc(first_signal) + 2]
    costly = backtest(
        StrategySpec("m", 1, "m", "", {"mom_12_1": 1.0}, top_n=3, rebalance_days=21, cost_bps=50),
        f,
        p.close,
        p.benchmark,
    )
    assert costly.returns.iloc[0] == pytest.approx(
        res.returns.iloc[0] - 1.0 * 50 / 1e4
    )  # first buy: turnover 1
    ew = equal_weight(p.close, p.benchmark, spec, 300, 600)
    assert metrics(ew)["days"] > 250


def test_a_signal_day_jump_is_not_captured():
    rets = np.zeros((400, 10))
    rets[:, 0] = 0.001  # S0 trends up (it will be ranked first)
    rets[300, 0] = 0.20  # a jump on day 300
    p = panel_from(rets)
    f = compute_features(p)
    spec = StrategySpec("m", 1, "m", "", {"ret_5d": 1.0}, top_n=1, rebalance_days=1, cost_bps=0)
    res = backtest(spec, f, p.close, p.benchmark, start=299, end=305)
    assert res.returns.index[0] == p.close.index[301]  # signal close 299, trade close 300, earn from 301
    assert res.returns.max() < 0.19  # the jump on day 300 was not earned by a signal from day 299


# ---------------------------------------------------------------------------------------------- validation
def test_deflated_sharpe_penalises_many_trials():
    r = np.random.default_rng(3).normal(0.0006, 0.01, 750)
    one = deflated_sharpe(r, 1, 0.0)
    many = deflated_sharpe(r, 50, 1.0)
    assert one is not None and many is not None and many < one
    assert deflated_sharpe(r[:10], 1, 0.0) is None


def test_a_real_effect_is_validated_and_luck_is_refused():
    real = universe(momentum=True)
    report = validate(
        from_template("momentum_12_1"), compute_features(real), real.close, real.benchmark, volume=real.volume
    )
    assert report["verdict"] == "validated", [g for g in report["gates"] if not g["passed"]]
    wf = report["walk_forward"]
    assert wf["oos_active_sharpe"] > 0 and wf["dsr"] > 0.9 and wf["fold_win_rate"] >= 0.6
    assert report["random_percentile"] >= 0.9 and len(wf["folds"]) >= 4
    gates = {g["gate"] for g in report["gates"]}
    assert {
        "robust to nearby parameters",
        "edge survives realistic costs",
        "capacity covers the paper book",
    } <= gates
    sc = report["scrutiny"]
    assert sc["sensitivity"]["positive_share"] == 1.0 and len(sc["sensitivity"]["neighbours"]) == 5
    assert sc["costs"]["break_even_bps"] > 2 * sc["costs"]["assumed_cost_bps"]
    assert sc["capacity"]["capacity_usd"] > 100_000 and len(sc["regimes"]) == 4
    assert sc["drawdowns"]["max_drawdown"] < 0 and "recovered" in sc["drawdowns"]
    # even a validated strategy is questioned: here, a few names carried most of the gains
    assert any("came from five names" in r for r in report["refutation"])

    noise = universe(momentum=False)
    report = validate(
        from_template("momentum_12_1"),
        compute_features(noise),
        noise.close,
        noise.benchmark,
        volume=noise.volume,
    )
    assert report["verdict"] == "rejected"
    failed = {g["gate"] for g in report["gates"] if not g["passed"]}
    assert "not explained by trying many variants (deflated Sharpe)" in failed
    # the single backtest alone looked fine: this is exactly what one attractive backtest hides
    assert report["backtest"]["sharpe"] > 0
    reasons = report["refutation"]
    assert any(r.startswith("failed: not explained by trying many variants") for r in reasons)
    assert any("loses to equal weight in" in r for r in reasons)  # and where it fails


def test_scrutiny_names_thin_edges_small_capacity_and_fragile_parameters():
    report = {
        "gates": [],
        "scrutiny": {
            "costs": {"assumed_cost_bps": 10.0, "break_even_bps": 12.0, "annual_turnover": 12.0},
            "capacity": {"capacity_usd": 40_000.0},
            "regimes": {"rising, calm": {"active_annual": 0.2, "share_of_active_return": 0.9},
                        "falling, volatile": {"active_annual": -0.05, "share_of_active_return": -0.2}},
            "drawdowns": {"max_drawdown": -0.4, "benchmark_max_drawdown": -0.2, "recovered": False},
            "concentration": {"top5_share_of_gains": 0.3, "best": []},
            "sensitivity": {"positive_share": 0.4, "neighbours": {"half the names": -0.2, "one more session of lag": 0.5}},
        },
    }  # fmt: skip
    reasons = " | ".join(refute(report, 100_000))
    assert "thin edge" in reasons and "capacity $40,000 is below" in reasons
    assert "loses to equal weight in falling, volatile" in reasons and "rising, calm markets only" in reasons
    assert "has not recovered" in reasons and "falls much further than the benchmark" in reasons
    assert "fragile: nearby parameters do not work (half the names)" in reasons
    assert "five names" not in reasons  # 30%: not concentrated


def test_thin_volume_fails_the_capacity_gate():
    real = universe(momentum=True)
    thin = real.volume * 0.0005  # ~$25k a day: a real book would move these prices
    report = validate(
        from_template("momentum_12_1"), compute_features(real), real.close, real.benchmark, volume=thin
    )
    gate = next(g for g in report["gates"] if g["gate"] == "capacity covers the paper book")
    assert not gate["passed"] and report["verdict"] == "rejected"
    assert any("capacity $" in r for r in report["refutation"])


def test_short_history_gets_no_verdict():
    p = universe(momentum=True, t=500)
    report = validate(from_template("momentum_12_1"), compute_features(p), p.close, p.benchmark)
    assert report["verdict"] == "rejected"
    assert (
        report["walk_forward"].get("insufficient")
        or not next(g for g in report["gates"] if g["gate"] == "enough out-of-sample history")["passed"]
    )


def test_paper_performance_chains_the_shadow_portfolios():
    idx = pd.bdate_range("2026-01-02", periods=30)
    close = pd.DataFrame({"A": np.linspace(100, 130, 30), "B": np.full(30, 50.0)}, index=idx)
    bench = pd.Series(np.linspace(400, 404, 30), index=idx)
    entries = [
        {"date": str(idx[0].date()), "holdings": ["A"], "prices": {"A": 100.0}, "benchmark": 400.0},
        {
            "date": str(idx[10].date()),
            "holdings": ["B"],
            "prices": {"B": 50.0},
            "benchmark": float(bench.iloc[10]),
        },
    ]
    perf = _paper_performance(entries, close, bench)
    assert perf["sessions"] == 29 and perf["entries"] == 2
    a_gain = float(close["A"].iloc[10]) / 100 - 1
    assert perf["return"] == pytest.approx(a_gain, abs=1e-4)  # B was flat afterwards
    assert perf["excess_return"] == pytest.approx((1 + a_gain) - 404 / 400, abs=1e-4)


def test_paper_tracking_measures_the_weights_that_were_validated():
    """An inverse-volatility strategy's paper record is measured at its own weights, not equal weights (the
    shadow portfolio it validated is the one tracked); entries recorded without weights stay equal-weight."""
    idx = pd.bdate_range("2026-01-02", periods=11)
    close = pd.DataFrame({"A": np.linspace(100, 120, 11), "B": np.full(11, 50.0)}, index=idx)
    bench = pd.Series(np.full(11, 400.0), index=idx)
    entry = {"date": str(idx[0].date()), "holdings": ["A", "B"], "prices": {"A": 100.0, "B": 50.0},
             "benchmark": 400.0}  # fmt: skip
    assert _paper_performance([entry], close, bench)["return"] == pytest.approx(0.10)  # 20% and 0%, equal
    weighted = {**entry, "weights": {"A": 0.25, "B": 0.75}}
    assert _paper_performance([weighted], close, bench)["return"] == pytest.approx(0.05)
    vols = pd.Series({"A": 0.40, "B": 0.20})
    w = portfolio_weights(["A", "B"], vols)
    assert w["B"] == pytest.approx(2 * w["A"]) and sum(w.values()) == pytest.approx(1.0)
    assert portfolio_weights(["A", "B"], None) == {"A": 0.5, "B": 0.5}
