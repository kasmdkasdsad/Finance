import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from quantpulse.core.errors import DomainError
from quantpulse.domain import alpha_model as am
from quantpulse.domain import features as feat

FEATURE_NAMES = ["mom_12_1", "mom_6_1", "mom_3m", "ret_1m"]


def random_panel(n_days=700, n_sym=30, seed=0) -> feat.Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    market = rng.normal(0.0003, 0.01, n_days)
    betas = rng.uniform(0.6, 1.4, n_sym)
    rets = market[:, None] * betas + rng.normal(0, 0.015, (n_days, n_sym))
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=[f"S{i:02d}" for i in range(n_sym)]
    )
    spread = np.abs(rng.normal(0, 0.01, close.shape))
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, close.shape), index=dates, columns=close.columns)
    bench = pd.Series(400 * np.exp(np.cumsum(market)), index=dates)
    return feat.Panel(
        close=close, high=close * (1 + spread), low=close * (1 - spread), volume=volume, benchmark=bench
    )


def planted_data(n_dates=800, n_sym=30, horizon=5, signal=0.02, seed=0) -> am.ModelData:
    """Feature 0 predicts the forward return with correlation ≈ signal / sqrt(signal² + 0.04²)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    symbols = [f"S{i:02d}" for i in range(n_sym)]
    F = rng.standard_normal((n_dates, n_sym, len(FEATURE_NAMES)))
    fwd = pd.DataFrame(
        signal * F[:, :, 0] + 0.04 * rng.standard_normal((n_dates, n_sym)), index=dates, columns=symbols
    )
    fwd.iloc[-horizon:] = np.nan  # not yet realised
    bench = pd.Series(rng.normal(0.0, 0.01, n_dates), index=dates)
    bench.iloc[-horizon:] = np.nan
    X = pd.DataFrame(
        F.reshape(n_dates * n_sym, -1),
        index=pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"]),
        columns=FEATURE_NAMES,
    )
    return am.ModelData(
        X=X, fwd=fwd, bench_fwd=bench, positions=pd.Series(np.arange(n_dates), index=dates), horizon=horizon
    )


CFG = am.ModelConfig(horizon=5, train_window=300, min_train=150, retrain_every=20, top_k=5)


# ----------------------------------------------------------------------------- features
def test_features_match_definitions_and_have_no_look_ahead():
    panel = random_panel()
    f = feat.compute_features(panel)
    assert set(f) == set(feat.FEATURES)
    c = panel.close
    pd.testing.assert_frame_equal(f["mom_3m"], c / c.shift(63) - 1)
    assert f["mom_12_1"].iloc[:252].isna().all().all() and f["mom_12_1"].iloc[252:].notna().all().all()
    assert f["rsi_14"].stack().dropna().between(0, 100).all()

    cut = 500
    shocked = feat.Panel(
        close=panel.close.copy(),
        high=panel.high.copy(),
        low=panel.low.copy(),
        volume=panel.volume.copy(),
        benchmark=panel.benchmark.copy(),
    )
    for frame in (shocked.close, shocked.high, shocked.low):
        frame.iloc[cut + 1 :] *= 1.7
    shocked.benchmark.iloc[cut + 1 :] *= 0.5
    g = feat.compute_features(shocked)
    for name in feat.FEATURES:
        pd.testing.assert_frame_equal(
            f[name].iloc[: cut + 1], g[name].iloc[: cut + 1], check_exact=False, rtol=1e-9
        )


def test_beta_of_a_levered_benchmark_is_its_leverage():
    panel = random_panel(n_days=400, n_sym=3)
    bench_ret = panel.benchmark.pct_change().fillna(0)
    levered = 50 * (1 + 2 * bench_ret).cumprod()
    close = panel.close.assign(LEV=levered)
    p2 = feat.Panel(
        close=close,
        high=close * 1.01,
        low=close * 0.99,
        volume=panel.volume.assign(LEV=1e6),
        benchmark=panel.benchmark,
    )
    f = feat.compute_features(p2)
    assert f["beta_252"]["LEV"].iloc[-1] == pytest.approx(2.0, rel=1e-6)
    assert f["idio_vol_63"]["LEV"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_cross_sectional_helpers():
    rng = np.random.default_rng(3)
    wide = pd.DataFrame(rng.standard_normal((50, 12)) * 5 + 3)
    z = feat.cross_sectional_z(wide)
    assert np.allclose(z.mean(axis=1), 0, atol=0.2) and z.abs().max().max() <= 3.0
    other = pd.DataFrame(rng.standard_normal((50, 12)))
    ic = feat.row_spearman(wide, other)
    assert ic.iloc[7] == pytest.approx(spearmanr(wide.iloc[7], other.iloc[7]).statistic)
    rg = feat.rank_gauss(wide)
    assert np.allclose(rg.mean(axis=1), 0, atol=1e-9)
    X = feat.feature_matrix(feat.compute_features(random_panel(n_days=320, n_sym=8)))
    assert list(X.columns) == list(feat.FEATURES) and not X.isna().any().any()
    assert X.index.get_level_values("date").min() >= random_panel(n_days=320, n_sym=8).close.index[250]


# ----------------------------------------------------------------------------- model
def test_model_finds_a_planted_signal():
    data = planted_data()
    res = am.run(data, CFG)
    assert res.oos.mean_ic > 0.3 and res.oos.t_stat > 5 and res.oos.hit_rate > 0.6
    coef = {name: c for name, (c, _) in res.importance.items()}
    assert max(coef, key=lambda k: abs(coef[k])) == "mom_12_1"
    assert res.importance["mom_12_1"][1] == 1.0  # same sign in every refit
    assert max(res.tree_importance, key=res.tree_importance.get) == "mom_12_1"
    assert set(res.models) == {"ridge", "gbm", "ensemble", "baseline"} and res.chosen == "ensemble"
    for name in ("ridge", "gbm", "ensemble"):
        assert res.models[name].oos.mean_ic > 0.25, name
        assert res.models[name].oos.n_dates == res.oos.n_dates  # judged on the same dates
    assert res.models["baseline"].oos.mean_ic < res.oos.mean_ic  # the rule only half-uses the signal
    probs = [b.probability for b in res.calibration.bins]
    assert probs == sorted(probs) and probs[-1] > 0.6 and probs[0] < 0.4
    assert res.buckets == sorted(res.buckets)  # higher predictions, higher realised returns
    m = res.backtest.metrics(252 / 5)
    assert m["strategy"]["total_return"] > m["universe"]["total_return"]
    assert [p.rank for p in res.live] == list(range(1, 31))
    assert res.live[0].prob_outperform > res.live[-1].prob_outperform
    assert res.live_date == data.dates[-1]


def test_model_does_not_invent_skill_from_noise():
    res = am.run(planted_data(signal=0.0, seed=1), CFG)
    assert abs(res.oos.mean_ic) < 0.03
    assert res.oos.t_stat is None or abs(res.oos.t_stat) < 3
    for b in res.calibration.bins:
        assert b.probability == pytest.approx(res.calibration.base_rate, abs=0.05)
    spread = [p.prob_outperform for p in res.live]
    assert max(spread) - min(spread) < 0.1


def test_walk_forward_uses_only_realised_labels():
    data = planted_data(n_dates=500, seed=2)
    base = am.walk_forward(data, CFG).predictions
    P = 400
    fwd = data.fwd.copy()
    fwd.iloc[P - data.horizon + 1 :] = np.random.default_rng(9).standard_normal(
        fwd.iloc[P - data.horizon + 1 :].shape
    )
    changed = am.ModelData(
        X=data.X, fwd=fwd, bench_fwd=data.bench_fwd, positions=data.positions, horizon=data.horizon
    )
    after = am.walk_forward(changed, CFG).predictions
    cutoff = data.dates[P]
    early = base.index.get_level_values("date") <= cutoff
    pd.testing.assert_series_equal(base[early], after[early])
    assert not np.allclose(base[~early].to_numpy(), after[~early].to_numpy())


def test_model_validation_and_short_history():
    with pytest.raises(DomainError):
        am.ModelConfig(horizon=0)
    with pytest.raises(DomainError):
        am.ModelConfig(features=("not_a_feature",))
    with pytest.raises(DomainError, match="not enough history"):
        am.walk_forward(planted_data(n_dates=150), CFG)
    assert am._pav([0.3, 0.2, 0.5], [1, 1, 1]) == pytest.approx([0.25, 0.25, 0.5])


def test_end_to_end_on_a_price_panel():
    panel = random_panel(n_days=700)
    data, _ = am.build_data(panel, horizon=21)
    res = am.run(data, am.ModelConfig(horizon=21, min_train=150, train_window=400))
    assert res.oos.n_dates > 50 and abs(res.oos.mean_ic) < 0.2
    assert len(res.backtest.strategy) == len(res.backtest.dates) == len(res.backtest.period_returns) + 1
    assert res.backtest.dates == sorted(res.backtest.dates)


# ----------------------------------------------------------------------------- earnings & industries
def test_earnings_reaction_is_known_after_the_window_and_carried():
    panel = random_panel(n_days=300, n_sym=3)
    close = panel.close.copy()
    day = close.index[200]
    close.iloc[200:, 0] *= 1.10  # a 10% jump on the reaction day
    bench = panel.benchmark
    f = feat.earnings_reaction(close, bench, {"S00": [day], "S01": []}, hold=63)
    s = f["S00"]
    assert s.iloc[:201].isna().all()  # only known after the close following the reaction day
    assert s.iloc[201:264].notna().all() and s.iloc[264:].isna().all()
    assert s.iloc[201] > 2  # a large positive surprise in volatility units
    assert f["S01"].isna().all() and f["S02"].isna().all()


def test_industry_features_and_neutralisation():
    dates = pd.bdate_range("2024-01-01", periods=3)
    wide = pd.DataFrame(
        {"A": [1.0, 2, 3], "B": [3.0, 4, 5], "C": [5.0, 6, 7], "D": [10.0, 10, 10], "E": [0.0, 0, np.nan]},
        index=dates,
    )
    sectors = pd.Series({"A": "tech", "B": "tech", "C": "tech", "D": "banks", "E": "banks"})
    n = feat.neutralise(wide, sectors)
    assert n.loc[dates[0], ["A", "B", "C"]].tolist() == [-2.0, 0.0, 2.0]  # vs the tech average
    assert n.loc[dates[0], "D"] == pytest.approx(10 - wide.loc[dates[0]].mean())  # banks too small: market
    assert np.isnan(n.loc[dates[2], "E"])
    raw = {"mom_6_1": wide, "ret_1m": wide * 2}
    sf = feat.sector_features(raw, sectors)
    assert sf["sector_mom_6_1"].loc[dates[1], "A"] == 4.0 and np.isnan(
        sf["sector_mom_6_1"].loc[dates[1], "D"]
    )
    assert sf["sector_ret_1m"].loc[dates[1], "C"] == 8.0


def test_membership_mask_and_delistings_shape_the_data():
    panel = random_panel(n_days=500, n_sym=8)
    close = panel.close.copy()
    close.iloc[400:, 7] = np.nan  # S07 is delisted after day 399
    panel = feat.Panel(
        close=close, high=panel.high, low=panel.low, volume=panel.volume, benchmark=panel.benchmark
    )
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    eligible.iloc[:300, 0] = False  # S00 joins the index on day 300
    eligible.iloc[395:, 7] = False  # S07 leaves on day 395
    sectors = pd.Series({s: "g1" if i < 4 else "g2" for i, s in enumerate(close.columns)})
    data, raw = am.build_data(panel, 21, sectors=sectors, eligible=eligible)
    rows = data.X.index
    s00_dates = rows[rows.get_level_values("symbol") == "S00"].get_level_values("date")
    assert s00_dates.min() >= close.index[300]
    assert raw["mom_3m"]["S00"].iloc[:300].isna().all()
    # S07's last eligible date still has a label: it is cashed out at its last price.
    d = close.index[390]
    assert data.fwd.loc[d, "S07"] == pytest.approx(close.iloc[399, 7] / close.iloc[390, 7] - 1)
    assert np.isnan(data.fwd.loc[close.index[396], "S07"])
    assert {"sector_mom_6_1", "sector_ret_1m"} <= set(data.feature_names)
    assert data.X_np.dtype == np.float32


def test_ensemble_combines_z_scores_and_is_scored_within_industries():
    data = planted_data(n_dates=600, seed=4)
    data.sectors = pd.Series({s: f"g{i % 3}" for i, s in enumerate(data.fwd.columns)})
    cfg = am.ModelConfig(horizon=5, train_window=300, min_train=150, retrain_every=20, gbm_retrain_every=60)
    res = am.run(data, cfg)
    ens = res.models["ensemble"].predictions.unstack()
    ridge = feat.cross_sectional_z(res.models["ridge"].predictions.unstack())
    gbm = feat.cross_sectional_z(res.models["gbm"].predictions.unstack())
    d = ens.index[10]
    assert ens.loc[d].to_numpy() == pytest.approx(((ridge.loc[d] + gbm.loc[d]) / 2).to_numpy())
    assert res.within_sector is not None and res.within_sector.mean_ic > 0.3  # the signal is stock-specific
    assert res.final.kind == "ensemble" and res.final.ridge is not None and res.final.gbm is not None
    labels = {f.label for _, f in res.fits["gbm"]}
    assert labels <= {"7 leaves × 100 trees", "31 leaves × 60 trees"}
    with pytest.raises(DomainError):
        am.ModelConfig(model_type="forest")
