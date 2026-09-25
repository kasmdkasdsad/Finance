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
