import numpy as np
import pandas as pd
import pytest

from quantpulse.core.errors import DomainError
from quantpulse.domain import features as feat
from quantpulse.domain import regime, research

from .test_alpha_model import random_panel


def test_research_measures_ic_decay_and_spreads():
    panel = random_panel(n_days=500, n_sym=20)
    raw = feat.compute_features(panel)
    leak = feat.forward_returns(panel.close, 21)  # a "feature" that knows the future: IC must be 1
    res = research.factor_research({**raw, "mom_3m": leak}, panel.close, horizons=(1, 5, 21))
    by_name = {f.name: f for f in res.features}
    h21 = next(h for h in by_name["mom_3m"].by_horizon if h.horizon == 21)
    assert h21.mean_ic == pytest.approx(1.0) and h21.positive_share == 1.0 and h21.t_stat is None
    assert by_name["mom_3m"].quintile_returns == sorted(by_name["mom_3m"].quintile_returns)
    assert by_name["mom_3m"].spread > 0
    honest = next(h for h in by_name["vol_63"].by_horizon if h.horizon == 21)
    assert abs(honest.mean_ic) < 0.2  # random walks carry no signal
    assert res.horizons == [1, 5, 21] and res.n_symbols == 20
    corr = res.correlation
    assert np.allclose(np.diag(corr), 1.0) and np.allclose(corr, corr.T)
    with pytest.raises(DomainError):
        research.factor_research(raw, panel.close, names=["nope"])


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)))


def test_regime_labels_trend_and_stress():
    rng = np.random.default_rng(0)
    calm_up = 100 * np.exp(np.cumsum(0.0008 + 0.006 * rng.standard_normal(600)))
    up = regime.market_regime(_series(calm_up), curve_10y=0.042, curve_3m=0.045)
    assert up.label in {"Uptrend", "Volatile uptrend"} and up.above_sma200 and up.sma50_above_sma200
    assert up.curve_inverted is True and up.curve_slope_10y_3m == pytest.approx(-0.003)
    assert any("inverted" in n for n in up.notes)
    assert sum(h.n for h in up.history) == 600 - 199 - 21

    crash = np.concatenate(
        [calm_up, calm_up[-1] * np.exp(np.cumsum(-0.01 + 0.035 * rng.standard_normal(60)))]
    )
    stress = regime.market_regime(_series(crash))
    assert stress.label == "Stress" and not stress.above_sma200
    assert stress.volatility_percentile > 0.9 and stress.drawdown_52w < -0.2
    assert stress.curve_inverted is None


def test_regime_breadth_and_validation():
    panel = random_panel(n_days=300, n_sym=10)
    r = regime.market_regime(panel.benchmark, panel.close)
    assert 0 <= r.breadth_above_sma200 <= 1 and 0 <= r.breadth_above_sma50 <= 1
    with pytest.raises(DomainError, match="at least 260"):
        regime.market_regime(panel.benchmark.iloc[:100])
