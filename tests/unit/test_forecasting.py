import math

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student

from quantpulse.core.errors import DomainError
from quantpulse.quant import forecasting as fc
from quantpulse.quant import implied
from quantpulse.quant import volatility as vol

LONG_RUN_DAILY_VAR = 0.25**2 / 252


def simulate_garch(n, alpha=0.08, beta=0.90, nu=6.0, seed=0):
    rng = np.random.default_rng(seed)
    omega = LONG_RUN_DAILY_VAR * (1 - alpha - beta)
    z = student.rvs(nu, size=n, random_state=rng) / math.sqrt(nu / (nu - 2))
    var, r = LONG_RUN_DAILY_VAR, np.empty(n)
    for i in range(n):
        r[i] = math.sqrt(var) * z[i]
        var = omega + alpha * r[i] ** 2 + beta * var
    return r


def prices_from(returns, start=100.0):
    return start * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))


# ----------------------------------------------------------------------------- volatility
@pytest.mark.parametrize("seed", [0, 1])
def test_garch_recovers_simulated_parameters(seed):
    fit = vol.fit_garch(simulate_garch(3000, seed=seed))
    assert fit.alpha == pytest.approx(0.08, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.04)
    assert 4.0 < fit.nu < 10.0
    assert math.sqrt(fit.long_run_variance * 252) == pytest.approx(0.25, rel=0.2)
    assert fit.persistence < 1 and fit.half_life_days > 0


def test_garch_likelihood_matches_scipy_student_t():
    r = simulate_garch(800, seed=4)
    fit = vol.fit_garch(r, demean=False)
    var = vol.variance_path(r, fit.omega, fit.alpha, fit.beta, float(r.var(ddof=1)))[:-1]
    scale = np.sqrt(var * (fit.nu - 2) / fit.nu)
    expected = student.logpdf(r / scale, fit.nu).sum() - np.log(scale).sum()
    assert fit.log_likelihood == pytest.approx(expected, rel=1e-9)


def test_variance_term_structure_mean_reverts():
    fit = vol.fit_garch(simulate_garch(2000, seed=2))
    ts = fit.variance_term_structure(2000)
    assert ts[0] == pytest.approx(fit.next_variance)
    assert ts[-1] == pytest.approx(fit.long_run_variance, rel=1e-3)
    assert fit.horizon_volatility(21) == pytest.approx(math.sqrt(ts[:21].sum()))


def test_short_history_falls_back_to_ewma_and_ewma_recursion():
    r = simulate_garch(120, seed=3)
    with pytest.raises(DomainError, match="at least 250"):
        vol.fit_garch(r)
    fit = vol.fit_best(r)
    assert fit.method == "ewma" and fit.persistence == pytest.approx(1.0)
    path = vol.ewma_variance(r)
    manual = path[0]
    for x in r:
        manual = 0.94 * manual + 0.06 * x**2
    assert path[-1] == pytest.approx(manual) == pytest.approx(fit.next_variance)


def test_range_estimators():
    closes = prices_from(simulate_garch(300, seed=5))
    assert vol.close_to_close(closes, 21) == pytest.approx(
        np.diff(np.log(closes))[-21:].std(ddof=1) * math.sqrt(252)
    )
    high, low = np.full(30, 101.0), np.full(30, 99.0)
    expected = math.sqrt(math.log(101 / 99) ** 2 / (4 * math.log(2)) * 252)
    assert vol.parkinson(high, low) == pytest.approx(expected)
    assert vol.garman_klass(np.full(30, 100.0), high, low, np.full(30, 100.0)) > 0
    with pytest.raises(DomainError):
        vol.parkinson(low, high)


# ----------------------------------------------------------------------------- forecasts
def _constant_vol_fit(daily_sd=0.02, n=20000):
    z = np.random.default_rng(1).standard_normal(n)
    return vol.GarchFit(
        mu=0.0,
        omega=daily_sd**2,
        alpha=0.0,
        beta=0.0,
        nu=math.inf,
        log_likelihood=0.0,
        n_obs=n,
        last_variance=daily_sd**2,
        next_variance=daily_sd**2,
        std_residuals=z,
    )


def test_simulation_matches_lognormal_under_constant_gaussian_volatility():
    fit = _constant_vol_fit()
    f = fc.forecast_prices([], 100.0, [21], 0.08, n_paths=40000, fit=fit)
    h = f.horizons[0]
    sd = 0.02 * math.sqrt(21)
    assert h.expected_price == pytest.approx(100 * math.exp(0.08 * 21 / 252), rel=1e-9)
    assert h.volatility == pytest.approx(sd, rel=0.02)
    mu = math.log(h.expected_price / 100) - 0.5 * sd**2
    assert h.quantiles[0.05] == pytest.approx(100 * math.exp(mu + sd * norm.ppf(0.05)), rel=0.01)
    assert h.prob_up == pytest.approx(norm.cdf(mu / sd), abs=0.01)
    assert f.prob_above(110.0, 21) == pytest.approx(1 - norm.cdf((math.log(1.1) - mu) / sd), abs=0.01)


def test_forecast_shape_and_validation():
    closes = prices_from(simulate_garch(600, seed=6))
    f = fc.forecast_prices(closes, float(closes[-1]), [5, 21, 63], 0.07)
    assert [h.days for h in f.horizons] == [5, 21, 63] and len(f.cone[0.5]) == 63
    for h in f.horizons:
        qs = [h.quantiles[q] for q in fc.QUANTILES]
        assert qs == sorted(qs) and h.var_95 > 0 and h.expected_shortfall_95 >= h.var_95
    widths = [h.quantiles[0.95] - h.quantiles[0.05] for h in f.horizons]
    assert widths == sorted(widths)  # uncertainty grows with the horizon
    assert f.prob_above(float(closes[-1]), 21) == pytest.approx(f.horizons[1].prob_up, abs=1e-9)
    for bad in ({"horizons": []}, {"horizons": [0]}, {"horizons": [600]}):
        with pytest.raises(DomainError):
            fc.forecast_prices(closes, 100.0, bad["horizons"], 0.0)
    with pytest.raises(DomainError):
        fc.forecast_prices(closes[:30], 100.0, [5], 0.0)
    with pytest.raises(DomainError, match="no simulation"):
        f.prob_above(100.0, 10)


def test_walk_forward_forecasts_are_calibrated_on_garch_data():
    # Pool three independent histories: one fat-tailed sample path can be unlucky on its own.
    reports = [
        fc.evaluate_forecasts(prices_from(simulate_garch(2000, seed=s)), horizon=10, step=5, min_obs=500)
        for s in (0, 1, 2)
    ]
    rep = fc.CalibrationReport(horizon=10, step=5, records=[r for x in reports for r in x.records])
    assert rep.n > 750 and rep.effective_n == pytest.approx(rep.n * 0.5)
    assert rep.coverage(90) == pytest.approx(0.90, abs=0.04)
    assert rep.coverage(50) == pytest.approx(0.50, abs=0.06)
    assert rep.volatility_ratio() == pytest.approx(1.0, abs=0.1)
    assert sum(rep.pit_histogram()) == pytest.approx(1.0)
    assert max(rep.pit_histogram()) < 0.16  # roughly flat
    # zero-drift data: no directional skill beyond the base rate
    assert rep.brier_skill() == pytest.approx(0.0, abs=0.05)


def test_walk_forward_has_no_look_ahead():
    closes = prices_from(simulate_garch(900, seed=8))
    base = fc.evaluate_forecasts(closes, horizon=5, step=10, min_obs=500)
    shocked = closes.copy()
    shocked[-1] *= 3.0  # only the final close changes
    after = fc.evaluate_forecasts(shocked, horizon=5, step=10, min_obs=500)
    last_index = closes.size - 1
    for a, b in zip(base.records, after.records, strict=True):
        if a.origin + 5 < last_index:
            assert a == b


# ----------------------------------------------------------------------------- implied
def test_flat_smile_gives_black_scholes_probabilities():
    f, t, r, s = 100.0, 0.5, 0.04, 0.3
    dist = implied.risk_neutral_distribution(f, t, r, implied.smile_function([-1, 1], [s, s]), s)
    for k in (80.0, 100.0, 125.0):
        d2 = (math.log(f / k) - 0.5 * s**2 * t) / (s * math.sqrt(t))
        assert dist.prob_above(k) == pytest.approx(norm.cdf(d2), abs=2e-3)
    assert dist.mean() == pytest.approx(f, rel=5e-3)
    assert np.trapezoid(dist.pdf, dist.strikes) == pytest.approx(1.0, abs=5e-3)
    assert dist.quantile(0.5) == pytest.approx(f * math.exp(-0.5 * s**2 * t), rel=5e-3)


def test_downside_skew_raises_crash_probability():
    f, t = 100.0, 0.25
    flat = implied.risk_neutral_distribution(f, t, 0.0, implied.smile_function([-1, 1], [0.25, 0.25]), 0.25)
    skew = implied.risk_neutral_distribution(
        f, t, 0.0, implied.smile_function([-0.3, 0.0, 0.3], [0.40, 0.25, 0.18]), 0.25
    )
    assert 1 - skew.prob_above(85.0) > 1 - flat.prob_above(85.0)
    assert np.all(np.diff(skew.cdf) >= 0)


def test_implied_move():
    sd, expected_abs = implied.implied_move(0.32, 30 / 365)
    assert sd == pytest.approx(0.32 * math.sqrt(30 / 365))
    assert expected_abs == pytest.approx(sd * 0.7978845608)
    with pytest.raises(DomainError):
        implied.implied_move(0.0, 1.0)
