import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import norm

from quantpulse.core.errors import DomainError
from quantpulse.quant import optimization as opt
from quantpulse.quant.risk import (
    annualized_return,
    annualized_volatility,
    beta,
    cornish_fisher_var,
    historical_var_cvar,
    horizon_returns,
    ledoit_wolf,
    max_drawdown,
    monte_carlo_var_cvar,
    parametric_var_cvar,
    risk_contributions,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
)


def test_simple_and_horizon_returns():
    r = simple_returns(np.array([100.0, 110.0, 99.0]))
    np.testing.assert_allclose(r, [0.1, -0.1])
    np.testing.assert_allclose(horizon_returns(r, 2), [110 / 100 * 99 / 110 - 1])


def test_historical_var_cvar_on_known_sample():
    r = np.linspace(-0.05, 0.05, 101)  # uniform grid
    var, cvar = historical_var_cvar(r, 0.95)
    q = np.quantile(r, 0.05)
    assert var == pytest.approx(-q)
    assert cvar == pytest.approx(-r[r <= q].mean())
    assert cvar >= var


def test_parametric_var_matches_closed_form():
    var, cvar = parametric_var_cvar(0.0005, 0.01, 0.99, horizon=4)
    z = norm.ppf(0.99)
    assert var == pytest.approx(-(0.002 - z * 0.02))
    assert cvar == pytest.approx(-(0.002 - 0.02 * norm.pdf(z) / 0.01))


def test_cornish_fisher_equals_gaussian_for_symmetric_mesokurtic_sample():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 200000)
    mod = cornish_fisher_var(r, 0.95)
    gauss, _ = parametric_var_cvar(r.mean(), r.std(ddof=1), 0.95)
    assert mod == pytest.approx(gauss, rel=0.02)


def test_monte_carlo_var_close_to_parametric():
    mean = np.array([0.0004, 0.0002])
    cov = np.array([[1e-4, 3e-5], [3e-5, 2e-4]])
    w = np.array([0.6, 0.4])
    var, cvar = monte_carlo_var_cvar(mean, cov, w, 0.95, paths=200000, seed=3)
    p_var, p_cvar = parametric_var_cvar(float(w @ mean), math.sqrt(w @ cov @ w), 0.95)
    assert var == pytest.approx(p_var, rel=0.02)
    assert cvar == pytest.approx(p_cvar, rel=0.02)


def test_performance_ratios():
    r = np.array([0.01, -0.005, 0.007, -0.002, 0.004] * 60)
    assert annualized_volatility(r) == pytest.approx(r.std(ddof=1) * math.sqrt(252))
    assert annualized_return(r) == pytest.approx(np.prod(1 + r) ** (252 / r.size) - 1)
    s = sharpe_ratio(r, 0.02)
    so = sortino_ratio(r, 0.02)
    assert s is not None and so is not None and so > s > 0
    assert sharpe_ratio(np.zeros(10)) is None


def test_max_drawdown_includes_initial_peak():
    assert max_drawdown(np.array([-0.5, 1.0])) == pytest.approx(-0.5)
    assert max_drawdown(np.array([0.1, -0.2, 0.05])) == pytest.approx(0.88 / 1.1 - 1)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0


def test_beta_and_risk_contributions():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 0.01, 5000)
    a = 1.5 * b + rng.normal(0, 0.002, 5000)
    assert beta(a, b) == pytest.approx(1.5, rel=0.02)
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    rc = risk_contributions(np.array([0.5, 0.5]), cov)
    assert rc.sum() == pytest.approx(1.0)
    assert rc[1] > rc[0]


def test_ledoit_wolf_matches_reference_formula():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 8)) @ rng.normal(size=(8, 8)) * 0.01
    shrunk, delta = ledoit_wolf(x)
    assert 0.0 <= delta <= 1.0
    xc = x - x.mean(0)
    sample = xc.T @ xc / len(x)
    mu = np.trace(sample) / 8
    np.testing.assert_allclose(shrunk, (1 - delta) * sample + delta * mu * np.eye(8))
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)


def _market():
    mu = np.array([0.08, 0.12, 0.15, 0.05])
    vol = np.array([0.15, 0.22, 0.30, 0.08])
    corr = np.array([[1, 0.3, 0.2, 0.1], [0.3, 1, 0.4, 0.0], [0.2, 0.4, 1, -0.1], [0.1, 0.0, -0.1, 1]])
    return mu, np.outer(vol, vol) * corr


def test_min_variance_is_minimal_and_feasible():
    mu, cov = _market()
    gmv = opt.min_variance(mu, cov, 0.03)
    assert gmv.weights.sum() == pytest.approx(1.0)
    assert np.all(gmv.weights >= -1e-9)
    rng = np.random.default_rng(0)
    for w in rng.dirichlet(np.ones(4), 500):
        assert gmv.volatility <= math.sqrt(w @ cov @ w) + 1e-9


def test_max_sharpe_beats_random_portfolios():
    mu, cov = _market()
    tangency = opt.max_sharpe(mu, cov, 0.03)
    for p in opt.random_portfolios(mu, cov, 0.03, count=2000):
        assert tangency.sharpe >= p.sharpe - 1e-6


def test_unconstrained_tangency_matches_analytic_solution():
    mu, cov = _market()
    rf = 0.03
    raw = np.linalg.solve(cov, mu - rf)
    analytic = raw / raw.sum()
    tangency = opt.max_sharpe(mu, cov, rf, bounds=(-2.0, 2.0))
    np.testing.assert_allclose(tangency.weights, analytic, atol=1e-4)


def test_frontier_is_monotonic():
    mu, cov = _market()
    frontier = opt.efficient_frontier(mu, cov, 0.03, points=20)
    rets = [p.expected_return for p in frontier]
    vols = [p.volatility for p in frontier]
    assert len(frontier) >= 15
    assert all(b > a for a, b in pairwise(rets))
    assert all(b >= a - 1e-9 for a, b in pairwise(vols))
    assert rets[-1] == pytest.approx(max(mu), abs=1e-4)


def test_infeasible_bounds_rejected():
    mu, cov = _market()
    with pytest.raises(DomainError):
        opt.min_variance(mu, cov, 0.0, bounds=(0.0, 0.2))
