import math

import numpy as np
import pytest

from quantpulse.core.errors import DomainError
from quantpulse.quant.black_scholes import (
    bsm_greeks,
    bsm_price,
    bsm_price_vec,
    implied_volatility,
    implied_volatility_vec,
)


def test_hull_textbook_values():
    # Hull, Options Futures & Other Derivatives, Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5
    call = bsm_price(42, 40, 0.5, 0.10, 0.20, kind="call")
    put = bsm_price(42, 40, 0.5, 0.10, 0.20, kind="put")
    assert call == pytest.approx(4.76, abs=0.005)
    assert put == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(q):
    s, k, t, r, vol = 100.0, 95.0, 0.75, 0.045, 0.3
    c = bsm_price(s, k, t, r, vol, q, "call")
    p = bsm_price(s, k, t, r, vol, q, "put")
    assert c - p == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    s, k, t, r, vol, q = 105.0, 100.0, 0.4, 0.04, 0.25, 0.015
    g = bsm_greeks(s, k, t, r, vol, q, kind)
    h = 1e-4
    price = lambda **kw: bsm_price(**{**dict(spot=s, strike=k, t=t, r=r, sigma=vol, q=q, kind=kind), **kw})
    delta = lambda **kw: (
        bsm_greeks(**{**dict(spot=s, strike=k, t=t, r=r, sigma=vol, q=q, kind=kind), **kw}).delta
    )
    assert g.delta == pytest.approx((price(spot=s + h) - price(spot=s - h)) / (2 * h), rel=1e-6)
    assert g.gamma == pytest.approx((price(spot=s + h) - 2 * g.price + price(spot=s - h)) / h**2, rel=1e-3)
    assert g.vega == pytest.approx((price(sigma=vol + h) - price(sigma=vol - h)) / (2 * h), rel=1e-6)
    assert g.theta == pytest.approx(-(price(t=t + h) - price(t=t - h)) / (2 * h), rel=1e-5)
    assert g.rho == pytest.approx((price(r=r + h) - price(r=r - h)) / (2 * h), rel=1e-6)
    assert g.vanna == pytest.approx((delta(sigma=vol + h) - delta(sigma=vol - h)) / (2 * h), rel=1e-5)
    vega = lambda v: bsm_greeks(s, k, t, r, v, q, kind).vega
    assert g.vomma == pytest.approx((vega(vol + h) - vega(vol - h)) / (2 * h), rel=1e-4)
    assert g.charm == pytest.approx(-(delta(t=t + h) - delta(t=t - h)) / (2 * h), rel=1e-4)


def test_expiry_and_zero_vol_limits():
    at_expiry = bsm_greeks(110, 100, 0.0, 0.05, 0.2, kind="call")
    assert at_expiry.price == pytest.approx(10.0)
    assert at_expiry.delta == 1.0 and at_expiry.gamma == 0.0
    otm_put = bsm_greeks(110, 100, 0.0, 0.05, 0.2, kind="put")
    assert otm_put.price == 0.0 and otm_put.delta == 0.0
    zero_vol = bsm_greeks(100, 100, 1.0, 0.05, 0.0, kind="call")
    assert zero_vol.price == pytest.approx(100 - 100 * math.exp(-0.05))


@pytest.mark.parametrize("bad", [dict(spot=0), dict(strike=-1), dict(t=-0.1), dict(sigma=-0.2)])
def test_invalid_inputs_raise(bad):
    args = dict(spot=100, strike=100, t=1.0, r=0.01, sigma=0.2)
    with pytest.raises(DomainError):
        bsm_greeks(**{**args, **bad})


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.8, 2.5])
def test_implied_vol_round_trip(kind, vol):
    s, k, t, r, q = 100.0, 110.0, 0.3, 0.05, 0.01
    px = bsm_price(s, k, t, r, vol, q, kind)
    assert implied_volatility(px, s, k, t, r, q, kind) == pytest.approx(vol, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    # A call cannot be worth less than its discounted intrinsic value or more than the spot.
    assert implied_volatility(5.0, 120, 100, 0.5, 0.05, kind="call") is None
    assert implied_volatility(130.0, 120, 100, 0.5, 0.05, kind="call") is None
    assert implied_volatility(0.0, 100, 100, 0.5, 0.05) is None


def test_vectorised_matches_scalar():
    s = 100.0
    strikes = np.array([80.0, 95.0, 100.0, 105.0, 130.0])
    vols = np.array([0.35, 0.28, 0.25, 0.24, 0.3])
    t = np.array([0.1, 0.5, 1.0, 0.25, 2.0])
    is_call = np.array([False, False, True, True, True])
    prices = bsm_price_vec(s, strikes, t, 0.04, vols, 0.01, is_call)
    for i in range(5):
        kind = "call" if is_call[i] else "put"
        assert prices[i] == pytest.approx(
            bsm_price(s, strikes[i], t[i], 0.04, vols[i], 0.01, kind), rel=1e-12
        )
    ivs = implied_volatility_vec(prices, s, strikes, t, 0.04, 0.01, is_call)
    np.testing.assert_allclose(ivs, vols, atol=1e-6)


def test_vectorised_iv_marks_unsolvable_as_nan():
    ivs = implied_volatility_vec(
        np.array([0.0, 1000.0, 5.0]),
        100.0,
        np.array([100.0, 100.0, 100.0]),
        np.array([0.5, 0.5, 0.0]),
        0.03,
        0.0,
        np.array([True, True, True]),
    )
    assert np.isnan(ivs).all()
