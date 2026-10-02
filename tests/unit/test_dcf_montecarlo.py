import numpy as np
import pytest
from pydantic import ValidationError

from quantpulse.quant.dcf import dcf_value_per_share_vec, fade_path, growth_path, run_dcf, sensitivity_grid
from quantpulse.quant.monte_carlo import simulate_dcf
from quantpulse.schemas.fundamentals import DCFInputs, MonteCarloConfig


def _inputs(**overrides):
    base = dict(
        base_revenue=1000.0,
        revenue_growth=[0.10, 0.10],
        ebit_margin=[0.20, 0.20],
        tax_rate=0.25,
        da_pct_revenue=0.05,
        capex_pct_revenue=0.06,
        nwc_pct_incremental_revenue=0.10,
        wacc=0.10,
        terminal_growth=0.02,
        cash=100.0,
        debt=300.0,
        shares_outstanding=100.0,
        mid_year_convention=False,
        current_price=15.0,
    )
    base.update(overrides)
    return DCFInputs(**base)


def test_dcf_hand_computed():
    out = run_dcf(_inputs())
    # Year 1: rev 1100, EBIT 220, NOPAT 165, D&A 55, capex 66, dNWC 10 -> FCF 144
    # Year 2: rev 1210, EBIT 242, NOPAT 181.5, D&A 60.5, capex 72.6, dNWC 11 -> FCF 158.4
    assert out.projections[0].free_cash_flow == pytest.approx(144.0)
    assert out.projections[1].free_cash_flow == pytest.approx(158.4)
    pv = 144.0 / 1.1 + 158.4 / 1.1**2
    tv = 158.4 * 1.02 / 0.08
    ev = pv + tv / 1.1**2
    assert out.sum_pv_fcf == pytest.approx(pv)
    assert out.terminal_value == pytest.approx(tv)
    assert out.enterprise_value == pytest.approx(ev)
    assert out.value_per_share == pytest.approx((ev - 300 + 100) / 100)
    assert out.upside == pytest.approx(out.value_per_share / 15.0 - 1)


def test_mid_year_convention_raises_value():
    assert run_dcf(_inputs(mid_year_convention=True)).enterprise_value > run_dcf(_inputs()).enterprise_value


def test_negative_ebit_is_not_taxed():
    out = run_dcf(_inputs(ebit_margin=[-0.1, -0.1]))
    assert out.projections[0].nopat == pytest.approx(out.projections[0].ebit)


def test_terminal_growth_must_be_below_wacc():
    with pytest.raises(ValidationError):
        _inputs(terminal_growth=0.10)
    with pytest.raises(ValidationError):
        _inputs(revenue_growth=[0.1])


def test_vectorised_value_matches_scalar_model():
    inp = _inputs(mid_year_convention=True)
    v = dcf_value_per_share_vec(
        inp,
        np.array([inp.wacc]),
        np.array([inp.terminal_growth]),
        np.array([inp.revenue_growth]),
        np.array([inp.ebit_margin]),
    )
    assert v[0] == pytest.approx(run_dcf(inp).value_per_share)


def test_sensitivity_grid_monotonic_and_masks_invalid():
    grid = sensitivity_grid(_inputs(wacc=0.03, terminal_growth=0.02))
    # wacc 0.01 row: every growth >= wacc -> None
    assert all(v is None for v in grid.values_per_share[0])
    mid = sensitivity_grid(_inputs())
    center = mid.values_per_share[2]
    assert all(center[i] < center[i + 1] for i in range(len(center) - 1))  # higher g -> higher value
    col = [row[2] for row in mid.values_per_share]
    assert all(col[i] > col[i + 1] for i in range(len(col) - 1))  # higher wacc -> lower value


def test_growth_and_fade_paths():
    assert growth_path([0.12, 0.08], 0.02, 5) == pytest.approx([0.12, 0.08, 0.06, 0.04, 0.02])
    assert growth_path([0.1, 0.1, 0.1], 0.02, 2) == [0.1, 0.1]
    assert fade_path(0.3, 0.2, 3) == pytest.approx([0.3, 0.25, 0.2])


def test_monte_carlo_is_reproducible_and_centred():
    inp = _inputs()
    cfg = MonteCarloConfig(paths=20000, seed=42)
    a, b = simulate_dcf(inp, cfg), simulate_dcf(inp, cfg)
    assert a == b
    assert a.valid_paths == 20000
    assert sum(a.histogram_counts) == a.valid_paths
    assert a.percentiles["p5"] < a.median < a.percentiles["p95"]
    assert a.median == pytest.approx(run_dcf(inp).value_per_share, rel=0.1)
    assert 0.0 <= a.prob_above_price <= 1.0


def test_monte_carlo_zero_volatility_collapses_to_point():
    cfg = MonteCarloConfig(
        paths=500, seed=1, wacc_sd=0, terminal_growth_sd=0, revenue_growth_sd=0, margin_sd=0
    )
    res = simulate_dcf(_inputs(), cfg)
    assert res.std == pytest.approx(0.0, abs=1e-9)
    assert res.mean == pytest.approx(run_dcf(_inputs()).value_per_share)
