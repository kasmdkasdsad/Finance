"""Unlevered free-cash-flow DCF (FCFF discounted at WACC) with Gordon-growth terminal value.

    FCFF_t = EBIT_t·(1 − tax) + D&A_t − CapEx_t − ΔNWC_t        (tax only applies to positive EBIT)
    TV_N   = FCFF_N·(1 + g) / (WACC − g)
    EV     = Σ FCFF_t·DF(t) + TV_N·DF_TV
    Equity = EV − debt + cash;   value/share = Equity / diluted shares

With the mid-year convention cash flows are discounted at ``t − 0.5``; the terminal value (the value at
``N`` of mid-year perpetuity flows) is then consistently discounted at ``N − 0.5``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantpulse.core.errors import DomainError
from quantpulse.schemas.fundamentals import (
    DCFInputs,
    DCFOutput,
    DCFProjectionRow,
    SensitivityGrid,
)


def _periods(years: int, mid_year: bool) -> tuple[NDArray[np.float64], float]:
    t = np.arange(1, years + 1, dtype=float)
    if mid_year:
        return t - 0.5, years - 0.5
    return t, float(years)


def run_dcf(inputs: DCFInputs) -> DCFOutput:
    periods, tv_period = _periods(inputs.years, inputs.mid_year_convention)
    rows: list[DCFProjectionRow] = []
    prev_revenue = inputs.base_revenue
    sum_pv = 0.0
    fcf = 0.0
    for i, (growth, margin) in enumerate(zip(inputs.revenue_growth, inputs.ebit_margin, strict=True)):
        revenue = prev_revenue * (1.0 + growth)
        ebit = revenue * margin
        nopat = ebit - max(ebit, 0.0) * inputs.tax_rate
        da = revenue * inputs.da_pct_revenue
        capex = revenue * inputs.capex_pct_revenue
        delta_nwc = inputs.nwc_pct_incremental_revenue * (revenue - prev_revenue)
        fcf = nopat + da - capex - delta_nwc
        df = (1.0 + inputs.wacc) ** (-periods[i])
        pv = fcf * df
        sum_pv += pv
        rows.append(
            DCFProjectionRow(
                year=i + 1,
                revenue=revenue,
                growth=growth,
                ebit=ebit,
                ebit_margin=margin,
                nopat=nopat,
                depreciation_amortization=da,
                capital_expenditure=capex,
                change_in_nwc=delta_nwc,
                free_cash_flow=fcf,
                discount_factor=float(df),
                present_value=float(pv),
            )
        )
        prev_revenue = revenue

    terminal_value = fcf * (1.0 + inputs.terminal_growth) / (inputs.wacc - inputs.terminal_growth)
    pv_tv = terminal_value * (1.0 + inputs.wacc) ** (-tv_period)
    ev = sum_pv + pv_tv
    equity = ev - inputs.debt + inputs.cash
    per_share = equity / inputs.shares_outstanding
    upside = (per_share / inputs.current_price - 1.0) if inputs.current_price else None
    return DCFOutput(
        projections=rows,
        sum_pv_fcf=float(sum_pv),
        terminal_value=float(terminal_value),
        pv_terminal_value=float(pv_tv),
        enterprise_value=float(ev),
        equity_value=float(equity),
        value_per_share=float(per_share),
        current_price=inputs.current_price,
        upside=None if upside is None else float(upside),
        terminal_value_share=float(pv_tv / ev) if ev != 0 else float("nan"),
    )


def dcf_value_per_share_vec(
    inputs: DCFInputs,
    wacc: NDArray[np.float64],
    terminal_growth: NDArray[np.float64],
    revenue_growth: NDArray[np.float64],
    ebit_margin: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorised value per share for ``n`` scenarios.

    ``wacc`` and ``terminal_growth`` have shape ``(n,)``; ``revenue_growth`` and ``ebit_margin`` have shape
    ``(n, years)``. Scenarios with ``terminal_growth >= wacc`` return NaN.
    """
    n, years = revenue_growth.shape
    if ebit_margin.shape != (n, years) or wacc.shape != (n,) or terminal_growth.shape != (n,):
        raise DomainError("scenario arrays have inconsistent shapes")
    periods, tv_period = _periods(years, inputs.mid_year_convention)
    growth_factors = np.cumprod(1.0 + revenue_growth, axis=1)
    revenue = inputs.base_revenue * growth_factors
    prev_revenue = np.concatenate([np.full((n, 1), inputs.base_revenue), revenue[:, :-1]], axis=1)
    ebit = revenue * ebit_margin
    nopat = ebit - np.maximum(ebit, 0.0) * inputs.tax_rate
    fcf = (
        nopat
        + revenue * inputs.da_pct_revenue
        - revenue * inputs.capex_pct_revenue
        - inputs.nwc_pct_incremental_revenue * (revenue - prev_revenue)
    )
    discount = (1.0 + wacc[:, None]) ** (-periods[None, :])
    sum_pv = np.sum(fcf * discount, axis=1)
    spread = wacc - terminal_growth
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = fcf[:, -1] * (1.0 + terminal_growth) / spread
    pv_tv = tv * (1.0 + wacc) ** (-tv_period)
    equity = sum_pv + pv_tv - inputs.debt + inputs.cash
    out = equity / inputs.shares_outstanding
    return np.where(spread > 0, out, np.nan)


def sensitivity_grid(
    inputs: DCFInputs,
    wacc_deltas: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02),
    growth_deltas: tuple[float, ...] = (-0.01, -0.005, 0.0, 0.005, 0.01),
) -> SensitivityGrid:
    waccs = [round(inputs.wacc + d, 6) for d in wacc_deltas]
    growths = [round(inputs.terminal_growth + d, 6) for d in growth_deltas]
    values: list[list[float | None]] = []
    growth_path = np.asarray(inputs.revenue_growth, dtype=float)[None, :]
    margin_path = np.asarray(inputs.ebit_margin, dtype=float)[None, :]
    for w in waccs:
        row: list[float | None] = []
        for g in growths:
            if w <= 0 or g >= w:
                row.append(None)
                continue
            v = dcf_value_per_share_vec(inputs, np.array([w]), np.array([g]), growth_path, margin_path)[0]
            row.append(float(v) if np.isfinite(v) else None)
        values.append(row)
    return SensitivityGrid(wacc_values=waccs, growth_values=growths, values_per_share=values)


def fade_path(start: float, end: float, years: int) -> list[float]:
    """Linear path from ``start`` (year 1) to ``end`` (final year)."""
    if years < 1:
        raise DomainError("years must be >= 1")
    if years == 1:
        return [end]
    return [float(v) for v in np.linspace(start, end, years)]


def growth_path(near_term: list[float], terminal_growth: float, years: int) -> list[float]:
    """Use explicit near-term (e.g. consensus) growth for the first years, then fade linearly to the
    terminal growth rate by the final projection year."""
    if years < 1:
        raise DomainError("years must be >= 1")
    near = list(near_term[:years])
    if not near:
        raise DomainError("at least one near-term growth rate is required")
    remaining = years - len(near)
    if remaining <= 0:
        return near
    fade = np.linspace(near[-1], terminal_growth, remaining + 1)[1:]
    return near + [float(v) for v in fade]
