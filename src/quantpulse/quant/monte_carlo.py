"""Monte Carlo sensitivity analysis of the DCF around the live baseline.

Each path draws (independently, normally distributed around the baseline):

* WACC                   — ``wacc + N(0, wacc_sd)``, floored at 1%;
* terminal growth        — ``g + N(0, g_sd)``, capped at ``WACC − 0.5%`` so the Gordon model stays finite;
* revenue growth shock   — one persistent level shift per path added to every projection year;
* EBIT margin shock      — one persistent shift per path added to every projection year.
"""

from __future__ import annotations

import secrets

import numpy as np

from quantpulse.quant.dcf import dcf_value_per_share_vec
from quantpulse.schemas.fundamentals import DCFInputs, MonteCarloConfig, MonteCarloResult

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
MIN_WACC = 0.01
MIN_SPREAD = 0.005


def simulate_dcf(inputs: DCFInputs, config: MonteCarloConfig) -> MonteCarloResult:
    seed = config.seed if config.seed is not None else secrets.randbelow(2**31)
    rng = np.random.default_rng(seed)
    n, years = config.paths, inputs.years

    wacc = np.maximum(inputs.wacc + rng.normal(0.0, config.wacc_sd, n), MIN_WACC)
    growth = inputs.terminal_growth + rng.normal(0.0, config.terminal_growth_sd, n)
    growth = np.minimum(growth, wacc - MIN_SPREAD)
    growth_shock = rng.normal(0.0, config.revenue_growth_sd, (n, 1))
    margin_shock = rng.normal(0.0, config.margin_sd, (n, 1))
    revenue_growth = np.maximum(np.asarray(inputs.revenue_growth)[None, :] + growth_shock, -0.95)
    margins = np.clip(np.asarray(inputs.ebit_margin)[None, :] + margin_shock, -1.0, 1.0)
    if revenue_growth.shape != (n, years):  # pragma: no cover - defensive
        raise ValueError("unexpected scenario shape")

    values = dcf_value_per_share_vec(inputs, wacc, growth, revenue_growth, margins)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no valid Monte Carlo paths")

    edges = _robust_edges(values, config.bins)
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    pct = np.percentile(values, PERCENTILES)
    prob = float(np.mean(values > inputs.current_price)) if inputs.current_price else None
    return MonteCarloResult(
        paths=n,
        valid_paths=int(values.size),
        seed=int(seed),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        percentiles={f"p{p}": float(v) for p, v in zip(PERCENTILES, pct, strict=True)},
        prob_above_price=prob,
        current_price=inputs.current_price,
        histogram_edges=[float(e) for e in edges],
        histogram_counts=[int(c) for c in counts],
    )


def _robust_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Uniform histogram edges spanning the 0.5–99.5th percentiles.

    Values outside that window are clipped into the first/last bin so extreme tail paths remain counted
    without flattening the chart.
    """
    lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.5]))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, bins + 1)
