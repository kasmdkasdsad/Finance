"""Benchmarks every options strategy must beat — or be flagged.

1. **Underlying buy-and-hold** — equal weights, held through the period;
2. **Simple directional stock** — long the underlying while its 50-day average is above its 200-day, else cash;
3. **Simple option baseline** — a plain, unoptimized structure of the same direction (a 45-day at-the-money
   long call or put, or for premium-selling families a 30-delta put spread), entered whenever allowed;
4. **Randomized controls** — the same family and constraints with random entries (and random strikes and
   expirations inside the same bounds), over several seeds: where the strategy's result falls in the
   distribution of luck;
5. **No trade** — zero: a strategy that loses money is worse than doing nothing.

The incremental value is the strategy's result minus each benchmark's, per dollar of capital.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np

from quantpulse.options.lab.backtest import BacktestConfig, run
from quantpulse.options.lab.chains import ChainSource
from quantpulse.options.lab.features import DayFeatures
from quantpulse.options.lab.genome import BOUNDS, Genome


def buy_and_hold(closes: Mapping[str, Mapping[date, float]], start: date, end: date) -> float | None:
    rets = []
    for c in closes.values():
        days = [d for d in sorted(c) if start <= d <= end]
        if len(days) >= 2:
            rets.append(c[days[-1]] / c[days[0]] - 1)
    return float(np.mean(rets)) if rets else None


def trend_following(closes: Mapping[str, Mapping[date, float]], start: date, end: date) -> float | None:
    rets = []
    for c in closes.values():
        days = sorted(c)
        px = np.array([c[d] for d in days])
        total = 1.0
        for i in range(200, len(days) - 1):
            if not (start <= days[i] < end):
                continue
            if px[i - 49 : i + 1].mean() > px[i - 199 : i + 1].mean():
                total *= px[i + 1] / px[i]
        if len(days) > 200:
            rets.append(total - 1)
    return float(np.mean(rets)) if rets else None


def simple_option(g: Genome) -> Genome:
    """The plain structure of the same direction, entered whenever allowed (no filters, no optimization)."""
    base: dict[str, Any] = {
        "entry_signal": "always", "iv_rank_min": None, "iv_rank_max": None, "iv_percentile_min": None,
        "iv_percentile_max": None, "iv_rv_min": None, "iv_rv_max": None, "term_filter": "any", "skew_filter": "any",
        "regime_filter": (), "event_filter": "avoid", "take_profit": 0.5, "stop_loss": 1.0, "exit_dte": 7,
        "max_hold_days": 45,
    }  # fmt: skip
    if g.direction == "bearish":
        return Genome("long_put", "bearish", dte_min=35, dte_max=55, delta_target=0.5, **base)
    if g.family in (
        "bull_put_spread",
        "cash_secured_put",
        "covered_call",
        "iron_condor",
        "bear_call_spread",
        "iron_butterfly",
        "put_butterfly",
        "broken_wing_butterfly",
    ):
        return Genome("bull_put_spread", "bullish", dte_min=25, dte_max=45, delta_target=0.3, width_pct=0.05,
                      **{**base, "take_profit": 0.5, "stop_loss": 2.0})  # fmt: skip
    return Genome("long_call", "bullish", dte_min=35, dte_max=55, delta_target=0.5, **base)


def random_control(g: Genome, seed: int) -> Genome:
    """Random entries — and a random strike (delta) and expiration window inside the genome's own bounds."""
    rng = random.Random(seed)
    lo, hi, _ = BOUNDS["delta_target"]
    delta = round(rng.uniform(max(lo, g.delta_target - 0.15), min(hi, g.delta_target + 0.15)), 2)
    span = g.dte_max - g.dte_min
    dmin = max(g.exit_dte + 1, g.dte_min + rng.randint(-5, 5))
    return replace(g, entry_signal="random", seed=seed, delta_target=delta, dte_min=dmin, dte_max=dmin + span,
                   regime_filter=(), iv_rank_min=None, iv_rank_max=None)  # fmt: skip


def compare(
    g: Genome,
    result_ror: float | None,
    source: ChainSource,
    features: Mapping[str, Mapping[date, DayFeatures]],
    closes: Mapping[str, Mapping[date, float]],
    cfg: BacktestConfig,
    *,
    random_seeds: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
) -> dict[str, Any]:
    """The strategy's expectancy per dollar at risk against every benchmark."""
    simple = run(simple_option(g), source, features, cfg).metrics
    randoms = []
    for s in random_seeds:
        rc = random_control(g, s)
        if rc.valid:
            m = run(rc, source, features, cfg).metrics
            if m.get("expectancy_on_risk") is not None:
                randoms.append(float(m["expectancy_on_risk"]))
    pct = None
    if result_ror is not None and randoms:
        pct = round(float((np.array(randoms) < result_ror).mean() * 100), 1)
    out: dict[str, Any] = {
        "buy_and_hold_return": buy_and_hold(closes, cfg.start, cfg.end),
        "trend_following_return": trend_following(closes, cfg.start, cfg.end),
        "simple_option": {"family": simple_option(g).family, "expectancy_on_risk": simple.get("expectancy_on_risk"),
                          "total_return": simple.get("total_return"), "trades": simple.get("trades")},
        "random": {"seeds": len(randoms), "expectancy_on_risk": [round(r, 4) for r in randoms],
                   "strategy_percentile": pct},
        "no_trade": 0.0,
        "strategy_expectancy_on_risk": result_ror,
    }  # fmt: skip
    flags = []
    if result_ror is not None and result_ror <= 0:
        flags.append("worse than not trading")
    s_ror = simple.get("expectancy_on_risk")
    if result_ror is not None and s_ror is not None and result_ror <= s_ror:
        flags.append("no better than the simple option baseline")
    if pct is not None and pct < 75:
        flags.append(f"only at the {pct:.0f}th percentile of random controls")
    out["flags"] = flags
    out["beats_baselines"] = not flags and result_ror is not None
    return out
