"""Statistical agent: does this stock's recent behaviour trend or mean-revert, and where is it now?

From a year of daily log returns: the Lo–MacKinlay variance ratio VR(5) with its z-statistic (VR < 1:
mean-reverting, > 1: trending), first-order autocorrelation, and a market-model regression on the
benchmark (beta, R²) whose residual over the last ten days, in residual-volatility units, says how far the
stock has moved on its own. In a mean-reverting regime a large residual is expected to fade; in a trending
one it is expected to continue. Without a significant regime the agent leans only weakly toward the
well-documented short-term reversal of idiosyncratic moves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..context import BrainContext
from ..types import AgentFamily, AgentSpec, Evidence, Opinion
from .base import Agent, symbols_only
from .common import opinion, sgn, squash, symbol_quality

LOOKBACK = 252
Q = 5
RESID_DAYS = 10


def variance_ratio(r: np.ndarray, q: int = Q) -> tuple[float, float]:
    """Lo–MacKinlay variance ratio and its homoskedastic z-statistic."""
    n = r.size
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    varq = np.sum((rq - q * mu) ** 2) / (q * (n - q + 1) * (1 - q / n))
    vr = varq / var1 if var1 > 0 else 1.0
    se = math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    return float(vr), float((vr - 1) / se)


def market_model(r: np.ndarray, b: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS beta, R² and residuals of ``r`` on ``b``."""
    bm, rm = b.mean(), r.mean()
    vb = np.sum((b - bm) ** 2)
    beta = float(np.sum((b - bm) * (r - rm)) / vb) if vb > 0 else 0.0
    alpha = rm - beta * bm
    resid = r - alpha - beta * b
    ss = np.sum((r - rm) ** 2)
    r2 = float(1 - np.sum(resid**2) / ss) if ss > 0 else 0.0
    return beta, r2, resid


class StatisticalAgent(Agent):
    spec = AgentSpec(
        id="statistical",
        source="prices",
        failure="no serial-dependence vote; the consensus lists it as missing",
        name="Statistical",
        description="Variance ratio and autocorrelation (trending vs mean-reverting), market-model beta and "
        "the last ten days' idiosyncratic move.",
        family=AgentFamily.SPECIALIST,
        capabilities=("variance_ratio", "autocorrelation", "market_model", "residual_reversal"),
        inputs=("close", "benchmark"),
        subjects=("symbol",),
        priority=35,
        horizon_days=5,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if len(ctx.benchmark.dropna()) >= 150 else "not enough benchmark history"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        return [self._one(ctx, s) for s in symbols_only(subjects)]

    def _one(self, ctx: BrainContext, s: str) -> Opinion:
        if s not in ctx.close.columns:
            return self.abstain(s, "no price history", ["daily bars"])
        pair = np.log(ctx.close[[s]].join(ctx.benchmark.rename("__b"), how="inner").dropna()).diff().dropna()
        pair = pair.iloc[-LOOKBACK:]
        if len(pair) < 150:
            return self.abstain(s, f"only {len(pair)} overlapping daily returns (150 needed)", ["daily bars"])
        r, b = pair[s].to_numpy(dtype=float), pair["__b"].to_numpy(dtype=float)
        q = symbol_quality(ctx, s)
        vr, vr_z = variance_ratio(r)
        ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        beta, r2, resid = market_model(r, b)
        sd = float(np.std(resid[:-RESID_DAYS], ddof=1))
        resid_z = float(resid[-RESID_DAYS:].sum() / (sd * math.sqrt(RESID_DAYS))) if sd > 0 else 0.0
        if vr_z < -1.5 or ac1 < -0.1:
            regime, score = "mean-reverting", -squash(resid_z, 2.0)
        elif vr_z > 1.5:
            regime, score = "trending", 0.8 * squash(resid_z, 2.0)
        else:
            regime, score = "no significant regime", -0.3 * squash(resid_z, 2.0)
        ev = [
            Evidence(
                "variance_ratio",
                round(vr, 3),
                f"variance ratio VR({Q}) {vr:.2f} (z {vr_z:+.1f}): {regime}",
                0,
                0.7,
                quality=q,
            ),
            Evidence("autocorr_1", round(ac1, 3), f"lag-1 autocorrelation {ac1:+.2f}", 0, 0.4, quality=q),
            Evidence(
                "residual_z",
                round(resid_z, 2),
                f"{RESID_DAYS}-day move beyond the market {resid_z:+.1f}σ",
                sgn(score),
                0.7,
                quality=q,
            ),
            Evidence("beta", round(beta, 2), f"beta {beta:.2f}, R² {r2:.0%}", 0, 0.3, quality=q),
        ]
        significance = min(abs(vr_z) / 3, 1.0)
        confidence = 0.15 + 0.3 * significance + 0.15 * min(abs(resid_z) / 2.5, 1.0)
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} statistically {regime}; " + ", ".join(e.detail for e in ev[1:3]),
            ev,
            quality=q,
            used=[f"{len(pair)} daily returns", f"{ctx.benchmark_symbol} returns"],
            invalidation=None,
            meta={
                "vr": round(vr, 3),
                "vr_z": round(vr_z, 2),
                "ac1": round(ac1, 3),
                "beta": round(beta, 3),
                "r2": round(r2, 3),
                "resid_z": round(resid_z, 2),
                "regime": regime,
            },
        )
