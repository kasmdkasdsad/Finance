"""Volatility agent: the risk regime of each focus symbol and of the market, and what it implies.

Per symbol it fits the existing GARCH(1,1)-t model (:func:`quantpulse.quant.volatility.fit_best`, EWMA when
history is short) on the daily closes for a 10-day volatility forecast, and reads realised volatility
(21/63-day), the 1- vs 3-month ratio, Bollinger band-width percentile, return skew and kurtosis, drawdown
and implied volatility from the shared table. Volatility says little about direction, so the agent's
directional votes are deliberately small and low-confidence; its main output is the forecast and a size
scale that the decision step uses (more volatile → smaller positions). For the market it reads the VIX and
the benchmark's own volatility regime.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import numpy as np

from quantpulse.quant.volatility import fit_best

from ..context import BrainContext
from ..types import MARKET, AgentFamily, AgentSpec, DataState, Evidence, Opinion
from .base import Agent, symbols_only
from .common import opinion, sgn, squash, symbol_quality

HORIZON = 10
TARGET_VOL = 0.30  # a position at this forecast volatility keeps its normal size


def regime_label(vol: float) -> str:
    return "calm" if vol < 0.15 else "normal" if vol < 0.35 else "elevated" if vol < 0.60 else "extreme"


def garch_forecast(closes: np.ndarray, horizon: int = HORIZON) -> tuple[float | None, str | None]:
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if closes.size < 60:
        return None, None
    returns = np.diff(np.log(closes[-750:]))
    try:
        fit = fit_best(returns)
    except Exception:  # an estimation failure only removes one input
        return None, None
    return fit.annualized_volatility(horizon), fit.method


class VolatilityAgent(Agent):
    spec = AgentSpec(
        id="volatility",
        name="Volatility",
        description="GARCH volatility forecast, realised-volatility regime, expansion/compression, tail shape "
        "and implied volatility; sets a volatility-based size scale.",
        family=AgentFamily.SPECIALIST,
        capabilities=("garch_forecast", "vol_regime", "vol_expansion", "tail_risk", "size_scale"),
        inputs=("close", "indicators"),
        subjects=("market", "symbol"),
        priority=25,
        horizon_days=HORIZON,
    )

    def unavailable(self, ctx: BrainContext) -> str | None:
        return None if not ctx.close.empty else "no price history"

    async def analyze(self, ctx: BrainContext, subjects: Sequence[str]) -> list[Opinion]:
        syms = [s for s in symbols_only(subjects) if s in ctx.close.columns]
        forecasts = await asyncio.gather(
            *(asyncio.to_thread(garch_forecast, ctx.close[s].to_numpy(dtype=float)) for s in syms)
        )
        by_symbol = dict(zip(syms, forecasts, strict=True))
        out = [self._one(ctx, s, *by_symbol.get(s, (None, None))) for s in symbols_only(subjects)]
        ctx.working.post(
            "vol_forecast",
            {o.subject: o.meta["forecast_vol"] for o in out if o.meta.get("forecast_vol") is not None},
        )
        if MARKET in subjects:
            out.insert(0, self._market(ctx))
        return out

    def _one(self, ctx: BrainContext, s: str, garch: float | None, method: str | None) -> Opinion:
        g = lambda c: ctx.ind(s, c)  # noqa: E731
        rv21, rv63 = g("rv21"), g("rv63")
        if rv21 is None and garch is None:
            return self.abstain(s, "not enough history for a volatility estimate", ["daily bars"])
        q = symbol_quality(ctx, s)
        forecast = garch if garch is not None else rv21
        assert forecast is not None
        ev = [
            Evidence(
                "forecast_vol",
                round(forecast, 4),
                f"{HORIZON}-day {method or 'realised'} volatility forecast {forecast:.0%} ({regime_label(forecast)})",
                0,
                0.8,
                quality=q,
            )
        ]
        parts: dict[str, float] = {}
        ratio, ret21 = g("vol_ratio"), g("ret_21d") or 0.0
        if ratio is not None:
            if ratio > 1.3:
                parts["expansion"] = -0.5 if ret21 < 0 else 0.15
                what = "into a decline" if ret21 < 0 else "with the price rising"
                ev.append(
                    Evidence(
                        "vol_ratio",
                        round(ratio, 2),
                        f"volatility expanding ({ratio:.1f}× the 3-month level) {what}",
                        sgn(parts["expansion"]),
                        0.6,
                        quality=q,
                    )
                )
            elif ratio < 0.75:
                ev.append(
                    Evidence(
                        "vol_ratio",
                        round(ratio, 2),
                        f"volatility contracting ({ratio:.1f}× the 3-month level)",
                        0,
                        0.4,
                        quality=q,
                    )
                )
        width = g("bb_width_pct")
        above50 = (g("px_vs_sma50") or 0.0) > 0
        if width is not None and width < 0.2:
            parts["squeeze"] = 0.3 if above50 else -0.2
            ev.append(
                Evidence(
                    "bb_width_pct",
                    round(width, 2),
                    f"Bollinger bands in their narrowest {width:.0%} of the year ({'above' if above50 else 'below'} the 50-day)",
                    sgn(parts["squeeze"]),
                    0.5,
                    quality=q,
                )
            )
        skew, kurt = g("skew_63"), g("kurt_63")
        if skew is not None and kurt is not None and skew < -0.5 and kurt > 3:
            parts["tails"] = -0.3
            ev.append(
                Evidence(
                    "tail_shape",
                    round(skew, 2),
                    f"negative skew {skew:.1f} with fat tails (kurtosis {kurt:.1f})",
                    -1,
                    0.5,
                    quality=q,
                )
            )
        dd = g("drawdown_252")
        if dd is not None and dd < -0.25 and (ratio or 1.0) > 1.1:
            parts["drawdown"] = -0.3
            ev.append(
                Evidence(
                    "drawdown_252",
                    round(dd, 3),
                    f"{dd:.0%} below its 1-year high with volatility rising",
                    -1,
                    0.5,
                    quality=q,
                )
            )
        iv = g("implied_vol")
        if iv is not None:
            ev.append(
                Evidence(
                    "implied_vol",
                    round(iv, 4),
                    f"options imply {iv:.0%} volatility vs {forecast:.0%} forecast",
                    0,
                    0.4,
                    source="options",
                    quality=q,
                )
            )
        score = squash(sum(parts.values()), 1.0) if parts else 0.0
        scale = max(0.25, min(1.0, TARGET_VOL / max(forecast, 1e-6)))
        confidence = 0.2 + 0.25 * min(len(parts) / 2, 1.0)
        return opinion(
            self,
            s,
            score,
            confidence,
            f"{s} volatility {regime_label(forecast)}: " + ", ".join(e.detail for e in ev[:3]),
            ev,
            quality=q,
            used=["daily closes (GARCH)", "realised volatility", *(["implied volatility"] if iv else [])],
            invalidation=None,
            directional=bool(parts),  # no volatility signal: the forecast is context, not a vote
            meta={
                "forecast_vol": round(forecast, 4),
                "method": method or "realised",
                "regime": regime_label(forecast),
                "size_scale": round(scale, 3),
                "rv21": rv21,
                "rv63": rv63,
            },
        )

    def _market(self, ctx: BrainContext) -> Opinion:
        b = ctx.benchmark.dropna()
        if len(b) < 60:
            return self.abstain(MARKET, "not enough benchmark history", ["benchmark"])
        r = np.diff(np.log(b.to_numpy(dtype=float)))
        rv21 = float(np.std(r[-21:], ddof=1) * math.sqrt(252))
        rv252 = float(np.std(r[-252:], ddof=1) * math.sqrt(252))
        ev = [
            Evidence(
                "benchmark_rv21",
                round(rv21, 4),
                f"{ctx.benchmark_symbol} 1-month volatility {rv21:.0%} (1-year {rv252:.0%})",
                0,
                0.6,
            )
        ]
        parts: dict[str, float] = {}
        if rv21 > 1.5 * rv252:
            parts["expansion"] = -0.4
            ev.append(
                Evidence(
                    "vol_expansion",
                    round(rv21 / rv252, 2),
                    "market volatility well above its 1-year level",
                    -1,
                    0.6,
                )
            )
        elif rv21 < 0.7 * rv252:
            parts["calm"] = 0.2
            ev.append(
                Evidence(
                    "vol_calm", round(rv21 / rv252, 2), "market volatility below its 1-year level", 1, 0.4
                )
            )
        if ctx.vix is not None:
            vix = ctx.vix
            parts["vix"] = -0.5 if vix >= 30 else -0.2 if vix >= 22 else 0.2 if vix < 15 else 0.0
            ev.append(Evidence("vix", round(vix, 2), f"VIX {vix:.1f}", sgn(parts["vix"]), 0.7, source="cboe"))
        score = squash(sum(parts.values()), 1.0) if parts else 0.0
        return opinion(
            self,
            MARKET,
            score,
            0.25 + 0.2 * min(len(parts), 2) / 2,
            f"market volatility {regime_label(rv21)}: " + ", ".join(e.detail for e in ev[:3]),
            ev,
            quality=DataState.LIVE if ctx.market_open else DataState.MARKET_CLOSED,
            used=["benchmark closes", *(["VIX"] if ctx.vix is not None else [])],
            meta={"benchmark_rv21": round(rv21, 4), "benchmark_rv252": round(rv252, 4), "vix": ctx.vix},
        )
