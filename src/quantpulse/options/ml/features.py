"""One fixed feature vector per option candidate — the same in research, in training and in the live Brain.

Every feature is computed from data available at the decision time only, and a missing value stays missing
(``nan``: the gradient-boosted trees route it on their own; nothing is imputed silently). Dollar amounts are
divided by the structure's maximum loss, so a feature means the same for one SPY spread and one AMD call.

Groups (the names are the contract with saved models — append, never rename or reorder):

* **underlying** — momentum and trend, realized volatility, IV level, rank, change and IV/RV, earnings timing;
* **surface** — the SVI surface at 30 days: level, skew, curvature, risk reversal, butterfly, term slope, fit
  error and arbitrage count (:mod:`.surface`);
* **forecast** — the HAR volatility forecast over the structure's life and the volatility risk premium against
  the implied volatility at the structure's own expiration (:mod:`.volatility`);
* **structure** — family, direction and volatility stance, days to expiration, the main leg's |delta| and
  moneyness, debit or credit, costs, Greeks and break-evens per dollar at risk, the rule's expected value and
  probability of profit (market and empirical distributions), tail loss, liquidity, and the residual edge
  against the smooth surface;
* **context** — the stock Brain's signed consensus on the underlying, and the data grade (model-priced or real),
  so the model can learn how far model-priced evidence is from the market.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np

from quantpulse.options.analytics import quote_iv
from quantpulse.options.lab.features import DayFeatures
from quantpulse.options.pricing import greeks as model_greeks
from quantpulse.options.selection import Candidate
from quantpulse.options.structures import FAMILIES

from .surface import Surface
from .volatility import har_forecast, vrp

# a fixed code per family (new families are appended: a saved model's codes never move)
FAMILY_CODES: tuple[str, ...] = (
    "long_call", "long_put", "bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread",
    "covered_call", "cash_secured_put", "long_straddle", "long_strangle", "iron_condor", "call_butterfly",
    "protective_put", "collar", "calendar", "put_butterfly", "iron_butterfly", "broken_wing_butterfly",
    "reverse_iron_condor",
)  # fmt: skip
GRADES = ("model", "recorded", "shadow", "paper")

UNDERLYING = ("ret5", "ret20", "z5", "rv20", "rv60", "atr14", "trend_50_200", "dist_sma50", "ret20_pct",
              "iv_rank", "iv_percentile", "iv_change5", "iv_rv", "event_in_life", "event_over_dte")  # fmt: skip
SURFACE = ("atm_iv", "iv_skew", "iv_curv", "rr", "bf", "term_slope", "svi_rmse", "arb_violations")
FORECAST = ("har_vol", "vrp", "vrp_ratio", "har_is_fit")
STRUCTURE = ("family", "vol_stance", "direction", "dte", "log_dte", "main_abs_delta", "moneyness_sd",
             "debit_on_risk", "spread_cost_on_risk", "fees_on_risk", "delta_on_risk", "gamma_on_risk",
             "theta_on_risk", "vega_on_risk", "pop_market", "eor_market", "eor_empirical", "pop_empirical",
             "cvar_on_risk", "reward_risk", "breakeven_sd", "liquidity", "surface_edge_on_risk", "legs",
             "multi_expiry")  # fmt: skip
CONTEXT = ("stock_score", "grade_real")
FEATURES: tuple[str, ...] = UNDERLYING + SURFACE + FORECAST + STRUCTURE + CONTEXT
CATEGORICAL = ("family",)
# features whose effect must not reverse: paying more to trade never helps (a monotone constraint in the trees)
MONOTONE: dict[str, int] = {"spread_cost_on_risk": -1, "fees_on_risk": -1}
INDEX = {name: i for i, name in enumerate(FEATURES)}

_STANCE = {"long_vol": 1.0, "short_vol": -1.0, "neutral": 0.0}
_DIRECTION = {"bullish": 1.0, "bearish": -1.0}


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _per(x: Any, risk: float) -> float:
    v = _f(x)
    return v / risk if math.isfinite(v) and risk > 0 else math.nan


def underlying_features(day: DayFeatures | None, dte: int) -> dict[str, float]:
    out = dict.fromkeys(UNDERLYING, math.nan)
    if day is None:
        return out
    out.update(ret5=_f(day.ret5), ret20=_f(day.ret20), z5=_f(day.z5), rv20=_f(day.rv20), rv60=_f(day.rv60),
               atr14=_f(day.atr14_pct), ret20_pct=_f(day.ret20_pct), iv_rank=_f(day.iv_rank),
               iv_percentile=_f(day.iv_percentile), iv_change5=_f(day.iv_change5), iv_rv=_f(day.iv_rv))  # fmt: skip
    if day.sma50 and day.sma200:
        out["trend_50_200"] = day.sma50 / day.sma200 - 1
    if day.sma50:
        out["dist_sma50"] = day.spot / day.sma50 - 1
    if day.event_days is not None:
        out["event_in_life"] = float(day.event_days <= dte)
        out["event_over_dte"] = day.event_days / max(dte, 1)
    return out


def _leg_vol_delta(cand: Candidate, spot: float, now: datetime) -> list[tuple[float, float, float]]:
    """(iv, delta, vega per vol point) per option leg, from the quote (vendor Greeks when present)."""
    out = []
    for q in cand.quotes:
        iv = q.iv if q.iv and q.iv > 0 else quote_iv(q, now)
        d, vg = q.greeks.delta, q.greeks.vega
        if (d is None or vg is None) and iv is not None:
            v = model_greeks(q.contract.kind, spot, q.contract.strike, q.contract.years(now), iv)
            d = v.delta if d is None else d
            vg = v.vega if vg is None else vg
        out.append((_f(iv), _f(d), _f(vg)))
    return out


def structure_features(
    cand: Candidate, spot: float, now: datetime, surface: Surface | None
) -> dict[str, float]:
    out = dict.fromkeys(STRUCTURE, math.nan)
    st, m = cand.structure, cand.metrics
    fam = FAMILIES.get(st.family)
    out["family"] = float(FAMILY_CODES.index(st.family)) if st.family in FAMILY_CODES else math.nan
    if fam is not None:
        out["vol_stance"] = _STANCE.get(fam.vol, 0.0)
        out["direction"] = _DIRECTION.get(fam.direction, 0.0)
    out["dte"] = float(cand.dte)
    out["log_dte"] = math.log(max(cand.dte, 1))
    out["legs"] = float(len(st.option_legs))
    out["multi_expiry"] = float(not st.single_expiry)
    risk = _f(m.get("max_loss"))
    if not (math.isfinite(risk) and risk > 0):
        return out
    legs = _leg_vol_delta(cand, spot, now)
    deltas = [abs(d) for _, d, _ in legs if math.isfinite(d)]
    if deltas:
        out["main_abs_delta"] = max(deltas)
        i = max(range(len(legs)), key=lambda j: abs(legs[j][1]) if math.isfinite(legs[j][1]) else -1)
        iv, q = legs[i][0], cand.quotes[i]
        years = q.contract.years(now)
        if math.isfinite(iv) and iv > 0 and years > 0:
            out["moneyness_sd"] = math.log(q.contract.strike / spot) / (iv * math.sqrt(years))
    out["debit_on_risk"] = _per(m.get("fill_debit"), risk)
    out["spread_cost_on_risk"] = _per(abs(_f(m.get("spread_cost"))), risk)
    out["fees_on_risk"] = _per(m.get("fees"), risk)
    g = m.get("greeks") or {}
    out["delta_on_risk"] = _per(_f(g.get("delta")) * spot, risk)
    out["gamma_on_risk"] = _per(_f(g.get("gamma")) * spot * spot * 0.0001, risk)  # a 1% move's gamma P&L
    out["theta_on_risk"] = _per(g.get("theta"), risk)
    out["vega_on_risk"] = _per(g.get("vega"), risk)
    out["pop_market"] = _f(m.get("pop"))
    out["eor_market"] = _f(m.get("expected_on_risk"))
    emp = m.get("empirical") or {}
    out["eor_empirical"] = _f(emp.get("expected_on_risk"))
    out["pop_empirical"] = _f(emp.get("pop"))
    out["cvar_on_risk"] = _per(m.get("cvar_5"), risk)
    mp = _f(m.get("max_profit"))
    out["reward_risk"] = min(mp / risk, 20.0) if math.isfinite(mp) else 20.0
    out["liquidity"] = _f(m.get("liquidity"))
    ivs = [iv for iv, _, _ in legs if math.isfinite(iv)]
    sig = float(np.mean(ivs)) if ivs else math.nan
    bes = [b for b in (m.get("breakevens") or []) if b and b > 0]
    years = cand.quotes[0].contract.years(now) if cand.quotes else 0.0
    if bes and math.isfinite(sig) and sig > 0 and years > 0:
        out["breakeven_sd"] = min(abs(math.log(b / spot)) for b in bes) / (sig * math.sqrt(years))
    if surface is not None and surface.usable:
        edge, known = 0.0, True
        for leg, q, (_, _, vg) in zip(st.option_legs, cand.quotes, legs, strict=False):
            r = surface.residual(q)
            if r is None or not math.isfinite(vg):
                known = False
                break
            edge -= leg.sign * leg.units * vg * r * 100.0  # buying a rich leg costs its excess at its vega
        out["surface_edge_on_risk"] = edge / risk if known else math.nan
    return out


def candidate_features(
    cand: Candidate,
    spot: float,
    now: datetime,
    *,
    day: DayFeatures | None,
    closes: Sequence[float],
    surface: Surface | None,
    stock_score: float | None = None,
    grade: str = "recorded",
    surface_features: Mapping[str, float | None] | None = None,
) -> dict[str, float]:
    """The full vector as a name → value mapping (``nan`` for unknown). ``surface_features`` may be passed in
    when the same surface serves many candidates (it is the same for all of one underlying's candidates)."""
    out = underlying_features(day, cand.dte)
    sf = surface_features if surface_features is not None else (surface.features() if surface else {})
    for k in SURFACE:
        out[k] = _f(sf.get(k))
    horizon = max(1, round(cand.dte * 252 / 365))
    fc = har_forecast(closes, horizon) if closes else None
    implied = None
    if surface is not None and surface.usable and cand.quotes:
        years = cand.quotes[0].contract.years(now)
        implied = surface.vol(0.0, years) if years > 0 else None
    if implied is None:
        implied = _f(cand.metrics.get("iv"))
        implied = implied if math.isfinite(implied) else None
    gap, ratio = vrp(implied, fc)
    out.update(har_vol=_f(fc.vol) if fc else math.nan, vrp=_f(gap), vrp_ratio=_f(ratio),
               har_is_fit=float(fc.method == "har") if fc else math.nan)  # fmt: skip
    out.update(structure_features(cand, spot, now, surface))
    out["stock_score"] = _f(stock_score)
    out["grade_real"] = 0.0 if grade == "model" else 1.0
    return out


def vector(features: Mapping[str, float]) -> np.ndarray:
    return np.array([_f(features.get(k)) for k in FEATURES], dtype=float)


def matrix(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(FEATURES)))
    return np.vstack([vector(r) for r in rows])
