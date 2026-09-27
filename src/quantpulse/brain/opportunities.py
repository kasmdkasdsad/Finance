"""The opportunity engine: the brain looks for ideas itself instead of waiting for a symbol.

**Pass 1 — the whole universe, cheap** (from the indicator table every cycle already computes, the stock
model's features, sectors and pair statistics on the closes):

========================  ===========================================================================
momentum_shift            momentum acceleration or deterioration ≥ 2σ across the universe, confirmed
                          by the 1-month move or a MACD cross
breakout                  a close through the 20-day high (low) on ≥ 1.5× normal volume, on the side
                          of the 50-day trend
abnormal_volume           ≥ 2.5× the 20-day average volume (session-adjusted once the first hour has
                          traded; before that, the last completed session)
valuation_dislocation     cheap (earnings / FCF / book yields ≥ 1.5σ) with at least average quality
mean_reversion            ≥ 2.2σ from the 20-day mean or RSI ≤ 25 / ≥ 80
volatility_event          today's move ≥ 3 daily σ, or 1-month volatility ≥ 1.8× its 3-month level
sector_rotation           a sector whose 1-month relative strength ranks top (bottom) while its
                          3-month ranks in the other half
relative_value            a highly correlated pair (same sector) whose price spread is ≥ 2σ from its
                          60-day relation: the cheap leg is the idea
regime_change             the market regime differs from the last cycle's
========================  ===========================================================================

**Pass 2 — the focus set, after its option chains and earnings calendars are read:** ``earnings`` (a
release within 10 days), ``catalyst`` (a large post-earnings surprise in the last 10 days) and
``unusual_options`` (turnover above open interest or an extreme put/call ratio).

Detection is not a recommendation. The best detections join the cycle's focus and go through the same
pipeline as everything else — data validation → the relevant agents → research → bull case → bear case
→ devil's advocate → consensus → portfolio fit → risk preview — and :func:`trace` records how far each got.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .types import EXECUTABLE_STATES, MARKET, DataState

# which agents speak to which kind of opportunity (used to report whether the right experts looked)
KIND_AGENTS: dict[str, tuple[str, ...]] = {
    "momentum_shift": ("momentum", "technical", "statistical"),
    "breakout": ("technical", "momentum", "volatility"),
    "abnormal_volume": ("technical", "statistical", "catalyst", "options"),
    "valuation_dislocation": ("valuation", "fundamental", "factor"),
    "mean_reversion": ("mean_reversion", "statistical", "technical"),
    "volatility_event": ("volatility", "statistical", "options"),
    "sector_rotation": ("momentum", "factor"),
    "relative_value": ("statistical", "valuation", "mean_reversion"),
    "regime_change": ("market_regime", "volatility"),
    "earnings": ("catalyst", "options", "volatility"),
    "catalyst": ("catalyst", "technical", "momentum"),
    "unusual_options": ("options", "volatility", "catalyst"),
}
MAX_PAIR_UNIVERSE = 80
PAIR_CORR = 0.8


@dataclass
class Opportunity:
    kind: str
    subject: str  # a symbol, "@market", "SECTOR:<name>" or "A/B" for a pair
    symbols: list[str]  # the tradable symbols it is about (the first is the lead)
    direction: int  # +1 a long idea, −1 a reason to avoid/reduce, 0 look closer
    strength: float  # 0…1
    headline: str
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "detected"
    stages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def lead(self) -> str | None:
        return self.symbols[0] if self.symbols else None

    def step(self, stage: str, result: str, **detail: Any) -> None:
        self.stages.append({"stage": stage, "result": result, **detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "symbols": self.symbols,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "headline": self.headline,
            "evidence": self.evidence,
            "status": self.status,
            "stages": self.stages,
        }


def _num(ind: pd.DataFrame, col: str) -> pd.Series:
    return (
        pd.to_numeric(ind[col], errors="coerce") if col in ind.columns else pd.Series(np.nan, index=ind.index)
    )


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * np.nan


def _strength(x: float, full: float) -> float:
    return float(max(0.0, min(1.0, abs(x) / full)))


# ---------------------------------------------------------------------------------------------- pass 1
def momentum_shift(ind: pd.DataFrame) -> list[Opportunity]:
    accel_z = _z(_num(ind, "mom_accel"))
    ret21, cross = _num(ind, "ret_21d"), _num(ind, "macd_cross")
    out = []
    for s, z in accel_z.dropna().items():
        if abs(z) < 2.0:
            continue
        d = 1 if z > 0 else -1
        if not (np.sign(ret21.get(s, 0)) == d or np.sign(cross.get(s, 0)) == d):
            continue
        what = "accelerating" if d > 0 else "deteriorating"
        out.append(
            Opportunity(
                "momentum_shift",
                str(s),
                [str(s)],
                d,
                _strength(z, 3.5),
                f"{s} momentum {what} ({z:+.1f}σ)",
                {"mom_accel_z": round(float(z), 2), "ret_21d": ret21.get(s)},
            )
        )
    return out


def breakout(ind: pd.DataFrame) -> list[Opportunity]:
    vol = _num(ind, "volume_ratio_1d")
    trend = _num(ind, "px_vs_sma50")
    out = []
    for s in ind.index:
        v = vol.get(s)
        if v is None or not math.isfinite(v) or v < 1.5:
            continue
        up, down = (
            bool(ind.at[s, "breakout_20"]) if "breakout_20" in ind else False,
            bool(ind.at[s, "breakdown_20"]) if "breakdown_20" in ind else False,
        )
        t = trend.get(s, 0.0)
        if up and t > 0:
            out.append(
                Opportunity(
                    "breakout",
                    str(s),
                    [str(s)],
                    1,
                    _strength(v - 1, 2.0),
                    f"{s} broke above its 20-day high on {v:.1f}× volume",
                    {"volume_ratio": round(float(v), 2)},
                )
            )
        elif down and t < 0:
            out.append(
                Opportunity(
                    "breakout",
                    str(s),
                    [str(s)],
                    -1,
                    _strength(v - 1, 2.0),
                    f"{s} broke below its 20-day low on {v:.1f}× volume",
                    {"volume_ratio": round(float(v), 2)},
                )
            )
    return out


MIN_SESSION_FRACTION = 0.15  # about the first hour: opening volume is front-loaded, so wait before judging


def abnormal_volume(
    ind: pd.DataFrame, market_open: bool, session_fraction: float | None = None
) -> list[Opportunity]:
    intraday = (
        market_open and "rel_volume" in ind.columns and (session_fraction or 0.0) >= MIN_SESSION_FRACTION
    )
    col = "rel_volume" if intraday else "volume_ratio_1d"
    ratio = _num(ind, col) + (1.0 if col == "rel_volume" else 0.0)  # rel_volume is "× − 1"
    ret = _num(ind, "ret_1d")
    out = []
    for s, r in ratio.dropna().items():
        if r >= 2.5:
            d = int(np.sign(ret.get(s, 0.0) or 0.0))
            out.append(
                Opportunity(
                    "abnormal_volume",
                    str(s),
                    [str(s)],
                    d,
                    _strength(r - 1, 4.0),
                    f"{s} trading {r:.1f}× its normal volume",
                    {"volume_multiple": round(float(r), 2), "ret_1d": ret.get(s)},
                )
            )
    return out


def mean_reversion(ind: pd.DataFrame) -> list[Opportunity]:
    z20, rsi = _num(ind, "z20"), _num(ind, "rsi14")
    out = []
    for s in ind.index:
        z, r = z20.get(s), rsi.get(s)
        low = (z is not None and z <= -2.2) or (r is not None and r <= 25)
        high = (z is not None and z >= 2.2) or (r is not None and r >= 80)
        if low or high:
            d = 1 if low else -1
            out.append(
                Opportunity(
                    "mean_reversion",
                    str(s),
                    [str(s)],
                    d,
                    _strength(z or 0, 3.5),
                    f"{s} {'oversold' if low else 'overbought'} ({(z or 0):+.1f}σ, RSI {(r or 0):.0f})",
                    {"z20": z, "rsi14": r},
                )
            )
    return out


def volatility_event(ind: pd.DataFrame, market_open: bool) -> list[Opportunity]:
    move, ratio = _num(ind, "move_z"), _num(ind, "vol_ratio")
    out = []
    for s in ind.index:
        m, v = move.get(s), ratio.get(s)
        if market_open and m is not None and math.isfinite(m) and abs(m) >= 3:
            out.append(
                Opportunity(
                    "volatility_event",
                    str(s),
                    [str(s)],
                    0,
                    _strength(m, 5),
                    f"{s} moved {m:+.1f} daily σ today",
                    {"move_z": round(float(m), 2)},
                )
            )
        elif v is not None and math.isfinite(v) and v >= 1.8:
            out.append(
                Opportunity(
                    "volatility_event",
                    str(s),
                    [str(s)],
                    0,
                    _strength(v - 1, 2),
                    f"{s} volatility {v:.1f}× its 3-month level",
                    {"vol_ratio": round(float(v), 2)},
                )
            )
    return out


def valuation_dislocation(features: pd.DataFrame | None, universe: Sequence[str]) -> list[Opportunity]:
    if features is None or features.empty:
        return []
    value_cols = [c for c in ("earnings_yield", "fcf_yield", "book_to_market") if c in features.columns]
    quality_cols = [c for c in ("gross_profitability", "roe") if c in features.columns]
    if len(value_cols) < 2:
        return []
    f = features.apply(pd.to_numeric, errors="coerce")
    value = pd.concat(
        [_z(f[c].clip(f[c].quantile(0.02), f[c].quantile(0.98))) for c in value_cols], axis=1
    ).mean(axis=1)
    quality = (
        pd.concat([_z(f[c]) for c in quality_cols], axis=1).mean(axis=1) if quality_cols else value * 0.0
    )
    out = []
    for s in set(universe) & set(value.index):
        v, q = value.get(s), quality.get(s)
        if v is not None and math.isfinite(v) and v >= 1.5 and (q is None or not math.isfinite(q) or q >= 0):
            out.append(
                Opportunity(
                    "valuation_dislocation",
                    s,
                    [s],
                    1,
                    _strength(v, 3),
                    f"{s} cheap ({v:+.1f}σ on yields) with {'good' if (q or 0) > 0.5 else 'average'} quality",
                    {
                        "value_z": round(float(v), 2),
                        "quality_z": None if q is None or not math.isfinite(q) else round(float(q), 2),
                    },
                )
            )
    return out


def sector_rotation(ind: pd.DataFrame, sectors: dict[str, str], held: Sequence[str]) -> list[Opportunity]:
    rs1, rs3 = _num(ind, "rs_1m"), _num(ind, "rel_strength")
    frame = pd.DataFrame({"rs1": rs1, "rs3": rs3, "sector": pd.Series(sectors).reindex(ind.index)}).dropna()
    frame = frame[frame["sector"] != "ETF"]
    groups = frame.groupby("sector")
    med = groups[["rs1", "rs3"]].median()[groups.size() >= 4]
    if len(med) < 4:
        return []
    r1, r3 = med["rs1"].rank(ascending=False), med["rs3"].rank(ascending=False)
    half = len(med) / 2
    out = []
    for sector in med.index:
        members = frame[frame["sector"] == sector].sort_values("rs1", ascending=False)
        if r1[sector] <= 2 and r3[sector] > half:
            leaders = [str(s) for s in members.index[:2]]
            out.append(
                Opportunity(
                    "sector_rotation",
                    f"SECTOR:{sector}",
                    leaders,
                    1,
                    0.6,
                    f"{sector} turning up: 1-month relative strength rank {int(r1[sector])} vs 3-month {int(r3[sector])}",
                    {"rank_1m": int(r1[sector]), "rank_3m": int(r3[sector])},
                )
            )
        elif r1[sector] > len(med) - 2 and r3[sector] <= half:
            exposed = [str(s) for s in members.index if s in held]
            out.append(
                Opportunity(
                    "sector_rotation",
                    f"SECTOR:{sector}",
                    exposed,
                    -1,
                    0.5,
                    f"{sector} rolling over: 1-month relative strength rank {int(r1[sector])} vs 3-month {int(r3[sector])}",
                    {"rank_1m": int(r1[sector]), "rank_3m": int(r3[sector])},
                )
            )
    return out


def relative_value(close: pd.DataFrame, ind: pd.DataFrame, sectors: dict[str, str]) -> list[Opportunity]:
    liquid = _num(ind, "adv_dollar").dropna().sort_values(ascending=False)
    names = [s for s in liquid.index[:MAX_PAIR_UNIVERSE] if s in close.columns and sectors.get(s) != "ETF"]
    if len(names) < 2:
        return []
    logp = np.log(close[names].iloc[-121:].astype(float)).dropna(axis=1)
    rets = logp.diff().dropna()
    if len(rets) < 100:
        return []
    corr = rets.corr()
    out: list[Opportunity] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            same = sectors.get(a) and sectors.get(a) == sectors.get(b)
            if corr.at[a, b] < PAIR_CORR or (sectors and not same):
                continue
            x, y = logp[b].to_numpy(), logp[a].to_numpy()
            beta = float(np.polyfit(x, y, 1)[0])
            spread = y - beta * x
            window = spread[-60:]
            sd = float(window.std(ddof=1))
            if sd <= 0:
                continue
            z = float((spread[-1] - window.mean()) / sd)
            if abs(z) >= 2.0:
                cheap, rich = (a, b) if z < 0 else (b, a)
                out.append(
                    Opportunity(
                        "relative_value",
                        f"{cheap}/{rich}",
                        [str(cheap), str(rich)],
                        1,
                        _strength(z, 3.5),
                        f"{cheap} cheap vs {rich} ({abs(z):.1f}σ from their 60-day relation, correlation {corr.at[a, b]:.2f})",
                        {
                            "spread_z": round(z, 2),
                            "correlation": round(float(corr.at[a, b]), 3),
                            "hedge_ratio": round(beta, 3),
                        },
                    )
                )
    return sorted(out, key=lambda o: -o.strength)[:10]


def regime_change(current: str | None, previous: str | None, trend_score: float | None) -> list[Opportunity]:
    if current is None or previous is None or current == previous:
        return []
    order = ["risk_off", "bearish", "high_volatility", "neutral", "bullish"]
    d = (
        int(np.sign(order.index(current) - order.index(previous)))
        if current in order and previous in order
        else 0
    )
    return [
        Opportunity(
            "regime_change",
            MARKET,
            [],
            d,
            0.8,
            f"market regime changed: {previous} → {current}",
            {"from": previous, "to": current, "trend_score": trend_score},
        )
    ]


def scan_universe(
    ind: pd.DataFrame,
    close: pd.DataFrame,
    *,
    market_open: bool,
    sectors: dict[str, str],
    held: Sequence[str],
    features: pd.DataFrame | None,
    regime: str | None,
    previous_regime: str | None,
    trend_score: float | None,
    exclude: Sequence[str] = (),
    session_fraction: float | None = None,
) -> list[Opportunity]:
    stocks = ind.drop(index=[s for s in exclude if s in ind.index])
    found = [
        *momentum_shift(stocks),
        *breakout(stocks),
        *abnormal_volume(stocks, market_open, session_fraction),
        *mean_reversion(stocks),
        *volatility_event(stocks, market_open),
        *valuation_dislocation(features, list(stocks.index)),
        *sector_rotation(stocks, sectors, held),
        *relative_value(close, stocks, sectors),
        *regime_change(regime, previous_regime, trend_score),
    ]
    return sorted(found, key=lambda o: -o.strength)


# ---------------------------------------------------------------------------------------------- pass 2
def scan_focus(
    focus: Sequence[str], options: dict[str, dict[str, Any]], events: dict[str, dict[str, Any]]
) -> list[Opportunity]:
    out: list[Opportunity] = []
    for s in focus:
        e = events.get(s) or {}
        days = e.get("days_to_next")
        if days is not None and 0 <= days <= 10:
            move = e.get("typical_move")
            out.append(
                Opportunity(
                    "earnings",
                    s,
                    [s],
                    0,
                    0.5 + 0.05 * (10 - days),
                    f"{s} reports in {days} days" + (f" (typical move ±{move:.1%})" if move else ""),
                    {"days_to_next": days, "typical_move": move},
                )
            )
        since, abnormal, typical = e.get("days_since_last"), e.get("last_abnormal"), e.get("typical_move")
        if (
            since is not None
            and since <= 10
            and abnormal is not None
            and (abs(abnormal) >= 0.05 or (typical and abs(abnormal) >= 2 * typical))
        ):
            out.append(
                Opportunity(
                    "catalyst",
                    s,
                    [s],
                    1 if abnormal > 0 else -1,
                    _strength(abnormal, 0.15),
                    f"{s} {abnormal:+.1%} beyond the market after its release {since} days ago",
                    {"abnormal": abnormal, "days_since": since},
                )
            )
        m = options.get(s) or {}
        turnover, pcv = m.get("volume_oi"), m.get("pc_volume")
        if (turnover is not None and turnover >= 1.0) or (pcv is not None and (pcv >= 2.0 or pcv <= 0.4)):
            d = 0 if pcv is None else (-1 if pcv >= 2.0 else 1 if pcv <= 0.4 else 0)
            out.append(
                Opportunity(
                    "unusual_options",
                    s,
                    [s],
                    d,
                    _strength(turnover or 1.0, 2.0),
                    f"{s} unusual options activity (turnover {turnover or 0:.1f}× open interest, put/call {pcv if pcv is not None else float('nan'):.2f})",
                    {"volume_oi": turnover, "pc_volume": pcv},
                )
            )
    return out


# ---------------------------------------------------------------------------------------------- pipeline trace
def trace(
    opportunities: Sequence[Opportunity],
    *,
    focus: Sequence[str],
    states: dict[str, DataState],
    market_open: bool,
    opinions: dict[str, list[Any]],
    consensus: dict[str, Any],
    debates: dict[str, Any],
    proposals: dict[str, Any],
) -> None:
    """Record, for each opportunity, how far it went through the pipeline and why it stopped."""
    for o in opportunities:
        o.stages = [{"stage": "detection", "result": o.headline}]
        lead = o.lead
        if lead is None:
            o.status = "context"  # market-level (a regime change) or a sector with no exposure
            o.step("research", "market context for every agent this cycle")
            continue
        if lead not in focus:
            o.status = "not_analysed"
            o.step("focus", "outside this cycle's focus budget (stronger ideas came first)")
            continue
        state = states.get(lead, DataState.UNAVAILABLE)
        if market_open and state not in EXECUTABLE_STATES:
            o.status = "rejected_data"
            o.step("data_validation", f"data {state.value}: not trustworthy enough to act on")
            continue
        o.step(
            "data_validation",
            f"data {state.value}" + ("" if market_open else " (market closed: research only)"),
        )
        ran = {op.agent_id for op in opinions.get(lead, []) if op.stance.value != "abstain"}
        wanted = KIND_AGENTS.get(o.kind, ())
        o.step(
            "relevant_agents",
            f"{len(ran & set(wanted))} of {len(wanted)} relevant agents gave a view",
            heard=sorted(ran & set(wanted)),
            missing=sorted(set(wanted) - ran),
        )
        research = [op for op in opinions.get(lead, []) if op.agent_id == "research"]
        if research:
            o.step("research", research[0].thesis)
        d = debates.get(lead)
        if d is not None:
            o.step("bull_case", d.bull[0].text if d.bull else "no argument for")
            o.step("bear_case", d.bear[0].text if d.bear else "no argument against")
            o.step(
                "devils_advocate", d.verdict, objections=[x.text for x in d.objections if x.severity != "low"]
            )
        c = consensus.get(lead)
        if c is None:
            o.status = "no_view"
            o.step("consensus", "no forecasting agent had a view")
            continue
        o.step(
            "consensus", "I don't know" if c.unknown else f"{c.stance.value}, confidence {c.confidence:.2f}"
        )
        p = proposals.get(lead)
        if p is None:
            o.status = "no_action"
            continue
        fit = p.fit or {}
        if fit:
            o.step("portfolio_fit", "fits" if fit.get("ok") else "poor fit", notes=fit.get("notes", []))
        o.step("risk_preview", p.status if p.is_trade else "no trade proposed", action=p.action.value)
        o.status = p.status if p.is_trade else p.action.value
