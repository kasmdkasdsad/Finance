"""Counterfactuals: what would the other choices have done?

For every trade actually taken, the same thesis is expressed other ways at the same moment — a long call
instead of a spread, a longer or shorter expiration, a different delta, a put spread, the stock, no trade —
and each is valued at the exit with the same data source (recorded quotes when they exist, the model
otherwise — labelled). This separates "the direction was right" from "the structure was right": a directional
thesis that was correct in a structure that lost is a structure lesson, not a direction lesson.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from quantpulse.options.fills import ExecutionModel, leg_fill
from quantpulse.options.lab.backtest import one_sided
from quantpulse.options.lab.chains import ChainSource, close_time
from quantpulse.options.selection import Spec, build
from quantpulse.options.structures import FAMILIES, Structure


@dataclass(frozen=True, slots=True)
class Alternative:
    label: str
    spec: Spec | None  # None: the stock (100 shares) or no trade


def alternatives(
    family: str, direction: str, dte: int, delta: float, width: float | None
) -> list[Alternative]:
    near = (max(1, dte - 12), max(2, dte - 4))
    far = (dte + 10, dte + 30)
    base = Spec(family, max(1, dte - 5), dte + 5, delta, width)
    alts = [
        Alternative("shorter_expiration", replace(base, dte_min=near[0], dte_max=near[1])),
        Alternative("longer_expiration", replace(base, dte_min=far[0], dte_max=far[1])),
        Alternative("higher_delta", replace(base, delta_target=min(0.9, delta + 0.15))),
        Alternative("lower_delta", replace(base, delta_target=max(0.1, delta - 0.15))),
    ]
    bullish = direction in ("bullish", "income")
    if bullish:
        alts += [Alternative("long_call", replace(base, family="long_call", delta_target=0.5, width_pct=None)),
                 Alternative("call_spread", replace(base, family="bull_call_spread", delta_target=0.5, width_pct=0.05)),
                 Alternative("put_spread", replace(base, family="bull_put_spread", delta_target=0.3, width_pct=0.05))]  # fmt: skip
    elif direction == "bearish":
        alts += [Alternative("long_put", replace(base, family="long_put", delta_target=0.5, width_pct=None)),
                 Alternative("put_spread", replace(base, family="bear_put_spread", delta_target=0.5, width_pct=0.05)),
                 Alternative("call_spread", replace(base, family="bear_call_spread", delta_target=0.3, width_pct=0.05))]  # fmt: skip
    alts += [Alternative("stock", None), Alternative("no_trade", None)]
    return [
        a
        for a in alts
        if a.spec is None or a.spec.family != family or a.label.endswith(("expiration", "delta"))
    ]


def _value(s: Structure, source: ChainSource, underlying: str, day: date, spot: float, model: ExecutionModel,
           closing: bool) -> float | None:  # fmt: skip
    total = 0.0
    for leg in s.legs:
        if leg.contract is None:
            total += leg.sign * leg.units * spot
            continue
        if day >= leg.contract.expiration:
            total += leg.sign * leg.units * leg.contract.intrinsic(spot)
            continue
        q = source.quote(leg.contract.symbol, underlying, day)
        if q is None or q.ask is None:
            return None
        side = -leg.sign if closing else leg.sign
        if q.two_sided:
            total += leg.sign * leg.units * leg_fill(side, q.bid or 0.0, q.ask, model)
        elif closing:
            total += leg.sign * leg.units * one_sided(leg.sign, q)
        else:
            return None  # cannot open against a one-sided market
    return total


def evaluate(trade: dict[str, Any], source: ChainSource, *, model: ExecutionModel = ExecutionModel.REALISTIC,
             delta: float = 0.3, width: float | None = 0.05) -> list[dict[str, Any]]:  # fmt: skip
    """Every alternative's P&L (one unit, and per dollar at risk) against the chosen trade's."""
    u = trade["underlying"]
    d0, d1 = date.fromisoformat(trade["entry_date"]), date.fromisoformat(trade["exit_date"])
    s0, s1 = source.spot(u, d0), source.spot(u, d1)
    chosen = trade["pnl"] / max(trade["qty"], 1)
    chosen_risk = (trade.get("max_loss") or 1.0) / max(trade["qty"], 1)
    if s0 is None or s1 is None:
        return []
    chain = source.chain(u, d0)
    out: list[dict[str, Any]] = []
    fam = trade["family"]
    direction = trade.get("direction") or FAMILIES[fam].direction
    for alt in alternatives(fam, direction, trade["dte_entry"], delta, width):
        if alt.label == "no_trade":
            pnl, risk, desc = 0.0, 0.0, "no trade"
        elif alt.label == "stock":
            sign = -1 if direction == "bearish" else 1
            pnl = sign * 100 * (s1 - s0)
            risk, desc = 100 * s0, f"{'short' if sign < 0 else 'long'} 100 shares"
        else:
            if chain is None or alt.spec is None:
                continue
            cands = build(alt.spec, chain.quotes, s0, close_time(d0))
            if not cands:
                continue
            c = min(cands, key=lambda c: abs(c.dte - trade["dte_entry"]))
            opened = _value(c.structure, source, u, d0, s0, model, closing=False)
            closed = _value(c.structure, source, u, d1, s1, model, closing=True)
            if opened is None or closed is None:
                continue
            pnl = closed - opened
            risk = c.structure.max_loss()
            desc = c.structure.describe()
        out.append({
            "alternative": alt.label,
            "structure": desc,
            "pnl": round(pnl, 2),
            "pnl_on_risk": round(pnl / risk, 4) if risk and risk != float("inf") else None,
            "chosen_pnl": round(chosen, 2),
            "chosen_on_risk": round(chosen / chosen_risk, 4) if chosen_risk else None,
            "better_than_chosen": pnl > chosen,
            "data_source": source.grade,
        })  # fmt: skip
    return out


def verdict(trade: dict[str, Any], cfs: list[dict[str, Any]]) -> dict[str, Any]:
    """Direction right? Structure right? The best alternative per dollar at risk."""
    move = trade.get("underlying_return")
    direction = trade.get("direction")
    right = (
        None
        if move is None or direction not in ("bullish", "bearish")
        else (move > 0) == (direction == "bullish")
    )
    comparable = [c for c in cfs if c["pnl_on_risk"] is not None and c["alternative"] != "no_trade"]
    best = max(comparable, key=lambda c: c["pnl_on_risk"], default=None)
    chosen = trade["pnl"] / max(trade.get("max_loss") or 1.0, 1e-9)
    structure_right = best is None or chosen >= best["pnl_on_risk"] - 0.05
    lesson = None
    if right and trade["pnl"] < 0 and best is not None and best["pnl_on_risk"] > 0:
        lesson = (f"the directional thesis was correct but the {trade['family']} lost; {best['alternative']} would "
                  f"have made {best['pnl_on_risk']:+.2f} per $ at risk")  # fmt: skip
    return {"direction_correct": right, "structure_correct": structure_right,
            "best_alternative": best["alternative"] if best else None,
            "best_alternative_on_risk": best["pnl_on_risk"] if best else None, "lesson": lesson}  # fmt: skip
