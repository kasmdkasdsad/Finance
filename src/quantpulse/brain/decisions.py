"""From consensus to proposed portfolio actions — and the deterministic risk engine's verdict on each.

Position management (holdings first):

* at its stop (portfolio agent) or a broken thesis → CLOSE (a protective exit);
* in the last half hour, earnings before the next session opens → DE_RISK half (the overnight gap risk);
* overweight → REDUCE to the position limit;
* at the target of a calibrated thesis → REDUCE half, once (take profits; no target is ever invented);
* a confident bearish consensus → REDUCE (half) or CLOSE (strongly bearish);
* a confident bullish consensus below its target weight → INCREASE toward it;
* otherwise HOLD — including when the consensus is *unknown*.

Then the book as a whole (one trim per rule per cycle, each converging — it stops once the book is back
inside): two holdings that are nearly the same bet (return correlation ≥ 0.85) and together above the
position limit → REDUCE the weaker by the excess; a sector above the portfolio agent's limit → REDUCE its
weakest holding by the excess; portfolio volatility above ``PORTFOLIO_VOL_CAP`` → DE_RISK a quarter of the
largest risk contributor. Only holdings the plan would otherwise HOLD are trimmed.

New positions: a confident bullish consensus on a focus symbol, with no data veto, outside a risk-off
market, with a free position slot and cash that is not borrowed → BUY at a volatility-budgeted target weight
scaled by confidence; at most ``max_new`` per cycle, best first. Each order is sized within the risk
engine's per-order limit (a bigger target is reached over later cycles). Everything else is WATCH or
NO_ACTION, with the reason — the brain is allowed (and expected) to do nothing.

Every proposed trade is then evaluated by the existing :class:`~quantpulse.services.trading_risk.RiskBook`
— the exact limits and checks that guard real orders — as a **preview**: sells first (committed so buys see
them), then buys. This module never sends an order: in ``paper_execution`` the decisions are executed
afterwards by the trading service (:mod:`quantpulse.brain.execution`), which checks them again on fresh data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np

from quantpulse.core.market_calendar import earnings_overnight
from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_data import assess_quote
from quantpulse.services.trading_risk import OrderIntent, QuoteCheck, RiskBook

from .agents.portfolio import BETA_LIMIT, SECTOR_LIMIT
from .consensus import Consensus
from .context import BrainContext
from .debate import Debate
from .types import BUYING, EXECUTABLE_STATES, MARKET, SELLING, Action, BrainMode, Stance

TRADES = BUYING | SELLING
MAX_RISK_SHARE = 0.40  # a new name carrying more of the portfolio's risk than this is a poor fit
MAX_VOL_RISE = 1.3  # nor one that raises portfolio volatility by more than 30% …
VOL_FLOOR_FOR_RISE = 0.20  # … to above 20% a year
REPLACE_MARGIN = 0.20  # a candidate this much stronger (score × confidence) may replace a fading holding
SAME_BET_CORR = 0.85  # two holdings this correlated are one bet: together they respect the position limit
PORTFOLIO_VOL_CAP = 0.35  # annualised: above this the book is trimmed at its largest risk contributor
PROFIT_TAKEN = 0.75  # a position already cut to this share of its entry has taken its profit
# a bullish holding is topped up only below this share of its target weight. Nearer, it is left alone: a
# no-trade band, the mirror of the trim at 1.5× the target. Without it, every price tick would buy a share.
TOP_UP_BELOW = 0.75
# the defensive posture trims a third of a holding at most once in this long: the pace it had with
# 30-minute cycles, so 5-minute cycles do not cut a position to almost nothing within the hour
DERISK_EVERY = timedelta(minutes=30)


@dataclass
class Proposal:
    subject: str
    action: Action
    confidence: float
    reasons: list[str]
    quantity: float | None = None
    est_price: float | None = None
    current_weight: float | None = None
    target_weight: float | None = None
    consensus: Consensus | None = None
    blocked_by: list[str] = field(default_factory=list)
    risk: dict[str, Any] = field(default_factory=dict)
    risk_approved: bool | None = None
    status: str = "proposed"
    fit: dict[str, Any] = field(default_factory=dict)  # portfolio fit of a new position
    memory: list[str] = field(default_factory=list)  # what memory says about this decision (context only)
    protective: bool = False  # an exit at a stop or a broken thesis (never held back by an entry halt)
    execution: dict[str, Any] = field(default_factory=dict)  # what happened to it (paper_execution)
    entry: dict[str, Any] = field(default_factory=dict)  # a buy's thesis, stop, target, horizon, agents

    @property
    def is_trade(self) -> bool:
        return self.action in TRADES and bool(self.quantity)

    @property
    def notional(self) -> float | None:
        if self.quantity is None or self.est_price is None:
            return None
        return self.quantity * self.est_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "action": self.action.value,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "quantity": self.quantity,
            "est_price": self.est_price,
            "notional": round(self.notional, 2) if self.notional is not None else None,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "blocked_by": self.blocked_by,
            "risk": self.risk,
            "risk_approved": self.risk_approved,
            "status": self.status,
            "fit": self.fit,
            "memory": self.memory,
            "protective": self.protective,
            "execution": self.execution,
            "entry": self.entry,
        }


def _shares(qty: float, whole: bool) -> float:
    return float(math.floor(qty)) if whole else round(qty, 6)


def target_weight(
    ctx: BrainContext, symbol: str, confidence: float, vol_budget: float, vol_floor: float
) -> float:
    """A volatility-budgeted weight (vol_budget / risk vol), capped by the position limit, scaled by
    confidence. Risk vol is the larger of the strategy's own measure and the volatility agent's forecast."""
    forecast = (ctx.working.facts.get("vol_forecast") or {}).get(symbol)
    vol = max(ctx.ind(symbol, "risk_vol") or ctx.ind(symbol, "rv63") or vol_floor, forecast or 0.0)
    base = vol_budget / max(vol, vol_floor)
    return round(min(base, ctx.limits.max_position_pct) * min(max(confidence, 0.0), 1.0), 4)


def portfolio_fit(ctx: BrainContext, symbol: str, add_weight: float) -> dict[str, Any]:
    """Would a new position fit the book? Nearly the same bet as a holding (return correlation ≥ 0.85
    over six months) or a sector above the portfolio agent's limit does not fit; high portfolio beta is noted."""
    notes: list[str] = []
    ok = True
    cons = ctx.working.facts.get("portfolio_constraints") or {}
    held = [h for h in ctx.held if h in ctx.close.columns and h != symbol]
    max_corr: float | None = None
    with_: str | None = None
    if held and symbol in ctx.close.columns:
        rets = np.log(ctx.close[[symbol, *held]].iloc[-121:].astype(float)).diff().dropna()
        if len(rets) >= 60:
            corr = rets.corr()[symbol].drop(symbol).dropna()
            if len(corr):
                with_, max_corr = str(corr.idxmax()), float(corr.max())
                if max_corr >= 0.85:
                    ok = False
                    notes.append(f"nearly the same bet as {with_} (return correlation {max_corr:.2f})")
                elif max_corr >= 0.7:
                    notes.append(f"correlated with {with_} ({max_corr:.2f})")
    sector = ctx.sectors.get(symbol)
    after = float((cons.get("sector_weights") or {}).get(sector, 0.0)) + add_weight if sector else None
    if sector and sector not in ("unknown", "ETF") and after is not None and after > SECTOR_LIMIT:
        ok = False
        notes.append(f"{sector} would be {after:.0%} of equity (limit {SECTOR_LIMIT:.0%})")
    beta = ctx.ind(symbol, "beta")
    port_beta = cons.get("beta")
    beta_after = (port_beta or 0.0) + add_weight * beta if beta is not None else None
    if beta_after is not None and beta_after > BETA_LIMIT:
        notes.append(f"portfolio beta would reach {beta_after:.2f}")
    risk = portfolio_risk(ctx, symbol, add_weight)
    if risk.get("risk_share") is not None and risk["risk_share"] > MAX_RISK_SHARE and len(held) >= 2:
        ok = False
        notes.append(
            f"it would carry {risk['risk_share']:.0%} of the portfolio's risk at a {add_weight:.0%} weight"
        )
    vb, va = risk.get("vol_before"), risk.get("vol_after")
    if vb and va and va > MAX_VOL_RISE * vb and va > VOL_FLOOR_FOR_RISE:
        ok = False
        notes.append(f"it would raise portfolio volatility from {vb:.1%} to {va:.1%}")
    return {
        "ok": ok,
        "notes": notes,
        "max_corr": round(max_corr, 3) if max_corr is not None else None,
        "max_corr_with": with_,
        "sector": sector,
        "sector_weight_after": round(after, 4) if after is not None else None,
        "beta_after": round(beta_after, 3) if beta_after is not None else None,
        **risk,
    }


def portfolio_risk(ctx: BrainContext, symbol: str, add_weight: float) -> dict[str, Any]:
    """The portfolio before and after adding ``add_weight`` of ``symbol``: annualised volatility (six months of
    daily returns), the new name's share of the portfolio's risk (its marginal contribution), concentration
    (Herfindahl of the invested weights) and the momentum tilt. Empty when the history is too short."""
    held = {h: ctx.portfolio.weight(h) for h in ctx.held if h in ctx.close.columns and h != symbol}
    if symbol not in ctx.close.columns:
        return {}
    names = [*held, symbol]
    rets = np.log(ctx.close[names].iloc[-127:].astype(float)).diff().dropna(how="all").fillna(0.0)
    if len(rets) < 60:
        return {}
    cov = rets.cov().to_numpy() * 252
    before = np.array([*held.values(), 0.0])
    after = before.copy()
    after[-1] += add_weight

    def vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ cov @ w, 0.0)))

    vb, va = vol(before), vol(after)
    share = float(after[-1] * (cov @ after)[-1] / (va**2)) if va > 0 else None

    def hhi(w: np.ndarray) -> float | None:
        total = float(w.sum())
        return float(((w / total) ** 2).sum()) if total > 0 else None

    def tilt(w: np.ndarray) -> float | None:
        mom = [ctx.ind(n, "mom_3m") for n in names]
        if any(m is None for m in mom) or float(w.sum()) <= 0:
            return None
        return float(np.dot(w, np.array(mom, dtype=float)) / w.sum())

    def r(x: float | None, n: int = 4) -> float | None:
        return round(x, n) if x is not None else None

    return {
        "vol_before": r(vb) if held else None,
        "vol_after": r(va),
        "risk_share": r(share),
        "hhi_before": r(hhi(before)),
        "hhi_after": r(hhi(after)),
        "momentum_tilt_before": r(tilt(before)),
        "momentum_tilt_after": r(tilt(after)),
    }


def plan(
    ctx: BrainContext,
    consensus: dict[str, Consensus],
    *,
    min_confidence: float,
    max_new: int,
    vol_budget: float,
    vol_floor: float,
    earnings_caution_days: int = 0,
    debates: dict[str, Debate] | None = None,
) -> list[Proposal]:
    out: list[Proposal] = []
    constraints = ctx.working.facts.get("portfolio_constraints") or {}
    event_risk = ctx.working.facts.get("event_risk") or {}
    situation = ctx.working.facts.get("situation") or {"posture": "normal", "risk_scale": 1.0, "reasons": []}
    posture, risk_scale = situation["posture"], float(situation["risk_scale"])
    posture_why = f"{posture} posture: " + "; ".join(situation.get("reasons") or [])
    if posture == "cautious" and max_new > 0:
        max_new = max(1, max_new // 2)
    debates = debates or {}

    def challenged(symbol: str) -> str | None:
        d = debates.get(symbol)
        if d is None or not d.challenged:
            return None
        return "devil's advocate: " + "; ".join(o.text for o in d.objections if o.severity == "high")

    def sized(symbol: str, confidence: float) -> float:
        return round(target_weight(ctx, symbol, confidence, vol_budget, vol_floor) * risk_scale, 4)

    def before_earnings(symbol: str) -> str | None:
        days = (event_risk.get(symbol) or {}).get("days_to_earnings")
        if days is not None and 0 <= days <= earnings_caution_days:
            return f"earnings in {days} day(s): no new risk before the release"
        return None

    regime = ctx.working.facts.get("regime")
    eq = ctx.portfolio.equity
    cash = float(constraints.get("spendable_cash", 0.0))  # shared by increases and new buys
    per_order = ctx.limits.max_order_notional  # sized to fit; a larger target is reached over cycles
    hints = {
        s: o.meta.get("action_hint")
        for s in ctx.held
        for o in ctx.working.opinions.get(s, [])
        if o.agent_id == "portfolio"
    }
    theses = ctx.working.facts.get("thesis_checks")  # the Alpaca account's position theses (paper_execution)
    registry = ctx.working.facts.get("theses") or {}  # the theses themselves (entry, stop, target)
    near_close = ctx.working.facts.get("near_close")  # the last half hour of the session
    recently_derisked = set(ctx.working.facts.get("recently_derisked") or ())  # see DERISK_EVERY

    def overnight(symbol: str) -> str | None:
        """Earnings before the next session, reviewed in the last half hour (once a day per holding)."""
        if not near_close or symbol in (near_close.get("derisked") or []):
            return None
        days = (event_risk.get(symbol) or {}).get("days_to_earnings")
        if earnings_overnight(ctx.as_of, days):
            return (
                f"overnight: earnings before the next session ({days} day(s)) — halve the position before "
                f"the close ({near_close.get('minutes_to_close')} min left)"
            )
        return None

    def take_profit(symbol: str, price: float, qty: float) -> str | None:
        """At the target of a thesis with a calibrated target (none is invented), and not yet taken."""
        t = registry.get(symbol) or {}
        target, entry_qty = t.get("target_price"), t.get("entry_qty")
        if not target or not entry_qty or price < target or qty <= PROFIT_TAKEN * entry_qty:
            return None
        exp = t.get("expected_return")
        return (
            f"take profit: ${price:,.2f} reached its target ${target:,.2f}"
            + (f" (calibrated expected return {exp:+.1%})" if exp is not None else "")
            + " — sell half, keep the rest under the thesis"
        )

    # --- holdings
    for s in ctx.held:
        pos = ctx.portfolio.positions[s]
        c = consensus.get(s)
        w = ctx.portfolio.weight(s)
        price = ctx.price(s) or pos.current_price
        whole = float(pos.qty).is_integer()
        hint = hints.get(s)
        base: dict[str, Any] = {
            "subject": s,
            "est_price": price,
            "current_weight": round(w, 4),
            "consensus": c,
        }
        if hint == "close":
            out.append(
                Proposal(
                    action=Action.CLOSE,
                    confidence=1.0,
                    quantity=pos.qty,
                    target_weight=0.0,
                    reasons=[f"at its stop ({pos.unrealized_plpc:+.1%})"],
                    protective=True,
                    **base,
                )
            )
        elif theses and (theses.get(s) or {}).get("status") == "broken":
            out.append(
                Proposal(
                    action=Action.CLOSE,
                    confidence=1.0,
                    quantity=pos.qty,
                    target_weight=0.0,
                    reasons=["thesis broken: " + "; ".join(theses[s]["reasons"])],
                    protective=True,
                    **base,
                )
            )
        elif overnight(s):
            qty = _shares(pos.qty / 2, whole) or pos.qty
            out.append(
                Proposal(
                    action=Action.DE_RISK,
                    confidence=1.0,
                    quantity=qty,
                    target_weight=round(w / 2, 4),
                    reasons=[overnight(s) or "", *(c.reasons if c else [])],
                    **base,
                )
            )
        elif hint == "reduce" and eq > 0 and price:
            excess = (w - ctx.limits.max_position_pct) * eq / price
            qty = min(float(math.ceil(excess)) if whole else round(excess, 6), pos.qty)
            out.append(
                Proposal(
                    action=Action.REDUCE,
                    confidence=1.0,
                    quantity=qty,
                    target_weight=ctx.limits.max_position_pct,
                    reasons=[f"overweight: {w:.1%} vs the {ctx.limits.max_position_pct:.0%} limit"],
                    **base,
                )
            )
        elif price and take_profit(s, price, pos.qty):
            qty = _shares(pos.qty / 2, whole) or pos.qty
            out.append(
                Proposal(
                    action=Action.REDUCE,
                    confidence=1.0,
                    quantity=qty,
                    target_weight=round(w / 2, 4),
                    reasons=[take_profit(s, price, pos.qty) or "", *(c.reasons if c else [])],
                    **base,
                )
            )
        elif (
            c is not None
            and c.actionable_view
            and c.stance is Stance.BEARISH
            and c.confidence >= min_confidence
        ):
            if c.score <= -0.5:
                out.append(
                    Proposal(
                        action=Action.CLOSE,
                        confidence=c.confidence,
                        quantity=pos.qty,
                        target_weight=0.0,
                        reasons=[f"strongly bearish consensus ({c.score:+.2f})", *c.reasons],
                        **base,
                    )
                )
            else:
                qty = _shares(pos.qty / 2, whole) or pos.qty
                out.append(
                    Proposal(
                        action=Action.REDUCE,
                        confidence=c.confidence,
                        quantity=qty,
                        target_weight=round(w / 2, 4),
                        reasons=[f"bearish consensus ({c.score:+.2f})", *c.reasons],
                        **base,
                    )
                )
        elif posture == "defensive" and not (
            c is not None and c.actionable_view and c.stance is Stance.BULLISH
        ):
            if s in recently_derisked:
                out.append(
                    Proposal(
                        action=Action.HOLD,
                        confidence=1.0,
                        target_weight=round(w, 4),
                        reasons=[
                            f"already trimmed for the defensive posture in the last "
                            f"{DERISK_EVERY.seconds // 60} minutes: the next third waits",
                            posture_why,
                        ],
                        **base,
                    )
                )
            else:
                out.append(
                    Proposal(
                        action=Action.DE_RISK,
                        confidence=1.0,
                        quantity=_shares(pos.qty / 3, whole) or pos.qty,
                        target_weight=round(w * 2 / 3, 4),
                        reasons=[f"trim a third: {posture_why}", *(c.reasons if c else ["no consensus"])],
                        **base,
                    )
                )
        elif (
            c is not None
            and c.actionable_view
            and c.stance is Stance.BULLISH
            and c.confidence >= min_confidence
            and eq > 0
            and price
            and w > 1.5 * sized(s, c.confidence) > 0
            and (w - sized(s, c.confidence)) * eq >= ctx.limits.min_order_notional
        ):
            tw = sized(s, c.confidence)
            trim = (w - tw) * eq / price
            qty = min(float(math.floor(trim)) if whole else round(trim, 6), pos.qty)
            out.append(
                Proposal(
                    action=Action.REBALANCE,
                    confidence=c.confidence,
                    quantity=qty,
                    target_weight=tw,
                    reasons=[
                        f"bullish, but {w:.1%} is well above its {tw:.1%} target: trim back",
                        *c.reasons,
                    ],
                    **base,
                )
            )
        elif (
            c is not None
            and c.actionable_view
            and c.stance is Stance.BULLISH
            and c.confidence >= min_confidence
            and not constraints.get("margin")
            and regime != "risk_off"
            and posture != "defensive"
            and eq > 0
            and price
            and not before_earnings(s)
            and not challenged(s)
        ):
            tw = sized(s, c.confidence)
            add = min((tw - w) * eq, cash, per_order)
            qty = _shares(add / price, True)
            if w < TOP_UP_BELOW * tw and qty >= 1 and add >= ctx.limits.min_order_notional:
                cash -= qty * price
                out.append(
                    Proposal(
                        action=Action.INCREASE,
                        confidence=c.confidence,
                        quantity=qty,
                        target_weight=tw,
                        reasons=[f"bullish consensus ({c.score:+.2f}); {w:.1%} → {tw:.1%}", *c.reasons],
                        **base,
                    )
                )
            else:
                out.append(
                    Proposal(
                        action=Action.HOLD,
                        confidence=c.confidence,
                        target_weight=round(w, 4),
                        reasons=["bullish, already at or near its target weight"],
                        **base,
                    )
                )
        else:
            why = c.reasons if c is not None else ["no consensus"]
            for blocker in (before_earnings(s), challenged(s)):
                if blocker:
                    why = [blocker, *why]
            out.append(
                Proposal(
                    action=Action.HOLD,
                    confidence=c.confidence if c else 0.0,
                    target_weight=round(w, 4),
                    reasons=["hold: " + "; ".join(why)],
                    **base,
                )
            )

    # --- the book as a whole: the same bet twice, a concentrated sector, too much volatility
    if eq > 0:
        _trim_same_bets(ctx, out, consensus)
        _trim_sector(ctx, out, consensus, constraints)
        _trim_volatility(ctx, out)

    # --- new positions
    candidates: list[tuple[float, str, Consensus, dict[str, Any]]] = []
    for s, c in consensus.items():
        if s in ctx.held or s == MARKET or s.startswith("@"):
            continue
        base = {"subject": s, "est_price": ctx.price(s), "current_weight": 0.0, "consensus": c}
        if c.unknown:
            out.append(
                Proposal(
                    action=Action.NO_ACTION,
                    confidence=c.confidence,
                    reasons=["I do not know: " + "; ".join(c.reasons)],
                    **base,
                )
            )
            continue
        if c.stance is not Stance.BULLISH:
            out.append(
                Proposal(
                    action=Action.NO_ACTION,
                    confidence=c.confidence,
                    reasons=[f"consensus {c.stance.value}"],
                    **base,
                )
            )
            continue
        blockers: list[str] = []
        if c.confidence < min_confidence:
            blockers.append(f"confidence {c.confidence:.2f} below {min_confidence:.2f}")
        if regime == "risk_off":
            blockers.append("risk-off market: no new positions")
        if constraints.get("margin"):
            blockers.append("the account is on margin (negative cash)")
        if c.vetoes:
            blockers.extend(v["reason"] for v in c.vetoes)
        if before_earnings(s):
            blockers.append(before_earnings(s) or "")
        if posture == "defensive":
            blockers.append(posture_why)
        if challenged(s):
            blockers.append(challenged(s) or "")
        if blockers:
            out.append(
                Proposal(
                    action=Action.WATCH,
                    confidence=c.confidence,
                    reasons=["bullish but " + "; ".join(blockers)],
                    **base,
                )
            )
            continue
        candidates.append((c.score * c.confidence, s, c, base))
    slots = min(int(constraints.get("free_slots", 0)), max_new)
    if slots == 0 and max_new > 0 and candidates:
        slots, freed = _replace_weakest(ctx, out, candidates, consensus, theses)
        cash += (
            freed  # an estimate: the risk engine re-checks the buy against the cash once the sale has filled
        )
    rank = 0
    for _, s, c, base in sorted(candidates, key=lambda x: -x[0]):
        price = base["est_price"]
        tw = sized(s, c.confidence)
        fit = portfolio_fit(ctx, s, tw)
        if not fit["ok"]:
            out.append(
                Proposal(
                    action=Action.WATCH,
                    confidence=c.confidence,
                    reasons=["bullish, but a poor portfolio fit: " + "; ".join(fit["notes"])],
                    fit=fit,
                    **base,
                )
            )
            continue
        notional = min(tw * eq, cash, per_order) if eq > 0 else 0.0
        qty = _shares(notional / price, True) if price else 0.0
        rank += 1
        if rank > slots:
            out.append(
                Proposal(
                    action=Action.WATCH,
                    confidence=c.confidence,
                    reasons=[f"bullish, ranked {rank}: beyond the {slots} new position(s) allowed this cycle"]
                    + ([posture_why] if posture == "cautious" else []),
                    fit=fit,
                    **base,
                )
            )
        elif qty < 1 or notional < ctx.limits.min_order_notional:
            out.append(
                Proposal(
                    action=Action.WATCH,
                    confidence=c.confidence,
                    reasons=[f"bullish; not enough spendable cash (${cash:,.0f})"],
                    fit=fit,
                    **base,
                )
            )
        else:
            cash -= qty * price
            out.append(
                Proposal(
                    action=Action.BUY,
                    confidence=c.confidence,
                    quantity=qty,
                    target_weight=tw,
                    reasons=[
                        f"bullish consensus ({c.score:+.2f}, confidence {c.confidence:.2f})",
                        *c.reasons,
                        *fit["notes"],
                        *([f"size scaled by {risk_scale:.0%}: {posture_why}"] if risk_scale < 1 else []),
                    ],
                    fit=fit,
                    **base,
                )
            )
    return out


def _strength(c: Consensus | None) -> float:
    return c.score * c.confidence if c is not None and not c.unknown else 0.0


def _trim(ctx: BrainContext, out: list[Proposal], i: int, action: Action, qty: float, reason: str) -> bool:
    """Turn the HOLD at ``out[i]`` into a trim of ``qty`` shares (whole shares for a whole position),
    unless it is below the smallest order worth sending."""
    old = out[i]
    pos = ctx.portfolio.positions[old.subject]
    price = old.est_price or pos.current_price
    whole = float(pos.qty).is_integer()
    qty = min(float(math.ceil(qty)) if whole else round(qty, 6), pos.qty)
    if qty <= 0 or not price or qty * price < ctx.limits.min_order_notional:
        return False
    eq = ctx.portfolio.equity
    out[i] = Proposal(
        subject=old.subject,
        action=action,
        confidence=1.0,
        quantity=qty,
        est_price=old.est_price,
        current_weight=old.current_weight,
        target_weight=round(max(pos.market_value - qty * price, 0.0) / eq, 4) if eq > 0 else None,
        consensus=old.consensus,
        reasons=[reason, *old.reasons],
    )
    return True


def _holds(ctx: BrainContext, out: list[Proposal]) -> dict[str, int]:
    return {
        p.subject: i
        for i, p in enumerate(out)
        if p.action is Action.HOLD and p.subject in ctx.portfolio.positions
    }


def _trim_same_bets(ctx: BrainContext, out: list[Proposal], consensus: dict[str, Consensus]) -> None:
    """Two holdings with a six-month return correlation ≥ ``SAME_BET_CORR`` are one bet: together they
    should respect the position limit. The weaker of the most correlated such pair gives up the excess."""
    held = [h for h in ctx.held if h in ctx.close.columns]
    if len(held) < 2:
        return
    rets = np.log(ctx.close[held].iloc[-121:].astype(float)).diff().dropna()
    if len(rets) < 60:
        return
    corr = rets.corr()
    limit = ctx.limits.max_position_pct
    pairs = sorted(
        (
            (float(corr.loc[a, b]), a, b)
            for i, a in enumerate(held)
            for b in held[i + 1 :]
            if float(corr.loc[a, b]) >= SAME_BET_CORR
        ),
        reverse=True,
    )
    holds = _holds(ctx, out)
    for rho, a, b in pairs:
        wa, wb = ctx.portfolio.weight(a), ctx.portfolio.weight(b)
        excess = wa + wb - limit
        if excess <= 1e-6:
            continue
        movable = [x for x in (a, b) if x in holds]
        if not movable:
            continue
        weaker = min(movable, key=lambda x: _strength(consensus.get(x)))
        other = b if weaker == a else a
        price = out[holds[weaker]].est_price or ctx.portfolio.positions[weaker].current_price
        reason = (
            f"the same bet as {other} (return correlation {rho:.2f}): together {wa + wb:.1%} of equity, "
            f"above the {limit:.0%} position limit — trim the weaker"
        )
        if price and _trim(
            ctx, out, holds[weaker], Action.REDUCE, excess * ctx.portfolio.equity / price, reason
        ):
            return


def _trim_sector(
    ctx: BrainContext, out: list[Proposal], consensus: dict[str, Consensus], constraints: dict[str, Any]
) -> None:
    """A sector above the portfolio agent's limit: its weakest holding gives up the excess."""
    holds = _holds(ctx, out)
    weights = constraints.get("sector_weights") or {}
    for sector, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if sector in ("unknown", "ETF") or w <= SECTOR_LIMIT:
            continue
        members = [s for s in holds if ctx.sectors.get(s) == sector]
        if not members:
            continue
        weakest = min(members, key=lambda x: _strength(consensus.get(x)))
        price = out[holds[weakest]].est_price or ctx.portfolio.positions[weakest].current_price
        reason = f"sector concentration: {sector} is {w:.0%} of equity (limit {SECTOR_LIMIT:.0%}) — trim its weakest"
        if price and _trim(
            ctx, out, holds[weakest], Action.REDUCE, (w - SECTOR_LIMIT) * ctx.portfolio.equity / price, reason
        ):
            return


def book_volatility(ctx: BrainContext) -> tuple[float, dict[str, float]] | None:
    """The book's annualised volatility (six months of daily returns) and each holding's share of it."""
    held = {h: ctx.portfolio.weight(h) for h in ctx.held if h in ctx.close.columns}
    if not held:
        return None
    rets = np.log(ctx.close[list(held)].iloc[-127:].astype(float)).diff().dropna(how="all").fillna(0.0)
    if len(rets) < 60:
        return None
    cov = rets.cov().to_numpy() * 252
    w = np.array(list(held.values()))
    var = float(w @ cov @ w)
    if var <= 0:
        return None
    contrib = w * (cov @ w) / var
    return float(np.sqrt(var)), {s: float(c) for s, c in zip(held, contrib, strict=True)}


def _trim_volatility(ctx: BrainContext, out: list[Proposal]) -> None:
    """Portfolio volatility above ``PORTFOLIO_VOL_CAP``: trim a quarter of the largest risk contributor."""
    measured = book_volatility(ctx)
    if measured is None or measured[0] <= PORTFOLIO_VOL_CAP:
        return
    vol, shares = measured
    holds = _holds(ctx, out)
    movable = [s for s in shares if s in holds]
    if not movable:
        return
    top = max(movable, key=lambda s: shares[s])
    reason = (
        f"portfolio volatility {vol:.0%} is above {PORTFOLIO_VOL_CAP:.0%}: trim a quarter of {top}, "
        f"the largest contributor ({shares[top]:.0%} of the risk)"
    )
    _trim(ctx, out, holds[top], Action.DE_RISK, ctx.portfolio.positions[top].qty / 4, reason)


def _replace_weakest(
    ctx: BrainContext,
    out: list[Proposal],
    candidates: list[tuple[float, str, Consensus, dict[str, Any]]],
    consensus: dict[str, Consensus],
    theses: dict[str, Any] | None,
) -> tuple[int, float]:
    """No free position slot: close the weakest fading holding when a candidate is clearly stronger (one per
    cycle; the sell goes first, and the buy is re-checked by the risk engine once it has filled)."""
    best = max(candidates, key=lambda x: x[0])
    fading: list[tuple[float, int, str]] = []
    for i, p in enumerate(out):
        if p.action is not Action.HOLD or p.subject not in ctx.held:
            continue
        c = consensus.get(p.subject)
        if theses is not None:
            if (theses.get(p.subject) or {}).get("status") != "weakening":
                continue
        elif c is not None and c.actionable_view and c.stance is Stance.BULLISH:
            continue
        strength = c.score * c.confidence if c is not None and not c.unknown else 0.0
        fading.append((strength, i, p.subject))
    if not fading:
        return 0, 0.0
    strength, i, weakest = min(fading)
    if best[0] - strength < REPLACE_MARGIN:
        return 0, 0.0
    old = out[i]
    pos = ctx.portfolio.positions[weakest]
    why = (theses or {}).get(weakest, {}).get("reasons") or old.reasons
    out[i] = Proposal(
        subject=weakest,
        action=Action.CLOSE,
        confidence=old.confidence,
        quantity=pos.qty,
        est_price=old.est_price,
        current_weight=old.current_weight,
        target_weight=0.0,
        consensus=old.consensus,
        reasons=[
            f"replace with {best[1]} ({best[0]:+.2f} vs {strength:+.2f}): no free position slot and this holding "
            "is fading",
            *why,
        ],
    )
    return 1, max(pos.market_value, 0.0)


def quote_checks(ctx: BrainContext) -> dict[str, QuoteCheck]:
    """What the risk engine knows about each symbol's market data (as the trading service builds it)."""
    synthetic = ctx.price_status is DataStatus.SYNTHETIC
    out: dict[str, QuoteCheck] = {}
    for sym, q in ctx.quotes.items():
        qq = ctx.quality.get(sym) or assess_quote(q, ctx.limits.max_quote_age_seconds)
        out[sym] = QuoteCheck(
            price=q.price,
            status=DataStatus.SYNTHETIC if synthetic else DataStatus.LIVE,
            provider=q.provider,
            age_seconds=q.age_seconds,
            spread_bps=qq.spread_bps,
            adv_dollar=ctx.ind(sym, "adv_dollar"),
            spread_source=qq.spread_source,
            quote_problems=qq.problems,
            entry_blocks=qq.entry_blocks,
            source=q.price_source,
        )
    return out


def risk_preview(ctx: BrainContext, proposals: list[Proposal], mode: BrainMode) -> None:
    """Ask the deterministic risk engine about every proposed trade (sells first). Mutates the proposals:
    ``risk``, ``risk_approved``, ``blocked_by`` and ``status``. Nothing is sent."""
    market_vetoes = [
        o.veto for o in ctx.working.opinions.get(MARKET, []) if o.agent_id == "data_quality" and o.veto
    ]
    market_vetoes += list(ctx.working.facts.get("system_vetoes") or [])  # checks that failed closed
    for p in proposals:
        if not p.is_trade:
            p.status = "no_trade"
            continue
        if p.consensus is not None:
            p.blocked_by.extend(v["reason"] for v in p.consensus.vetoes if v["reason"] not in p.blocked_by)
        if ctx.state(p.subject) not in EXECUTABLE_STATES:
            p.blocked_by.append(f"market data {ctx.state(p.subject).value}")
        p.blocked_by.extend(v for v in market_vetoes if v not in p.blocked_by)
    trades = [p for p in proposals if p.is_trade]
    account = ctx.portfolio.account
    if account is None:
        for p in trades:
            p.risk_approved, p.status = False, "not_checked"
            p.risk = {
                "approved": False,
                "summary": "the paper account could not be read: no risk check possible",
            }
        return
    book = RiskBook(
        ctx.limits,
        account,
        ctx.portfolio.positions,
        ctx.portfolio.open_orders,
        ctx.market_open,
        ctx.kill_switch,
        quote_checks(ctx),
    )
    for p in sorted(trades, key=lambda p: 0 if p.action in SELLING else 1):
        pos = ctx.portfolio.positions.get(p.subject)
        closes = p.action is Action.CLOSE or (
            pos is not None and p.action in SELLING and (p.quantity or 0) >= pos.qty - 1e-9
        )
        intent = OrderIntent(
            symbol=p.subject,
            side="sell" if p.action in SELLING else "buy",
            qty=float(p.quantity or 0),
            est_price=float(p.est_price or 0),
            kind=f"brain_{p.action.value}",
            reason="; ".join(p.reasons)[:300],
            closes_position=closes,
            score=p.consensus.score if p.consensus else None,
        )
        decision = book.evaluate(intent)
        if decision.approved:
            book.commit(intent)
        p.risk_approved = decision.approved
        p.risk = {
            "approved": decision.approved,
            "summary": decision.summary,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in decision.checks],
        }
        if not decision.approved:
            p.status = "risk_rejected"
        elif p.blocked_by:
            p.status = "blocked"
        else:
            p.status = {
                BrainMode.DRY_RUN: "dry_run_approved",
                BrainMode.PAPER_RECOMMENDATION: "recommended",
            }.get(mode, "approved")
