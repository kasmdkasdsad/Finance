"""From consensus to proposed portfolio actions — and the deterministic risk engine's verdict on each.

Position management (holdings first):

* at its stop (portfolio agent) → CLOSE; overweight → REDUCE to the position limit;
* a confident bearish consensus → REDUCE (half) or CLOSE (strongly bearish);
* a confident bullish consensus below its target weight → INCREASE toward it;
* otherwise HOLD — including when the consensus is *unknown*.

New positions: a confident bullish consensus on a focus symbol, with no data veto, outside a risk-off
market, with a free position slot and cash that is not borrowed → BUY at a volatility-budgeted target weight
scaled by confidence; at most ``max_new`` per cycle, best first. Each order is sized within the risk
engine's per-order limit (a bigger target is reached over later cycles). Everything else is WATCH or
NO_ACTION, with the reason — the brain is allowed (and expected) to do nothing.

Every proposed trade is then evaluated by the existing :class:`~quantpulse.services.trading_risk.RiskBook`
— the exact limits and checks that guard real orders — as a **preview**: sells first (committed so buys see
them), then buys. The brain never sends an order; this module only records what it would do and whether
the risk engine would allow it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from quantpulse.schemas.common import DataStatus
from quantpulse.services.trading_data import assess_quote
from quantpulse.services.trading_risk import OrderIntent, QuoteCheck, RiskBook

from .consensus import Consensus
from .context import BrainContext
from .types import BUYING, EXECUTABLE_STATES, MARKET, SELLING, Action, BrainMode, Stance

TRADES = BUYING | SELLING


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
        }


def _shares(qty: float, whole: bool) -> float:
    return float(math.floor(qty)) if whole else round(qty, 6)


def target_weight(
    ctx: BrainContext, symbol: str, confidence: float, vol_budget: float, vol_floor: float
) -> float:
    """A volatility-budgeted weight (vol_budget / risk vol), capped by the position limit, scaled by
    confidence."""
    vol = ctx.ind(symbol, "risk_vol") or ctx.ind(symbol, "rv63") or vol_floor
    base = vol_budget / max(vol, vol_floor)
    return round(min(base, ctx.limits.max_position_pct) * min(max(confidence, 0.0), 1.0), 4)


def plan(
    ctx: BrainContext,
    consensus: dict[str, Consensus],
    *,
    min_confidence: float,
    max_new: int,
    vol_budget: float,
    vol_floor: float,
) -> list[Proposal]:
    out: list[Proposal] = []
    constraints = ctx.working.facts.get("portfolio_constraints") or {}
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
        elif (
            c is not None
            and c.actionable_view
            and c.stance is Stance.BULLISH
            and c.confidence >= min_confidence
            and not constraints.get("margin")
            and regime != "risk_off"
            and eq > 0
            and price
        ):
            tw = target_weight(ctx, s, c.confidence, vol_budget, vol_floor)
            add = min((tw - w) * eq, cash, per_order)
            qty = _shares(add / price, True)
            if qty >= 1 and add >= ctx.limits.min_order_notional:
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
            out.append(
                Proposal(
                    action=Action.HOLD,
                    confidence=c.confidence if c else 0.0,
                    target_weight=round(w, 4),
                    reasons=["hold: " + "; ".join(why)],
                    **base,
                )
            )

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
    for rank, (_, s, c, base) in enumerate(sorted(candidates, key=lambda x: -x[0])):
        price = base["est_price"]
        tw = target_weight(ctx, s, c.confidence, vol_budget, vol_floor)
        notional = min(tw * eq, cash, per_order) if eq > 0 else 0.0
        qty = _shares(notional / price, True) if price else 0.0
        if rank >= slots:
            out.append(
                Proposal(
                    action=Action.WATCH,
                    confidence=c.confidence,
                    reasons=[
                        f"bullish, ranked {rank + 1}: beyond the {slots} new position(s) allowed this cycle"
                    ],
                    **base,
                )
            )
        elif qty < 1 or notional < ctx.limits.min_order_notional:
            out.append(
                Proposal(
                    action=Action.WATCH,
                    confidence=c.confidence,
                    reasons=[f"bullish; not enough spendable cash (${cash:,.0f})"],
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
                    ],
                    **base,
                )
            )
    return out


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
        )
    return out


def risk_preview(ctx: BrainContext, proposals: list[Proposal], mode: BrainMode) -> None:
    """Ask the deterministic risk engine about every proposed trade (sells first). Mutates the proposals:
    ``risk``, ``risk_approved``, ``blocked_by`` and ``status``. Nothing is sent."""
    market_vetoes = [
        o.veto for o in ctx.working.opinions.get(MARKET, []) if o.agent_id == "data_quality" and o.veto
    ]
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
