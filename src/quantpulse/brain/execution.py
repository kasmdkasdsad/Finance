"""Execution: the Brain's decisions become Alpaca **paper** orders — only through the trading service.

The Brain owns the paper account when ``QP_BRAIN_MODE=paper_execution``. It still has no broker access of
its own: :meth:`BrainExecutor.execute` hands the decisions to
:meth:`~quantpulse.services.trading.TradingService.run_brain`, which reconciles with Alpaca, re-reads the
account, prices every order from fresh quotes and sends it through the same risk engine, order manager and
switches as every other order (sells first, then buys against the cash actually available). There is no
second risk engine here — only the questions the risk engine cannot answer about the Brain itself:

======================  ================================================================================
nothing is sent         another mode (proposals only, managed in the simulated paper book); the market is
                        closed; the **Brain kill switch**; any reason the trading service would not send
                        (keys, ``QP_ALPACA_TRADING_ENABLED``, ``QP_TRADING_DRY_RUN``, the trading kill
                        switch, arming for scheduled cycles); Alpaca unavailable
entries halted          (exits and trims still go) the daily loss limit; data quality insufficient (the
                        data-quality agent's market veto, or a fail-closed check); this computer's clock
                        far from Alpaca's; an account Alpaca reports blocked or a state that does not add
                        up (short positions, non-positive equity, positions that do not match the account);
                        exposure above the limits
per decision            buys need the risk preview's approval and no veto; a discretionary sell needs no
                        veto; a protective exit (a stop, a broken thesis) is always handed on — the risk
                        engine decides
======================  ================================================================================

Every decision records what happened to it (``execution`` on the decision, and the trading cycle, orders,
fills and events in the trading service's audit trail).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.providers.alpaca_trading import BrokerError
from quantpulse.schemas.trading import CycleOut, ProposedTradeOut
from quantpulse.services.order_manager import DUPLICATE_NOTE
from quantpulse.services.trading import BrainOrder, TradingService

from .context import BrainContext
from .decisions import Proposal
from .types import BUYING, MARKET, SELLING, Action, BrainMode

logger = logging.getLogger(__name__)

DATA_BLOCKED = "TRADING BLOCKED — DATA QUALITY INSUFFICIENT"
MAX_CLOCK_SKEW_SECONDS = 10.0  # beyond this every quote age is unreliable: no new positions
EXPOSURE_TOLERANCE = 0.02  # above the exposure limit by more than this share of equity: unexpected
POSITION_TOLERANCE = 0.25  # a position more than 25% above its limit was not put there by the Brain


@dataclass
class Gate:
    """Whether Brain orders may go out this cycle, and every reason they may not."""

    send: bool
    entries: bool
    blockers: list[str] = field(default_factory=list)  # nothing is sent
    halts: list[dict[str, str]] = field(default_factory=list)  # new positions and increases held back

    def to_dict(self) -> dict[str, Any]:
        return {
            "orders_allowed": self.send,
            "entries_allowed": self.send and self.entries,
            "blockers": self.blockers,
            "entry_halts": self.halts,
        }


def entry_halts(ctx: BrainContext) -> list[dict[str, str]]:
    """Conditions under which the Brain opens nothing new (it may still exit and trim)."""
    out: list[dict[str, str]] = []
    acct = ctx.account.account
    limits = ctx.limits
    if acct is not None and acct.last_equity > 0 and acct.day_pl_pct <= -limits.max_daily_loss_pct:
        out.append(
            {
                "code": "daily_loss",
                "reason": f"daily loss {acct.day_pl_pct:+.2%} reached the −{limits.max_daily_loss_pct:.0%} "
                "limit: no new positions today",
            }
        )
    vetoes = [o.veto for o in ctx.working.opinions.get(MARKET, []) if o.agent_id == "data_quality" and o.veto]
    vetoes += list(ctx.working.facts.get("system_vetoes") or [])
    if ctx.market_open and vetoes:
        out.append({"code": "data_quality", "reason": f"{DATA_BLOCKED}: " + "; ".join(vetoes)})
    skew = (ctx.feed or {}).get("clock_skew_s")
    if skew is not None and abs(skew) > MAX_CLOCK_SKEW_SECONDS:
        out.append(
            {
                "code": "clock_skew",
                "reason": f"this computer's clock is {skew:+.0f}s off Alpaca's (limit ±{MAX_CLOCK_SKEW_SECONDS:.0f}s):"
                " quote ages cannot be trusted",
            }
        )
    unexpected = ctx.working.facts.get("unexpected_positions") or []
    if unexpected:
        out.append(
            {
                "code": "unexpected_exposure",
                "reason": "positions the Brain did not open or adopt: "
                + ", ".join(sorted(unexpected))
                + " (adopt them on the Brain page, or close them): nothing new until then",
            }
        )
    if acct is not None:
        problems: list[str] = []
        if acct.blocked:
            problems.append("Alpaca reports the account blocked")
        if acct.equity <= 0:
            problems.append(f"equity is ${acct.equity:,.2f}")
        shorts = [s for s, p in ctx.account.positions.items() if p.qty < 0]
        if shorts and not limits.allow_shorts:
            problems.append(
                "short positions although short selling is disabled: " + ", ".join(sorted(shorts))
            )
        held_value = sum(p.market_value for p in ctx.account.positions.values() if p.qty > 0)
        if acct.equity > 0 and abs(held_value - acct.long_market_value) > max(0.01 * acct.equity, 1.0):
            problems.append(
                f"positions are worth ${held_value:,.0f} but the account reports ${acct.long_market_value:,.0f}"
            )
        if problems:
            out.append({"code": "inconsistent_state", "reason": "; ".join(problems)})
        if acct.equity > 0:
            exposure = acct.long_market_value / acct.equity
            over = [
                f"{s} {p.market_value / acct.equity:.0%}"
                for s, p in ctx.account.positions.items()
                if p.market_value / acct.equity > limits.max_position_pct * (1 + POSITION_TOLERANCE)
            ]
            if exposure > limits.max_total_exposure_pct + EXPOSURE_TOLERANCE or over:
                out.append(
                    {
                        "code": "unexpected_exposure",
                        "reason": (
                            f"long exposure {exposure:.0%} (limit {limits.max_total_exposure_pct:.0%})"
                            if exposure > limits.max_total_exposure_pct + EXPOSURE_TOLERANCE
                            else "positions far above the position limit: " + ", ".join(over)
                        )
                        + ": nothing new until it is back within the limits",
                    }
                )
    return out


def to_order(p: Proposal, ctx: BrainContext) -> BrainOrder:
    pos = ctx.portfolio.positions.get(p.subject)
    selling = p.action in SELLING
    return BrainOrder(
        symbol=p.subject,
        side="sell" if selling else "buy",
        qty=float(p.quantity or 0),
        est_price=float(p.est_price or 0),
        action=p.action.value,
        reason="; ".join(p.reasons)[:300],
        closes_position=p.action is Action.CLOSE
        or (pos is not None and selling and (p.quantity or 0) >= pos.qty - 1e-9),
        protective=p.protective,
        score=p.consensus.score if p.consensus else None,
        adv_dollar=ctx.ind(p.subject, "adv_dollar"),
        current_weight=float(p.current_weight or 0.0),
        target_weight=float(p.target_weight or 0.0),
    )


def _outcome(row: ProposedTradeOut, cycle: CycleOut) -> dict[str, Any]:
    duplicate = row.error == DUPLICATE_NOTE
    return {
        # sent in this cycle: Alpaca acknowledged it, or its fate is still unknown (never resent either way)
        "sent": not duplicate and (row.alpaca_order_id is not None or row.stage in ("unknown", "submitting")),
        "duplicate_prevented": duplicate,
        "trading_cycle_id": cycle.id,
        "trading_cycle_key": cycle.cycle_key,
        "mode": cycle.mode,
        "stage": row.stage,
        "status": row.status,
        "client_order_id": row.client_order_id,
        "alpaca_order_id": row.alpaca_order_id,
        "qty": row.qty,
        "est_price": row.est_price,
        "order_type": row.order_type,
        "limit_price": row.limit_price,
        "filled_qty": row.filled_qty,
        "filled_avg_price": row.filled_avg_price,
        "submitted_at": row.submitted_at.isoformat() if row.submitted_at else None,
        "risk": row.risk,
        "checks": [c.model_dump() for c in row.checks],
        "error": row.error,
    }


class BrainExecutor:
    def __init__(self, settings: Settings, clock: Clock, trading: TradingService) -> None:
        self._s = settings
        self._clock = clock
        self._trading = trading

    async def gate(self, ctx: BrainContext, *, scheduled: bool) -> Gate:
        blockers: list[str] = []
        if ctx.mode is not BrainMode.PAPER_EXECUTION:
            blockers.append(
                f"QP_BRAIN_MODE={ctx.mode.value}: proposals only (the Brain does not own the account)"
            )
        if not ctx.market_open:
            blockers.append("the market is closed")
        if not ctx.account.available:
            blockers.append(f"Alpaca unavailable ({ctx.account.error})")
        kill = await self._trading.kill_switch()
        blockers += await self._trading.submit_blockers(kill, scheduled=scheduled, owner="brain")
        halts = entry_halts(ctx)
        return Gate(send=not blockers, entries=not halts, blockers=blockers, halts=halts)

    @staticmethod
    def sendable(p: Proposal, gate: Gate) -> str | None:
        """Why this decision is not handed to the trading service (``None``: it is)."""
        if p.action in BUYING:
            if not gate.entries:
                return "entries halted: " + "; ".join(h["reason"] for h in gate.halts)
            if p.blocked_by:
                return "blocked: " + "; ".join(p.blocked_by)
            if not p.risk_approved:
                return "the risk preview did not approve it"
            return None
        if p.protective:
            return None  # an exit at a stop or a broken thesis: the risk engine decides
        if p.blocked_by:
            return "blocked: " + "; ".join(p.blocked_by)
        if p.risk_approved is False:
            return "the risk preview did not approve it"
        return None

    async def execute(
        self, ctx: BrainContext, proposals: list[Proposal], *, cycle_id: int, scheduled: bool
    ) -> dict[str, Any]:
        """Send what may be sent; record on every trade decision what happened to it."""
        trades = [p for p in proposals if p.is_trade]
        gate = await self.gate(ctx, scheduled=scheduled)
        report: dict[str, Any] = {**gate.to_dict(), "orders_sent": 0, "trading_cycle_id": None}
        if not trades:
            return report
        if not gate.send:
            for p in trades:
                p.execution = {"sent": False, "reason": "not sent: " + "; ".join(gate.blockers)}
            return report
        chosen: list[Proposal] = []
        for p in trades:
            why = self.sendable(p, gate)
            if why is None:
                chosen.append(p)
            else:
                p.status = "halted" if why.startswith("entries halted") else p.status
                p.execution = {"sent": False, "reason": why}
        if not chosen:
            return report
        try:
            cycle = await self._trading.run_brain(
                [to_order(p, ctx) for p in chosen], brain_cycle_id=cycle_id, scheduled=scheduled
            )
        except (BrokerError, DomainError) as exc:
            logger.warning("Brain execution failed: %s", exc)
            for p in chosen:
                p.status, p.execution = "failed", {"sent": False, "reason": f"execution failed: {exc}"}
            report["error"] = str(exc)
            return report
        report["trading_cycle_id"] = cycle.id
        report["trading_mode"] = cycle.mode
        report["trading_status"] = cycle.status
        report["notes"] = cycle.notes
        by_key = {(t.symbol, t.side): t for t in cycle.trades}
        for p in chosen:
            side = "sell" if p.action in SELLING else "buy"
            row = by_key.get((p.subject, side))
            if cycle.status == "failed":
                p.status = "failed"
                p.execution = {"sent": False, "reason": f"trading cycle failed: {cycle.error}",
                               "trading_cycle_id": cycle.id}  # fmt: skip
            elif row is None:
                why = cycle.skipped.get(p.subject) or "not traded by the trading service"
                p.status, p.execution = (
                    "skipped",
                    {"sent": False, "reason": why, "trading_cycle_id": cycle.id},
                )
            else:
                p.execution = _outcome(row, cycle)
                p.status = (
                    "duplicate_prevented" if p.execution["duplicate_prevented"] else row.stage or row.status
                )
        report["orders_sent"] = sum(1 for p in chosen if (p.execution or {}).get("sent"))
        return report
