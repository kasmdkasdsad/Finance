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

**The final execution audit** (:meth:`BrainExecutor.audit`). Before the Brain's first order after the
process starts, and again on every new trading day, every gate is checked and the whole picture is
reported — the mode, the endpoint (verified), paper key, trading switches, both kill switches, the
environment (.env unchanged since start), the account, a fresh reconciliation, the market clock and this
computer's clock, a live benchmark quote, positions, buying power, the risk limits, the agents, and the
orders about to go with their consensus, risk preview and reasons. Orders go only if every required check
passes; a passing audit also arms scheduled Brain cycles (``QP_TRADING_SCHEDULER_REQUIRES_ARMING``), so the
first autonomous trade needs no click — and no order goes without it. Each audit is kept
(``GET /brain/execution-audit``), logged, and written to the trading service's event log. The trading
service still re-checks the kill switches, switches, key, endpoint and .env immediately before each order.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from quantpulse.config import Settings, env_file_drift
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.market_calendar import NEW_YORK, is_trading_day, regular_close
from quantpulse.providers.alpaca_trading import PAPER_URL, BrokerError
from quantpulse.schemas.trading import CycleOut, ProposedTradeOut
from quantpulse.services.order_manager import DUPLICATE_NOTE
from quantpulse.services.trading import BrainOrder, TradingService
from quantpulse.services.trading_data import TradingDataLoader
from quantpulse.services.trading_risk import OptionOrderIntent, RiskLimits

from .context import BrainContext
from .decisions import Proposal
from .types import BUYING, MARKET, SELLING, Action, BrainMode

logger = logging.getLogger(__name__)

DATA_BLOCKED = "TRADING BLOCKED — DATA QUALITY INSUFFICIENT"
AUDIT_KEY = "execution_audit"  # brain_state: the latest audit
AUDITS_KEY = "execution_audits"  # brain_state: the last few
AUDITS_KEPT = 30
# what must pass before an order goes (the rest of an audit is the report)
PRE_TRADE_REQUIRED = (
    "brain_mode", "paper_setting", "paper_endpoint", "paper_key", "trading_enabled", "dry_run_off",
    "trading_kill_switch", "brain_kill_switch", "environment", "account", "reconciliation", "market_open",
    "clock_skew", "market_data",
)  # fmt: skip
STARTUP_REQUIRED = tuple(c for c in PRE_TRADE_REQUIRED if c not in ("market_open", "market_data"))
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
    local = ctx.as_of.astimezone(NEW_YORK)
    if ctx.market_open and is_trading_day(local.date()):
        close = datetime.combine(local.date(), regular_close(local.date()), NEW_YORK)
        left = (close - local).total_seconds() / 60
        if 0 <= left <= ctx.stop_minutes_before_close:
            out.append(
                {
                    "code": "closing_soon",
                    "reason": f"{left:.0f} minutes to the close: no new position this late in the session",
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


def refused(p: Proposal) -> str:
    """Why the risk preview refused a decision, naming the checks that failed (all of them are kept on the
    decision's ``risk``)."""
    risk = p.risk or {}
    failed = [
        f"{c['name'].replace('_', ' ')}: {c['detail']}"
        for c in risk.get("checks") or []
        if not c.get("passed")
    ]
    why = "; ".join(failed) or risk.get("summary") or "no reason was recorded"
    return f"the risk preview did not approve it: {why}"


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
        # not sent: why, in the same words as every other decision that did not go
        **({"reason": f"stopped at submission: {row.error}"} if row.status == "blocked_at_submit" else {}),
        "quote_price": row.quote_price,
        "quote_bid": row.quote_bid,
        "quote_ask": row.quote_ask,
        "quote_spread_bps": row.quote_spread_bps,
        "quote_age_seconds": row.quote_age_seconds,
        "quote_source": row.quote_source,
        "submit_latency_ms": row.submit_latency_ms,
    }


class BrainExecutor:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        trading: TradingService,
        data: TradingDataLoader | None = None,
        store: Any = None,  # BrainStore: where audits are kept
        agents: Callable[[], dict[str, int]] | None = None,  # registered / enabled agents
    ) -> None:
        self._s = settings
        self._clock = clock
        self._trading = trading
        self._data = data
        self._store = store
        self._agents = agents
        self._audited_day: Any = None  # the trading day a pre-trade audit last passed in this process
        # the session's execution readiness (pre-market audit, data health, reconciliation, strategy status):
        # set by the Brain service; an order waits until it has passed today (see brain.research.operating)
        self.readiness: Callable[[], Awaitable[dict[str, Any]]] | None = None

    # ------------------------------------------------------------------ the final execution audit
    async def audit(
        self,
        *,
        purpose: str,
        ctx: BrainContext | None = None,
        orders: Sequence[Proposal] = (),
    ) -> dict[str, Any]:
        """Every gate an order must pass, checked now, and a report of what is about to happen.
        ``purpose``: ``pre_trade`` (orders are waiting on it) or ``startup`` (after a restart)."""
        s, t = self._s, self._trading
        now = self._clock.now()
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool | None, detail: str) -> None:
            checks.append({"name": name, "ok": ok, "detail": detail})

        add("brain_mode", s.brain_owns_account, f"QP_BRAIN_MODE={s.brain_mode}")
        add("paper_setting", s.alpaca_paper, f"QP_ALPACA_PAPER={str(s.alpaca_paper).lower()}")
        try:
            endpoint = t.verify_paper()
            add("paper_endpoint", endpoint == PAPER_URL, endpoint)
        except BrokerError as exc:
            add("paper_endpoint", False, str(exc))
        key = s.alpaca_api_key_id.get_secret_value() if s.alpaca_api_key_id is not None else None
        add(
            "paper_key",
            bool(key and key.startswith("PK")),
            "set; looks like a paper key (PK…)"
            if key and key.startswith("PK")
            else ("not set" if not key else "does not look like a paper key (paper keys start with PK)"),
        )
        add(
            "trading_enabled",
            s.alpaca_trading_enabled,
            f"QP_ALPACA_TRADING_ENABLED={str(s.alpaca_trading_enabled).lower()}",
        )
        add("dry_run_off", not s.trading_dry_run, f"QP_TRADING_DRY_RUN={str(s.trading_dry_run).lower()}")
        kill = await t.kill_switch()
        add(
            "trading_kill_switch",
            not kill.active,
            "off" if not kill.active else f"ON ({kill.reason or kill.source})",
        )
        brain_kill = await t.brain_kill_switch()
        add("brain_kill_switch", not brain_kill.active,
            "off" if not brain_kill.active else f"ON ({brain_kill.reason or brain_kill.source})")  # fmt: skip
        drift = env_file_drift(s)
        add(
            "environment",
            not drift,
            "the running settings match .env" if not drift else "; ".join(drift)[:300],
        )
        account: Any = None
        positions: list[Any] = []
        open_orders: list[Any] = []
        try:
            account = await t.broker.account()
            add(
                "account",
                not account.blocked and account.equity > 0,
                f"{account.status}; equity ${account.equity:,.2f}, cash ${account.cash:,.2f}, buying power "
                f"${account.buying_power:,.2f}" + ("; BLOCKED by Alpaca" if account.blocked else ""),
            )
        except Exception as exc:  # fail closed
            add("account", False, f"{type(exc).__name__}: {exc}")
        try:
            rec = await t.reconcile("brain execution audit")
            positions = await t.broker.positions()
            open_orders = await t.broker.open_orders()
            add("reconciliation", True, f"{rec.positions} position(s), {rec.open_orders} open order(s), "
                f"{rec.orders_updated} update(s), {rec.orders_added} found on Alpaca")  # fmt: skip
        except Exception as exc:
            add("reconciliation", False, f"{type(exc).__name__}: {exc}")
        try:
            before = self._clock.now()
            clock = await t.broker.clock()
            skew = ((before + (self._clock.now() - before) / 2) - clock.timestamp).total_seconds()
            add("market_open", clock.is_open, "open" if clock.is_open else "closed"
                + (f" (next open {clock.next_open.astimezone(NEW_YORK):%a %H:%M} New York)" if clock.next_open else ""))  # fmt: skip
            add("clock_skew", abs(skew) <= MAX_CLOCK_SKEW_SECONDS,
                f"this computer is {skew:+.1f}s from Alpaca's clock (limit ±{MAX_CLOCK_SKEW_SECONDS:.0f}s)")  # fmt: skip
        except Exception as exc:
            add("market_open", False, f"Alpaca's clock unavailable: {type(exc).__name__}: {exc}")
            add("clock_skew", False, "unknown (Alpaca's clock unavailable)")
        bench = s.benchmark_symbol
        quote = None
        if ctx is not None and self._data is None:
            quote = ctx.quotes.get(bench)
        elif self._data is not None:
            try:
                quote = (await self._data.live_quotes([bench], consolidated=False)).get(bench)
            except Exception as exc:
                add("market_data", False, f"{type(exc).__name__}: {exc}")
        if not any(c["name"] == "market_data" for c in checks):
            limit = s.trading_max_quote_age_seconds
            add(
                "market_data",
                quote is not None and quote.age_seconds <= limit,
                f"{bench}: {quote.price_source}, {quote.age_seconds:,.0f}s old (limit {limit:,.0f}s), "
                f"feed {s.alpaca_stock_feed}"
                if quote is not None
                else f"no live quote for {bench} (feed {s.alpaca_stock_feed})",
            )
        required = PRE_TRADE_REQUIRED if purpose == "pre_trade" else STARTUP_REQUIRED
        failed = [c for c in checks if c["name"] in required and not c["ok"]]
        report: dict[str, Any] = {
            "at": now.isoformat(),
            "purpose": purpose,
            "ok": not failed,
            "failed": [f"{c['name']}: {c['detail']}" for c in failed],
            "checks": checks,
            "mode": s.brain_mode,
            "endpoint": t.broker.base_url,
            # read from the client and the setting, not asserted: the settings refuse anything but paper and
            # there is no trading-URL setting, so a live endpoint would be a defect (and fail the audit)
            "paper": t.broker.base_url == PAPER_URL and s.alpaca_paper,
            "live_trading_possible": not (t.broker.base_url == PAPER_URL and s.alpaca_paper),
            "supervisor": {"enabled": s.brain_supervisor_enabled, "armed": await t.armed()},
            "data_source": {"stock_feed": s.alpaca_stock_feed, "vendors_asked_first": ["alpaca"]},
            "buying_power": account.buying_power if account is not None else None,
            "equity": account.equity if account is not None else None,
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "market_value": p.market_value,
                    "unrealized_plpc": p.unrealized_plpc,
                }
                for p in positions
            ],
            "open_orders": [
                {
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": o.qty,
                    "status": o.status,
                    "client_order_id": o.client_order_id,
                }
                for o in open_orders
            ],
            "risk_limits": asdict(RiskLimits.from_settings(s)),
            "agents": self._agents() if self._agents is not None else None,
            "orders": [
                {
                    "subject": p.subject,
                    "action": p.action.value,
                    "quantity": p.quantity,
                    "est_price": p.est_price,
                    "consensus": {
                        "stance": p.consensus.stance.value,
                        "score": round(p.consensus.score, 3),
                        "confidence": round(p.consensus.confidence, 3),
                        "sources": p.consensus.sources,
                    }
                    if p.consensus is not None
                    else None,
                    "risk_preview": (p.risk or {}).get("summary"),
                    "reasons": p.reasons[:4],
                }
                for p in orders
            ],
        }
        await self._keep(report)
        return report

    async def _keep(self, report: dict[str, Any]) -> None:
        text = "; ".join(
            f"{c['name']}={'ok' if c['ok'] else 'FAIL' if c['ok'] is False else '-'}"
            for c in report["checks"]
        )
        logger.info(
            "Brain execution audit (%s): %s — %s", report["purpose"], "PASS" if report["ok"] else "FAIL", text
        )
        await self._trading.log_event(
            "brain_execution_audit",
            f"Brain execution audit ({report['purpose']}): "
            + ("every gate passed" if report["ok"] else "FAILED — " + "; ".join(report["failed"])[:400]),
            details={"ok": report["ok"], "failed": report["failed"], "orders": len(report["orders"])},
        )
        if self._store is not None:
            now = self._clock.now()
            await self._store.set_state(AUDIT_KEY, report, now)
            history = (await self._store.get_state(AUDITS_KEY) or {}).get("items", [])
            await self._store.set_state(AUDITS_KEY, {"items": [*history, report][-AUDITS_KEPT:]}, now)

    async def _audit_outcome(self, **outcome: Any) -> None:
        if self._store is None:
            return
        latest = await self._store.get_state(AUDIT_KEY)
        if latest:
            latest["outcome"] = outcome
            await self._store.set_state(AUDIT_KEY, latest, self._clock.now())

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
        # arming is not a blocker here: a passing pre-trade audit arms scheduled cycles (see execute)
        blockers += await self._trading.submit_blockers(kill, scheduled=False, owner="brain")
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
                return refused(p)
            return None
        if p.protective:
            return None  # an exit at a stop or a broken thesis: the risk engine decides
        if p.blocked_by:
            return "blocked: " + "; ".join(p.blocked_by)
        if p.risk_approved is False:
            return refused(p)
        return None

    async def execute(
        self,
        ctx: BrainContext,
        proposals: list[Proposal],
        *,
        cycle_id: int,
        scheduled: bool,
        option_orders: Sequence[OptionOrderIntent] = (),
    ) -> dict[str, Any]:
        """Send what may be sent; record on every trade decision what happened to it. Option orders (from the
        Options Brain) go with the stock orders into the same trading cycle, through the same gate: none is
        sent when the gate is closed, and option entries wait while entries are halted (exits still go)."""
        trades = [p for p in proposals if p.is_trade]
        gate = await self.gate(ctx, scheduled=scheduled)
        report: dict[str, Any] = {
            **gate.to_dict(),
            "orders_sent": 0,
            "trading_cycle_id": None,
            "option_trades": [],
        }
        if not trades and not option_orders:
            return report
        if not gate.send:
            for p in trades:
                p.execution = {"sent": False, "reason": "not sent: " + "; ".join(gate.blockers)}
            return report
        options = [o for o in option_orders if not o.opening or gate.entries]
        if len(options) < len(option_orders):
            report["option_entries_halted"] = "entries halted: " + "; ".join(h["reason"] for h in gate.halts)
        chosen: list[Proposal] = []
        for p in trades:
            why = self.sendable(p, gate)
            if why is None:
                chosen.append(p)
            else:
                p.status = "halted" if why.startswith("entries halted") else p.status
                p.execution = {"sent": False, "reason": why}
        if not chosen and not options:
            return report
        today = self._clock.now().astimezone(NEW_YORK).date()
        armed = await self._trading.armed()
        if not armed or self._audited_day != today:
            audit = await self.audit(purpose="pre_trade", ctx=ctx, orders=chosen)
            report["audit"] = {"ok": audit["ok"], "failed": audit["failed"], "at": audit["at"]}
            if not audit["ok"]:
                for p in chosen:
                    p.status = "blocked"
                    p.execution = {
                        "sent": False,
                        "reason": "execution audit failed: " + "; ".join(audit["failed"]),
                    }
                return report
            if not armed:
                await self._trading.arm("Brain pre-trade execution audit passed")
            self._audited_day = today
        # then the session's execution readiness (an extra gate after the audit: it only ever holds orders)
        if self.readiness is not None and self._s.brain_owns_account:
            ready = await self.readiness()
            report["readiness"] = {
                "passed": ready.get("passed"),
                "failed": ready.get("failed"),
                "at": ready.get("at"),
            }
            if not ready.get("passed"):
                for p in chosen:
                    p.status = "blocked"
                    p.execution = {
                        "sent": False,
                        "reason": "execution readiness has not passed today: "
                        + "; ".join(ready.get("failed") or ["not checked"]),
                    }
                return report
        try:
            cycle = await self._trading.run_brain(
                [to_order(p, ctx) for p in chosen],
                brain_cycle_id=cycle_id,
                scheduled=scheduled,
                option_orders=options,
            )
        except (BrokerError, DomainError) as exc:
            logger.warning("Brain execution failed: %s", exc)
            for p in chosen:
                p.status, p.execution = "failed", {"sent": False, "reason": f"execution failed: {exc}"}
            report["error"] = str(exc)
            return report
        report["trading_cycle_id"] = cycle.id
        report["option_trades"] = [
            t.model_dump(mode="json") for t in cycle.trades if t.asset_class == "us_option"
        ]
        report["option_orders_sent"] = sum(1 for t in report["option_trades"] if t.get("alpaca_order_id"))
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
                    "duplicate_prevented"
                    if p.execution["duplicate_prevented"]
                    else "blocked_at_submit"
                    if row.status == "blocked_at_submit"
                    else row.stage or row.status
                )
        report["orders_sent"] = sum(1 for p in chosen if (p.execution or {}).get("sent"))
        if "audit" in report:
            await self._audit_outcome(trading_cycle_id=cycle.id, orders_sent=report["orders_sent"],
                                      statuses={p.subject: p.status for p in chosen})  # fmt: skip
        return report
