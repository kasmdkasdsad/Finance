"""Order manager: sends risk-approved orders to the Alpaca paper account exactly once and keeps QuantPulse's
record of them in step with Alpaca.

Duplicate protection
    * **Deterministic client order ids.** An order's id is derived from the cycle slot, the symbol and the
      side (``qp-20260925T1030-AAPL-b``). A second run of the same slot — a double click, a poller that fires
      twice, a restart mid-cycle — produces the same id, and a given id is only ever sent once.
    * **Write-ahead record.** The order is stored as ``pending_submit`` (unique on the client id) *before*
      the request leaves; a second attempt fails on the unique constraint and sends nothing.
    * **No blind resubmission.** If the request times out or the connection drops, the outcome is unknown:
      the order is looked up by its client id and, if Alpaca has not seen it, marked ``submit_unknown`` and
      left for reconciliation — never resent. Alpaca itself also rejects a repeated client id.

Reconciliation
    Alpaca is authoritative. Open and recent orders are read from Alpaca; every local order is updated from
    it (status, filled quantity, average fill price, timestamps), orders found on Alpaca but missing locally
    are added (``external`` when QuantPulse did not place them), and unresolved local orders Alpaca never
    received are closed out as ``submit_failed`` after a grace period. Status changes are written to the
    trading event log (submitted, partially filled, filled, canceled, rejected, expired).
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy.exc import IntegrityError

from quantpulse.core.clock import Clock
from quantpulse.db import repositories as repo
from quantpulse.db.models import BrokerOrderRow
from quantpulse.db.session import Database
from quantpulse.providers.alpaca_trading import (
    TERMINAL_STATUSES,
    AlpacaPaperBroker,
    BrokerError,
    BrokerOrder,
    DuplicateClientOrderId,
    OrderRejected,
    OrderSpec,
)
from quantpulse.services.trading_risk import OrderIntent

logger = logging.getLogger(__name__)

PREFIX = "qp"
STRATEGY = "quantpulse"
EXTERNAL = "external"
PENDING_SUBMIT = "pending_submit"
SUBMIT_UNKNOWN = "submit_unknown"
SUBMIT_FAILED = "submit_failed"
UNRESOLVED = frozenset({PENDING_SUBMIT, SUBMIT_UNKNOWN})
FINAL = TERMINAL_STATUSES | {SUBMIT_FAILED}
UNKNOWN_GRACE = timedelta(minutes=2)
RECONCILE_LOOKBACK = timedelta(days=7)
EVENT_FOR_STATUS = {
    "partially_filled": "order_partially_filled",
    "filled": "order_filled",
    "canceled": "order_canceled",
    "rejected": "order_rejected",
    "expired": "order_expired",
}


# Alpaca statuses of an order that is working (acknowledged, not yet done).
WORKING_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "accepted_for_bidding",
        "held",
        "calculated",
        "pending_replace",
        "pending_cancel",
        "stopped",
        "suspended",
    }
)
_STAGE_FOR_STATUS = {
    PENDING_SUBMIT: "submitting",
    SUBMIT_UNKNOWN: "unknown",
    SUBMIT_FAILED: "failed",
    "rejected": "rejected",
    "pending_new": "submitted",
    "partially_filled": "partially_filled",
    "filled": "filled",
    "canceled": "canceled",
    "done_for_day": "canceled",
    "replaced": "canceled",
    "expired": "expired",
}
# Stages that mean Alpaca has (or had) the order.
AT_ALPACA = frozenset({"submitted", "accepted", "partially_filled", "filled", "canceled", "expired"})


def trade_stage(approved: bool, status: str | None) -> str:
    """Where a proposed trade got to, from its risk decision and its order status (see ``TradeStage``)."""
    if not approved or status == "risk_rejected":
        return "risk_rejected"
    if status in (None, "", "dry_run", "not_submitted"):
        return "risk_approved"
    if status in WORKING_STATUSES:
        return "accepted"
    return _STAGE_FOR_STATUS.get(status, "submitted")


def client_order_id(slot: str, symbol: str, side: str) -> str:
    """Deterministic id for (cycle slot, symbol, side): at most one such order can ever be sent."""
    return f"{PREFIX}-{slot}-{symbol.upper().replace('.', '_')}-{side[:1].lower()}"


def is_ours(cid: str) -> bool:
    return cid.startswith(f"{PREFIX}-")


@dataclass
class Submission:
    client_order_id: str
    symbol: str
    side: str
    status: str
    submitted: bool  # Alpaca acknowledged the order (it has an Alpaca order id)
    duplicate: bool = False
    order: BrokerOrder | None = None
    error: str | None = None

    @property
    def alpaca_order_id(self) -> str | None:
        return self.order.id if self.order is not None else None


@dataclass
class ReconcileReport:
    checked: int = 0
    updated: int = 0
    added: int = 0
    resolved_unknown: int = 0
    open_orders: int = 0
    changes: list[str] = field(default_factory=list)


def _apply(row: BrokerOrderRow, o: BrokerOrder, now: datetime) -> list[tuple[str, str]]:
    """Copy Alpaca's view of an order onto the local row; the events the change represents."""
    events: list[tuple[str, str]] = []
    before_status, before_filled = row.status, row.filled_quantity or 0.0
    row.alpaca_order_id = o.id
    row.status = o.status
    row.filled_quantity = o.filled_qty
    row.average_fill_price = o.filled_avg_price
    row.submitted_at = o.submitted_at or row.submitted_at
    row.filled_at = o.filled_at or row.filled_at
    row.canceled_at = o.canceled_at or o.expired_at or row.canceled_at
    if o.qty is not None:
        row.quantity = o.qty
    if o.limit_price is not None:
        row.limit_price = o.limit_price
    row.updated_at = now
    label = f"{row.side.upper()} {row.quantity or 0:g} {row.symbol}"
    if o.status != before_status or o.filled_qty > before_filled + 1e-9:
        kind = EVENT_FOR_STATUS.get(o.status)
        if kind == "order_partially_filled" or (kind is None and o.filled_qty > before_filled + 1e-9):
            kind = "order_partially_filled"
        if kind is not None:
            price = f" @ ${o.filled_avg_price:,.2f}" if o.filled_avg_price else ""
            detail = {
                "order_partially_filled": f"{label}: {o.filled_qty:g} filled so far{price}",
                "order_filled": f"{label} filled{price}",
                "order_canceled": f"{label} canceled ({o.filled_qty:g} filled)",
                "order_rejected": f"{label} rejected by Alpaca",
                "order_expired": f"{label} expired ({o.filled_qty:g} filled)",
            }[kind]
            events.append((kind, detail))
    return events


class OrderManager:
    def __init__(
        self,
        db: Database,
        broker: AlpacaPaperBroker,
        clock: Clock,
        *,
        order_timeout: timedelta = timedelta(minutes=20),
    ) -> None:
        self._db = db
        self._broker = broker
        self._clock = clock
        self._order_timeout = order_timeout

    # ------------------------------------------------------------------ submission
    async def submit(
        self,
        intent: OrderIntent,
        *,
        cid: str,
        order_type: str,
        limit_price: float | None,
        cycle_id: int | None,
        strategy: str = STRATEGY,
        notional: float | None = None,
    ) -> Submission:
        """Send one risk-approved order exactly once. ``notional`` (dollars, market orders only) replaces
        the share quantity when given."""
        now = self._clock.now()
        base = Submission(cid, intent.symbol, intent.side, PENDING_SUBMIT, submitted=False)
        # 1. write-ahead record, unique on the client order id: a repeat attempt stops here
        try:
            async with self._db.session() as s:
                existing = await repo.get_broker_order(s, cid)
                if existing is not None:
                    base.status, base.duplicate = existing.status, True
                    base.error = "an order with this client order id was already sent: not resending"
                    await repo.add_trading_event(
                        s,
                        "duplicate_prevented",
                        f"{cid}: already sent, not resending",
                        now,
                        cycle_id=cycle_id,
                        symbol=intent.symbol,
                        client_order_id=cid,
                    )
                    return base
                s.add(
                    BrokerOrderRow(
                        client_order_id=cid,
                        cycle_id=cycle_id,
                        symbol=intent.symbol,
                        side=intent.side,
                        quantity=None if notional is not None else intent.qty,
                        notional=round(notional if notional is not None else intent.notional, 2),
                        order_type=order_type,
                        time_in_force="day",
                        limit_price=limit_price,
                        status=PENDING_SUBMIT,
                        filled_quantity=0.0,
                        strategy=strategy,
                        kind=intent.kind,
                        signal_score=intent.score,
                        reason=intent.reason,
                        created_at=now,
                        updated_at=now,
                    )
                )
                await s.flush()
        except IntegrityError:
            base.duplicate, base.error = True, "a concurrent run already recorded this order: not resending"
            return base

        # 2. send it once
        order: BrokerOrder | None = None
        error: str | None = None
        status = PENDING_SUBMIT
        try:
            spec = OrderSpec(
                symbol=intent.symbol,
                side="buy" if intent.side == "buy" else "sell",
                qty=None if notional is not None else intent.qty,
                order_type="limit" if order_type in ("limit", "marketable_limit") else "market",
                client_order_id=cid,
                limit_price=limit_price,
                notional=notional,
            )
            order = await self._broker.submit(spec)
        except ValueError as exc:  # the order spec itself is invalid: nothing was sent
            status, error = SUBMIT_FAILED, f"invalid order (never sent): {exc}"
        except DuplicateClientOrderId:
            order = await self._lookup(cid)  # an earlier attempt got through: adopt it
            error = "Alpaca already had this client order id; adopted the existing order"
        except OrderRejected as exc:
            status, error = "rejected", str(exc)
        except BrokerError as exc:
            if exc.ambiguous:
                order = await self._lookup(cid)
                if order is None:
                    status, error = SUBMIT_UNKNOWN, f"{exc} — left for reconciliation, not resent"
            else:
                status, error = SUBMIT_FAILED, str(exc)
        size = f"${notional:,.2f} of" if notional is not None else f"{intent.qty:g}"

        # 3. record the outcome
        async with self._db.session() as s:
            row = await repo.get_broker_order(s, cid)
            assert row is not None
            events: list[tuple[str, str]] = []
            if order is not None:
                row.status = "new"  # so a first status already past "new" still produces its event
                events = [
                    (
                        "order_submitted",
                        f"{intent.side.upper()} {size} {intent.symbol} sent: Alpaca order {order.id} "
                        f"({order.status})",
                    )
                ]
                events += _apply(row, order, now)
                status = order.status
            else:
                row.status, row.updated_at = status, now
            row.error = error
            if status == "rejected":
                events.append(
                    (
                        "order_rejected",
                        f"{intent.side.upper()} {size} {intent.symbol} rejected: {error}",
                    )
                )
            elif status == SUBMIT_UNKNOWN:
                events.append(("order_unknown", f"{intent.side.upper()} {size} {intent.symbol}: {error}"))
            elif status == SUBMIT_FAILED:
                events.append(("order_failed", f"{intent.side.upper()} {size} {intent.symbol}: {error}"))
            for kind, message in events:
                await repo.add_trading_event(
                    s,
                    kind,
                    message,
                    now,
                    cycle_id=cycle_id,
                    symbol=intent.symbol,
                    client_order_id=cid,
                    details={
                        "kind": intent.kind,
                        "reason": intent.reason,
                        "alpaca_order_id": order.id if order is not None else None,
                        "status": status,
                    },
                )
        return Submission(
            cid, intent.symbol, intent.side, status, submitted=order is not None, order=order, error=error
        )

    async def _lookup(self, cid: str) -> BrokerOrder | None:
        try:
            return await self._broker.order_by_client_id(cid)
        except BrokerError as exc:
            logger.warning("order lookup for %s failed: %s", cid, exc)
            return None

    # ------------------------------------------------------------------ waiting for fills
    async def wait_for(
        self, cids: Sequence[str], timeout: float, poll: float = 1.0
    ) -> dict[str, BrokerOrder]:
        """Poll Alpaca until the orders are done or ``timeout`` seconds pass; rows are updated as they change."""
        pending = list(cids)
        seen: dict[str, BrokerOrder] = {}
        deadline = _time.monotonic() + max(timeout, 0.0)
        while pending:
            for cid in list(pending):
                o = await self._lookup(cid)
                if o is None:
                    continue
                seen[cid] = o
                if not o.is_open:
                    pending.remove(cid)
            await self._store(seen.values())
            if not pending or _time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll)
        return seen

    async def _store(self, orders: Iterable[BrokerOrder], cycle_id: int | None = None) -> None:
        now = self._clock.now()
        async with self._db.session() as s:
            for o in orders:
                row = await repo.get_broker_order(s, o.client_order_id)
                if row is None:
                    continue
                for kind, message in _apply(row, o, now):
                    await repo.add_trading_event(
                        s,
                        kind,
                        message,
                        now,
                        cycle_id=row.cycle_id,
                        symbol=row.symbol,
                        client_order_id=row.client_order_id,
                    )

    # ------------------------------------------------------------------ reconciliation
    async def reconcile(self) -> ReconcileReport:
        """Bring every local order in line with Alpaca (Alpaca wins) and record orders placed elsewhere."""
        now = self._clock.now()
        open_orders = await self._broker.orders("open", limit=500)
        closed = await self._broker.orders("closed", limit=500, after=now - RECONCILE_LOOKBACK)
        remote = {o.client_order_id: o for o in [*closed, *open_orders]}
        async with self._db.session() as s:
            recent = await repo.broker_orders(s, 2000, since=now - RECONCILE_LOOKBACK)
            unresolved = await repo.broker_orders(s, 2000, exclude_statuses=FINAL)
            local = {r.client_order_id: (r.status, r.created_at) for r in [*recent, *unresolved]}
        missing = [cid for cid, (status, _) in local.items() if cid not in remote and status not in FINAL]
        looked_up: dict[str, BrokerOrder | None] = {cid: await self._lookup(cid) for cid in missing}

        report = ReconcileReport(checked=len(local), open_orders=len(open_orders))
        async with self._db.session() as s:
            for cid, (status, created_at) in local.items():
                o = remote.get(cid) or looked_up.get(cid)
                row = await repo.get_broker_order(s, cid)
                if row is None:
                    continue
                if o is not None:
                    changed = (o.status, o.filled_qty) != (row.status, row.filled_quantity)
                    events = _apply(row, o, now)
                    if status in UNRESOLVED:
                        report.resolved_unknown += 1
                        events.insert(0, ("order_submitted", f"{cid} found on Alpaca during reconciliation"))
                    if changed:
                        report.updated += 1
                        report.changes.append(f"{cid}: {status} → {o.status}")
                    for kind, message in events:
                        await repo.add_trading_event(
                            s,
                            kind,
                            message,
                            now,
                            cycle_id=row.cycle_id,
                            symbol=row.symbol,
                            client_order_id=cid,
                        )
                elif status in UNRESOLVED and now - created_at >= UNKNOWN_GRACE:
                    row.status, row.updated_at = SUBMIT_FAILED, now
                    row.error = (
                        row.error or ""
                    ) + " | Alpaca has no order with this client id: it never arrived"
                    report.resolved_unknown += 1
                    report.changes.append(f"{cid}: {status} → {SUBMIT_FAILED}")
                    await repo.add_trading_event(
                        s,
                        "order_failed",
                        f"{cid} never reached Alpaca (checked after {UNKNOWN_GRACE})",
                        now,
                        cycle_id=row.cycle_id,
                        symbol=row.symbol,
                        client_order_id=cid,
                    )
            for cid, o in remote.items():
                if cid in local or await repo.get_broker_order(s, cid) is not None:
                    continue
                strategy = STRATEGY if is_ours(cid) else EXTERNAL
                row = BrokerOrderRow(
                    client_order_id=cid,
                    symbol=o.symbol,
                    side=o.side,
                    quantity=o.qty,
                    notional=o.notional,
                    order_type=o.order_type or "market",
                    time_in_force=o.time_in_force or "day",
                    limit_price=o.limit_price,
                    status=o.status,
                    filled_quantity=o.filled_qty,
                    strategy=strategy,
                    reason="found on the Alpaca paper account during reconciliation",
                    created_at=o.created_at or now,
                    updated_at=now,
                )
                _apply(row, o, now)
                s.add(row)
                report.added += 1
                report.changes.append(f"{cid}: added from Alpaca ({strategy}, {o.status})")
        return report

    # ------------------------------------------------------------------ housekeeping
    async def cancel_stale(self, open_orders: Sequence[BrokerOrder]) -> list[str]:
        """Cancel QuantPulse's own working orders older than the order timeout (manual orders are left)."""
        now = self._clock.now()
        canceled: list[str] = []
        for o in open_orders:
            placed = o.submitted_at or o.created_at
            if not is_ours(o.client_order_id) or placed is None or now - placed < self._order_timeout:
                continue
            try:
                await self._broker.cancel(o.id)
                canceled.append(o.client_order_id)
            except BrokerError as exc:
                logger.warning("could not cancel stale order %s: %s", o.client_order_id, exc)
        if canceled:
            async with self._db.session() as s:
                await repo.add_trading_event(
                    s,
                    "order_cancel_requested",
                    f"canceled {len(canceled)} unfilled order(s) older than {self._order_timeout}",
                    now,
                    details={"client_order_ids": canceled},
                )
        return canceled

    async def last_trades(self, since: datetime) -> dict[str, tuple[datetime, str]]:
        """When QuantPulse last sent an order per symbol, and its side (for the anti-churn cooldown)."""
        async with self._db.session() as s:
            rows = await repo.broker_orders(s, 2000, since=since)
        out: dict[str, tuple[datetime, str]] = {}
        for r in rows:
            if r.strategy != STRATEGY or r.status in (SUBMIT_FAILED, "rejected"):
                continue
            at = r.submitted_at or r.created_at
            if r.symbol not in out or at > out[r.symbol][0]:
                out[r.symbol] = (at, r.side)
        return out

    async def unresolved_symbols(self) -> set[str]:
        """Symbols with a local order whose fate is still unknown (never trade them until resolved)."""
        async with self._db.session() as s:
            rows = await repo.broker_orders(s, 500, statuses=list(UNRESOLVED))
        return {r.symbol for r in rows}
