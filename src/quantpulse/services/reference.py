"""Reference data service: S&P 500 membership, company profiles/sectors and earnings events.

Reference data changes slowly, so it is read from the warehouse first while it is fresh
(``QP_TTL_REFERENCE``, default 7 days) and only refreshed from the source after that — a restart
never re-downloads hundreds of SEC profiles. Fallbacks follow the platform's usual chain; the S&P 500
falls back to the snapshot packaged with QuantPulse (real, but dated: labelled STALE).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date, datetime, timedelta

import numpy as np

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import ProviderError
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.domain import earnings as earn
from quantpulse.domain.universe import Constituent, IndexChange, Membership
from quantpulse.providers import sp500, synthetic
from quantpulse.providers.fmp import FinancialModelingPrep
from quantpulse.providers.sec_edgar import SecEdgar
from quantpulse.schemas.common import DataStatus, Provenance
from quantpulse.schemas.reference import CompanyEvents, EarningsOut, EarningsReaction
from quantpulse.services.market import STANDARD_HISTORY_DAYS, MarketService

EVENTS_LOOKBACK_DAYS = 6 * 365
MEMBERSHIP_KEY = "sp500"
CONCURRENCY = 6

Snapshot = tuple[list[Constituent], list[IndexChange]]


def _snapshot_payload(snap: Snapshot) -> dict[str, str]:
    return {"constituents": sp500.constituents_to_csv(snap[0]), "changes": sp500.changes_to_csv(snap[1])}


def _snapshot_from_payload(payload: dict[str, str]) -> Snapshot:
    return sp500.constituents_from_csv(payload["constituents"]), sp500.changes_from_csv(payload["changes"])


class ReferenceService:
    def __init__(
        self,
        settings: Settings,
        gateway: DataGateway,
        db: Database,
        clock: Clock,
        market: MarketService,
        sec: SecEdgar,
        wiki: sp500.SP500Wikipedia,
        fmp: FinancialModelingPrep,
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._market = market
        self._sec = sec
        self._wiki = wiki
        self._fmp = fmp

    def _fresh(self, at: datetime) -> bool:
        return (self._clock.now() - at).total_seconds() < self._settings.ttl_reference

    def _cached(self, at: datetime, provider: str) -> Provenance:
        return Provenance(status=DataStatus.CACHED, provider=provider, as_of=at, fetched_at=at)

    # ------------------------------------------------------------------ S&P 500 membership
    async def membership(self, *, force_refresh: bool = False) -> Resolved[Membership]:
        async with self._db.session() as s:
            stored = await repo.get_blob(s, MEMBERSHIP_KEY)
        if stored is not None and not force_refresh and self._fresh(stored[1]):
            payload, at, provider = stored
            snap = _snapshot_from_payload(payload)
            return Resolved(Membership(*snap, as_of=at.date()), self._cached(at, provider))

        async def persist(snap: Snapshot, provider: str) -> None:
            async with self._db.session() as s:
                await repo.put_blob(s, MEMBERSHIP_KEY, _snapshot_payload(snap), provider)

        async def archive() -> tuple[Snapshot, datetime, str] | None:
            if stored is None:
                return None
            return _snapshot_from_payload(stored[0]), stored[1], stored[2]

        resolved = await self._gw.resolve(
            "reference:sp500",
            [Source(self._wiki.name, self._wiki.snapshot)],
            sp500.packaged_snapshot,
            self._settings.ttl_reference,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
            fallback_provider="packaged snapshot",
            fallback_status=DataStatus.STALE,
        )
        cons, changes = resolved.value
        as_of = resolved.provenance.as_of.date() if resolved.status is not DataStatus.STALE else None
        return Resolved(Membership(cons, changes, as_of=as_of), resolved.provenance)

    # ------------------------------------------------------------------ company events
    async def events(self, symbol: str, *, force_refresh: bool = False) -> Resolved[CompanyEvents]:
        since = self._clock.now().astimezone(NEW_YORK).date() - timedelta(days=EVENTS_LOOKBACK_DAYS)
        async with self._db.session() as s:
            stored = await repo.load_company_events(s, symbol)
        if (
            stored is not None
            and not force_refresh
            and self._fresh(stored[1])
            and stored[0].earnings_since <= since + timedelta(days=30)
            and (stored[2] != "synthetic" or not self._settings.enable_live_data)
        ):
            return Resolved(stored[0], self._cached(stored[1], stored[2]))

        async def fetch() -> CompanyEvents:
            return await self._sec.company_events(symbol, since)

        async def persist(ev: CompanyEvents, provider: str) -> None:
            async with self._db.session() as s:
                await repo.save_company_events(s, ev, provider)

        async def archive() -> tuple[CompanyEvents, datetime, str] | None:
            return stored

        return await self._gw.resolve(
            f"reference:events:{symbol}",
            [Source(self._sec.name, fetch)],
            lambda: synthetic.synthetic_company_events(symbol, since, self._clock.now()),
            self._settings.ttl_reference,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )

    async def events_many(
        self, symbols: Sequence[str], *, simulated: bool = False
    ) -> dict[str, Resolved[CompanyEvents]]:
        """Events for many symbols; ``simulated`` returns simulated companies (for synthetic price panels)."""
        if simulated:
            now = self._clock.now()
            since = now.astimezone(NEW_YORK).date() - timedelta(days=EVENTS_LOOKBACK_DAYS)
            prov = Provenance(status=DataStatus.SYNTHETIC, provider="synthetic", as_of=now, fetched_at=now)

            def simulate() -> dict[str, Resolved[CompanyEvents]]:
                return {s: Resolved(synthetic.synthetic_company_events(s, since, now), prov) for s in symbols}

            return await asyncio.to_thread(simulate)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(symbol: str) -> Resolved[CompanyEvents]:
            async with sem:
                return await self.events(symbol)

        results = await asyncio.gather(*(one(s) for s in symbols))
        return dict(zip(symbols, results, strict=True))

    # ------------------------------------------------------------------ earnings view
    async def next_scheduled(self, symbol: str, today: date) -> date | None:
        if not (self._settings.enable_live_data and self._fmp.configured()):
            return None
        try:
            return await self._fmp.next_earnings(symbol, today)
        except ProviderError:
            return None

    async def earnings(self, symbol: str) -> tuple[EarningsOut, Resolved[CompanyEvents]]:
        bench = self._settings.benchmark_symbol
        ev_r, hist_r, bench_r = await asyncio.gather(
            self.events(symbol),
            self._market.history(symbol, "1d", STANDARD_HISTORY_DAYS),
            self._market.history(bench, "1d", STANDARD_HISTORY_DAYS),
        )
        today = self._clock.now().astimezone(NEW_YORK).date()
        closes = {b.timestamp.astimezone(NEW_YORK).date(): b.close for b in hist_r.value.bars}
        bench_closes = {b.timestamp.astimezone(NEW_YORK).date(): b.close for b in bench_r.value.bars}
        events = ev_r.value.earnings
        rs = earn.reactions(events, closes, bench_closes)
        scheduled = await self.next_scheduled(symbol, today)
        estimated = earn.estimate_next(events, today)
        nxt, source = (
            (scheduled, "scheduled") if scheduled else (estimated, "estimated" if estimated else None)
        )
        reaction = None
        if nxt is not None:  # estimates are reaction days already; calendar dates need the company's habit
            reaction = earn.reaction_for_announcement(nxt, events) if source == "scheduled" else nxt
        out = EarningsOut(
            symbol=symbol,
            last=events[-1] if events else None,
            next_date=nxt,
            next_source=source,
            next_reaction_date=reaction,
            days_to_next=(nxt - today).days if nxt else None,
            typical_move=earn.typical_move(rs),
            reactions=[
                EarningsReaction(
                    announced_at=r.announced_at,
                    reaction_date=r.reaction_date,
                    stock_return=r.stock_return,
                    benchmark_return=r.benchmark_return,
                    abnormal_return=r.abnormal,
                )
                for r in rs[-earn.MAX_REACTIONS :]
            ],
        )
        return out, ev_r


def reaction_array(rs: Sequence[earn.Reaction]) -> np.ndarray:
    """Historical earnings-day log returns (the jump sample used by the forecaster)."""
    return np.log1p(np.array([r.stock_return for r in rs], dtype=float))
