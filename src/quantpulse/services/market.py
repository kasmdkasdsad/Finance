"""Market data service: quotes, OHLCV history and dividend yields through the data gateway."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.providers import synthetic
from quantpulse.providers.yahoo import YahooFinance
from quantpulse.schemas.market import INTRADAY_INTERVALS, Interval, PriceHistory, Quote

# One daily-history window shared by picks, forecasts, the stock model and reports, so each symbol is
# fetched (and cached) once instead of once per feature.
STANDARD_HISTORY_DAYS = 1825


class MarketProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    async def quote(self, symbol: str) -> Quote: ...

    async def history(
        self, symbol: str, interval: Interval, start: datetime, end: datetime
    ) -> PriceHistory: ...


class MarketService:
    def __init__(
        self,
        settings: Settings,
        gateway: DataGateway,
        db: Database,
        clock: Clock,
        providers: Sequence[MarketProvider],
        yahoo: YahooFinance,
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._providers = list(providers)
        self._yahoo = yahoo
        self._history_sem = asyncio.Semaphore(6)

    @property
    def providers(self) -> list[MarketProvider]:
        return self._providers

    def _sources(self, fetch_name: str, *args: object) -> list[Source]:
        sources: list[Source] = []
        for p in self._providers:
            fetch = functools.partial(getattr(p, fetch_name), *args)
            sources.append(Source(p.name, fetch, configured=p.configured()))
        return sources

    async def quote(self, symbol: str, *, force_refresh: bool = False) -> Resolved[Quote]:
        async def persist(q: Quote, provider: str) -> None:
            async with self._db.session() as s:
                await repo.insert_quote(s, q, provider)

        async def archive() -> tuple[Quote, datetime, str] | None:
            async with self._db.session() as s:
                return await repo.latest_quote(s, symbol)

        return await self._gw.resolve(
            f"quote:{symbol}",
            self._sources("quote", symbol),
            lambda: synthetic.synthetic_quote(symbol, self._clock.now()),
            self._settings.ttl_quote,
            as_of=lambda q: q.timestamp,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )

    async def quotes(
        self, symbols: Sequence[str], *, force_refresh: bool = False
    ) -> dict[str, Resolved[Quote]]:
        results = await asyncio.gather(*(self.quote(s, force_refresh=force_refresh) for s in symbols))
        return dict(zip(symbols, results, strict=True))

    async def history(
        self, symbol: str, interval: Interval = "1d", lookback_days: int = 365, *, force_refresh: bool = False
    ) -> Resolved[PriceHistory]:
        end = self._clock.now()
        start = end - timedelta(days=lookback_days)
        intraday = interval in INTRADAY_INTERVALS
        ttl = self._settings.ttl_bars_intraday if intraday else self._settings.ttl_bars_daily

        async def persist(h: PriceHistory, provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_bars(s, h, provider)
                await repo.record_ingestion(s, "bars", f"{symbol}:{interval}", provider, rows)

        async def archive() -> tuple[PriceHistory, datetime, str] | None:
            async with self._db.session() as s:
                found = await repo.load_bars(s, symbol, interval, since=start)
            if found is None or len(found[0].bars) < 5:
                return None
            return found

        async with self._history_sem:
            return await self._gw.resolve(
                f"bars:{symbol}:{interval}:{lookback_days}",
                self._sources("history", symbol, interval, start, end),
                lambda: synthetic.synthetic_history(symbol, interval, start, end, self._clock.now()),
                ttl,
                as_of=lambda h: h.bars[-1].timestamp if h.bars else None,
                archive=archive,
                on_live=persist,
                force_refresh=force_refresh,
            )

    async def dividend_yield(self, symbol: str) -> Resolved[float]:
        async def fetch() -> float:
            value = await self._yahoo.dividend_yield(symbol)
            return value or 0.0

        return await self._gw.resolve(
            f"divyield:{symbol}",
            [Source(self._yahoo.name, fetch)],
            lambda: synthetic.synthetic_quote(symbol, self._clock.now()).dividend_yield or 0.0,
            self._settings.ttl_fundamentals,
        )
