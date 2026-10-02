"""Fundamentals (SEC EDGAR) and consensus estimates (FMP → Yahoo) service."""

from __future__ import annotations

from datetime import datetime

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.providers import synthetic
from quantpulse.providers.fmp import FinancialModelingPrep
from quantpulse.providers.sec_edgar import SecEdgar
from quantpulse.providers.yahoo import YahooFinance
from quantpulse.schemas.fundamentals import AnalystEstimates, CompanyFundamentals


class FundamentalsService:
    def __init__(
        self,
        settings: Settings,
        gateway: DataGateway,
        db: Database,
        clock: Clock,
        sec: SecEdgar,
        fmp: FinancialModelingPrep,
        yahoo: YahooFinance,
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._sec = sec
        self._fmp = fmp
        self._yahoo = yahoo

    async def fundamentals(
        self, symbol: str, *, force_refresh: bool = False
    ) -> Resolved[CompanyFundamentals]:
        async def persist(data: CompanyFundamentals, provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_fundamentals(s, data, provider)
                await repo.record_ingestion(s, "fundamentals", symbol, provider, rows)

        async def archive() -> tuple[CompanyFundamentals, datetime, str] | None:
            async with self._db.session() as s:
                return await repo.load_fundamentals(s, symbol)

        return await self._gw.resolve(
            f"fundamentals:{symbol}",
            [Source(self._sec.name, lambda: self._sec.fundamentals(symbol))],
            lambda: synthetic.synthetic_fundamentals(symbol, self._clock.now().date()),
            self._settings.ttl_fundamentals,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )

    async def estimates(self, symbol: str, *, force_refresh: bool = False) -> Resolved[AnalystEstimates]:
        async def persist(data: AnalystEstimates, provider: str) -> None:
            async with self._db.session() as s:
                await repo.insert_estimates(s, data, provider)

        async def archive() -> tuple[AnalystEstimates, datetime, str] | None:
            async with self._db.session() as s:
                return await repo.latest_estimates(s, symbol)

        return await self._gw.resolve(
            f"estimates:{symbol}",
            [
                Source(
                    self._fmp.name, lambda: self._fmp.estimates(symbol), configured=self._fmp.configured()
                ),
                Source(self._yahoo.name, lambda: self._yahoo.estimates(symbol)),
            ],
            lambda: synthetic.synthetic_estimates(symbol, self._clock.now().date()),
            self._settings.ttl_estimates,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )
