"""Risk-free rate service backed by the live Treasury par yield curve."""

from __future__ import annotations

from datetime import datetime

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.providers import synthetic
from quantpulse.providers.treasury import Treasury
from quantpulse.quant.rates import bey_to_continuous, interpolate_rate
from quantpulse.schemas.options import RateAtTenor, YieldCurve


class RatesService:
    def __init__(
        self, settings: Settings, gateway: DataGateway, db: Database, clock: Clock, treasury: Treasury
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._treasury = treasury

    async def curve(self, *, force_refresh: bool = False) -> Resolved[YieldCurve]:
        async def persist(curve: YieldCurve, provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_curve(s, curve, provider)
                await repo.record_ingestion(s, "yield_curve", curve.as_of.isoformat(), provider, rows)

        async def archive() -> tuple[YieldCurve, datetime, str] | None:
            async with self._db.session() as s:
                return await repo.latest_curve(s)

        return await self._gw.resolve(
            "rates:curve",
            [Source(self._treasury.name, self._treasury.curve)],
            lambda: synthetic.synthetic_curve(self._clock.now().date()),
            self._settings.ttl_yield_curve,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )

    @staticmethod
    def rate_from_curve(curve: YieldCurve, years: float) -> RateAtTenor:
        bey = interpolate_rate(curve.tenors, curve.rates, years)
        return RateAtTenor(
            years=years, bey_rate=bey, continuous_rate=bey_to_continuous(bey), curve_date=curve.as_of
        )

    async def rate_at(self, years: float) -> tuple[RateAtTenor, Resolved[YieldCurve]]:
        resolved = await self.curve()
        return self.rate_from_curve(resolved.value, years), resolved
