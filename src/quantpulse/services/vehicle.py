"""Asset lifecycle & operating-cost service for the vehicle module."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.core.market_calendar import NEW_YORK
from quantpulse.db import repositories as repo
from quantpulse.db.models import FuelLogRow, MaintenanceRecordRow, TelemetryRow, VehicleRow
from quantpulse.db.session import Database
from quantpulse.domain import vehicle as domain
from quantpulse.providers import synthetic
from quantpulse.providers.eia import EIA, REGIONS, validate_region
from quantpulse.providers.fueleconomy import FuelEconomyGov
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus
from quantpulse.schemas.vehicle import (
    CostPerMile,
    EPARating,
    FuelLogIn,
    FuelLogOut,
    FuelPrice,
    FuelPriceSeries,
    MaintenanceRecordIn,
    MaintenanceRecordOut,
    TelemetryIn,
    TelemetryOut,
    VehicleCreate,
    VehicleDashboard,
    VehicleOut,
    VehicleProfile,
)


def _vehicle_out(row: VehicleRow) -> VehicleOut:
    return VehicleOut(
        id=row.id,
        nickname=row.nickname,
        profile_id=row.profile_id,
        purchase_price=row.purchase_price,
        purchase_date=row.purchase_date,
        purchase_odometer=row.purchase_odometer,
        annual_miles=row.annual_miles,
        city_share=row.city_share,
        fuel_region=row.fuel_region,
        created_at=row.created_at,
    )


class VehicleService:
    def __init__(
        self,
        settings: Settings,
        gateway: DataGateway,
        db: Database,
        clock: Clock,
        eia: EIA,
        fueleconomy: FuelEconomyGov,
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._eia = eia
        self._fe = fueleconomy

    # ------------------------------------------------------------------ reference data
    @staticmethod
    def profile(profile_id: str) -> VehicleProfile:
        return domain.load_profile(profile_id)

    async def epa(self, profile: VehicleProfile) -> Resolved[EPARating]:
        vehicle_id = profile.epa.vehicle_id
        sources = (
            [Source(self._fe.name, lambda: self._fe.epa_rating(vehicle_id))] if vehicle_id is not None else []
        )
        return await self._gw.resolve(
            f"epa:{vehicle_id}",
            sources,
            lambda: profile.epa,
            self._settings.ttl_vehicle_specs,
            fallback_provider="packaged-profile",
            fallback_status=DataStatus.STALE,
        )

    async def fuel_prices(
        self, region: str | None = None, grade: str | None = None
    ) -> Resolved[FuelPriceSeries]:
        code = validate_region(region or self._settings.fuel_region)
        grade = grade or self._settings.fuel_grade

        async def persist(series: FuelPriceSeries, provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.upsert_fuel_prices(
                    s,
                    code,
                    series.region_name,
                    grade,
                    [(p.period, p.price, p.series_id) for p in series.history],
                    provider,
                )
                await repo.record_ingestion(s, "fuel_prices", f"{code}:{grade}", provider, rows)

        async def archive() -> tuple[FuelPriceSeries, datetime, str] | None:
            async with self._db.session() as s:
                rows = await repo.fuel_price_history(s, code, grade)
            if not rows:
                return None
            history = [
                FuelPrice(
                    region=code,
                    region_name=r.region_name,
                    grade=grade,
                    price=r.price,
                    period=r.period,
                    series_id=r.series_id,
                )
                for r in rows
            ]
            series = FuelPriceSeries(
                region=code,
                region_name=history[-1].region_name,
                grade=grade,
                latest=history[-1],
                history=history,
            )
            return series, max(r.ingested_at for r in rows), rows[-1].provider

        return await self._gw.resolve(
            f"fuel:{code}:{grade}",
            [
                Source(
                    self._eia.name,
                    lambda: self._eia.fuel_prices(code, grade),
                    configured=self._eia.configured(),
                )
            ],
            lambda: synthetic.synthetic_fuel_series(code, REGIONS[code], grade, self._clock.now().date()),
            self._settings.ttl_fuel_prices,
            as_of=lambda s: datetime.combine(s.latest.period, time(0), NEW_YORK),
            archive=archive,
            on_live=persist,
        )

    # ------------------------------------------------------------------ CRUD
    async def create(self, data: VehicleCreate) -> VehicleOut:
        domain.load_profile(data.profile_id)  # validates the profile id
        if data.fuel_region:
            validate_region(data.fuel_region)
        async with self._db.session() as s:
            row = await repo.create_vehicle(
                s,
                nickname=data.nickname,
                profile_id=data.profile_id,
                purchase_price=data.purchase_price,
                purchase_date=data.purchase_date,
                purchase_odometer=data.purchase_odometer,
                annual_miles=data.annual_miles,
                city_share=data.city_share,
                fuel_region=data.fuel_region.upper() if data.fuel_region else None,
            )
            return _vehicle_out(row)

    async def list_all(self) -> list[VehicleOut]:
        async with self._db.session() as s:
            return [_vehicle_out(r) for r in await repo.list_vehicles(s)]

    async def get(self, vehicle_id: int) -> VehicleOut:
        async with self._db.session() as s:
            return _vehicle_out(await repo.get_vehicle(s, vehicle_id))

    async def delete(self, vehicle_id: int) -> None:
        async with self._db.session() as s:
            await repo.delete_vehicle(s, vehicle_id)

    async def add_telemetry(self, vehicle_id: int, data: TelemetryIn) -> TelemetryOut:
        async with self._db.session() as s:
            await repo.get_vehicle(s, vehicle_id)
            row = await repo.add_row(s, TelemetryRow(vehicle_id=vehicle_id, **data.model_dump()))
            return TelemetryOut(id=row.id, **data.model_dump())

    async def telemetry(self, vehicle_id: int) -> list[TelemetryOut]:
        async with self._db.session() as s:
            await repo.get_vehicle(s, vehicle_id)
            return [
                TelemetryOut(
                    id=r.id,
                    recorded_at=r.recorded_at,
                    odometer=r.odometer,
                    fuel_level_pct=r.fuel_level_pct,
                    source=r.source,
                )
                for r in await repo.telemetry(s, vehicle_id)
            ]

    async def add_fuel_log(self, vehicle_id: int, data: FuelLogIn) -> FuelLogOut:
        async with self._db.session() as s:
            await repo.get_vehicle(s, vehicle_id)
            row = await repo.add_row(s, FuelLogRow(vehicle_id=vehicle_id, **data.model_dump()))
            return FuelLogOut(
                id=row.id, total_cost=round(data.gallons * data.price_per_gallon, 2), **data.model_dump()
            )

    async def fuel_logs(self, vehicle_id: int) -> list[FuelLogOut]:
        async with self._db.session() as s:
            await repo.get_vehicle(s, vehicle_id)
            return [
                FuelLogOut(
                    id=r.id,
                    filled_at=r.filled_at,
                    odometer=r.odometer,
                    gallons=r.gallons,
                    price_per_gallon=r.price_per_gallon,
                    full_tank=r.full_tank,
                    station=r.station,
                    total_cost=round(r.gallons * r.price_per_gallon, 2),
                )
                for r in await repo.fuel_logs(s, vehicle_id)
            ]

    async def add_maintenance(self, vehicle_id: int, data: MaintenanceRecordIn) -> MaintenanceRecordOut:
        async with self._db.session() as s:
            vehicle = await repo.get_vehicle(s, vehicle_id)
            codes = {i.code for i in domain.load_profile(vehicle.profile_id).maintenance}
            if data.service_code not in codes:
                from quantpulse.core.errors import DomainError

                raise DomainError(
                    f"unknown service_code '{data.service_code}'. Valid: {', '.join(sorted(codes))}"
                )
            row = await repo.add_row(s, MaintenanceRecordRow(vehicle_id=vehicle_id, **data.model_dump()))
            return MaintenanceRecordOut(id=row.id, **data.model_dump())

    async def maintenance_records(self, vehicle_id: int) -> list[MaintenanceRecordOut]:
        async with self._db.session() as s:
            await repo.get_vehicle(s, vehicle_id)
            return [
                MaintenanceRecordOut(
                    id=r.id,
                    service_code=r.service_code,
                    performed_on=r.performed_on,
                    odometer=r.odometer,
                    cost=r.cost,
                    notes=r.notes,
                )
                for r in await repo.maintenance_records(s, vehicle_id)
            ]

    async def delete_record(self, vehicle_id: int, kind: str, record_id: int) -> None:
        model = {"telemetry": TelemetryRow, "fuel-logs": FuelLogRow, "maintenance": MaintenanceRecordRow}[
            kind
        ]
        async with self._db.session() as s:
            await repo.delete_child(s, model, vehicle_id, record_id)

    # ------------------------------------------------------------------ analytics
    async def dashboard(self, vehicle_id: int) -> CompositeEnvelope[VehicleDashboard]:
        async with self._db.session() as s:
            row = await repo.get_vehicle(s, vehicle_id)
            vehicle = _vehicle_out(row)
            tele_rows = await repo.telemetry(s, vehicle_id)
            fuel_rows = await repo.fuel_logs(s, vehicle_id)
            maint_rows = await repo.maintenance_records(s, vehicle_id)
        profile = domain.load_profile(vehicle.profile_id)
        epa_r, fuel_r = await asyncio.gather(
            self.epa(profile),
            self.fuel_prices(vehicle.fuel_region or self._settings.fuel_region, profile.fuel_grade),
        )
        now = self._clock.now()
        today = now.astimezone(NEW_YORK).date()
        warnings: list[str] = []

        readings: list[tuple[datetime, float]] = [(r.recorded_at, r.odometer) for r in tele_rows]
        readings += [(r.filled_at, r.odometer) for r in fuel_rows]
        readings += [(datetime.combine(r.performed_on, time(12), NEW_YORK), r.odometer) for r in maint_rows]
        if readings:
            telemetry_source = "recorded"
        else:
            telemetry_source = "synthetic"
            warnings.append("No telemetry, fuel or service records yet — odometer and mileage are simulated.")
            readings = [
                (datetime.combine(d, time(18), NEW_YORK), odo)
                for d, odo in domain.synthetic_odometer_readings(
                    vehicle.purchase_date, vehicle.purchase_odometer, vehicle.annual_miles, today
                )
            ]
        odometer = max([vehicle.purchase_odometer, *(o for _, o in readings)])
        avg_daily = domain.average_daily_miles(readings, vehicle.annual_miles / 365.0)

        fills = [domain.FuelFill(r.odometer, r.gallons, r.full_tank, r.price_per_gallon) for r in fuel_rows]
        realized = domain.realized_mpg(fills)
        epa = epa_r.value
        if realized is not None:
            effective = realized
        else:
            effective = domain.calibrated_mpg(
                epa.city_mpg, epa.highway_mpg, epa.combined_mpg, vehicle.city_share
            )
            warnings.append(
                "Fuel economy uses EPA ratings adjusted to your city/highway mix until two full-tank fill-ups are logged."
            )

        price = vehicle.purchase_price or profile.pricing.total_msrp
        params = profile.depreciation
        age = max((today - vehicle.purchase_date).days, 0) / 365.25
        driven = max(odometer - vehicle.purchase_odometer, 0.0)
        value_now = domain.depreciated_value(price, age, driven, params)
        value_tomorrow = domain.depreciated_value(price, age + 1 / 365.25, driven + avg_daily, params)
        value_next_year = domain.depreciated_value(price, age + 1.0, driven + vehicle.annual_miles, params)

        fuel_price = fuel_r.value.latest.price
        fuel_cpm = fuel_price / effective
        dep_cpm = max(value_now - value_next_year, 0.0) / vehicle.annual_miles
        maint_annual = domain.annual_maintenance_cost(profile.maintenance, vehicle.annual_miles)
        maint_cpm = maint_annual / vehicle.annual_miles
        cpm = CostPerMile(
            fuel=fuel_cpm, depreciation=dep_cpm, maintenance=maint_cpm, total=fuel_cpm + dep_cpm + maint_cpm
        )
        annual = {
            "fuel": fuel_cpm * vehicle.annual_miles,
            "depreciation": dep_cpm * vehicle.annual_miles,
            "maintenance": maint_annual,
        }
        annual["total"] = sum(annual.values())
        monthly = {k: v / 12.0 for k, v in annual.items()}

        schedule = domain.maintenance_schedule(
            profile.maintenance,
            [domain.ServiceRecord(r.service_code, r.performed_on, r.odometer) for r in maint_rows],
            odometer,
            today,
            avg_daily,
            vehicle.purchase_date,
            vehicle.purchase_odometer,
        )
        if epa_r.status is not DataStatus.LIVE and epa_r.status is not DataStatus.CACHED:
            warnings.append("fueleconomy.gov unreachable — using the packaged EPA ratings.")

        dash = VehicleDashboard(
            vehicle=vehicle,
            profile=profile,
            epa_live=epa,
            fuel_price=fuel_r.value.latest,
            odometer=round(odometer, 1),
            telemetry_source=telemetry_source,
            avg_daily_miles=round(avg_daily, 2),
            realized_mpg=None if realized is None else round(realized, 2),
            effective_mpg=round(effective, 2),
            current_value=round(value_now, 2),
            total_depreciation=round(price - value_now, 2),
            depreciation_per_day=round(max(value_now - value_tomorrow, 0.0), 2),
            cost_per_mile=cpm,
            projected_monthly_cost={k: round(v, 2) for k, v in monthly.items()},
            projected_annual_cost={k: round(v, 2) for k, v in annual.items()},
            depreciation_curve=domain.depreciation_curve(
                price, vehicle.purchase_date, vehicle.purchase_odometer, vehicle.annual_miles, params
            ),
            maintenance=schedule,
            fuel_spend_to_date=round(sum(r.gallons * r.price_per_gallon for r in fuel_rows), 2),
            warnings=warnings,
        )
        sources = {"epa": epa_r.provenance, "fuel_prices": fuel_r.provenance}
        return CompositeEnvelope(data=dash, meta=CompositeMeta.from_sources(sources, now))

    @staticmethod
    def default_purchase_date(today: date) -> date:
        return today - timedelta(days=180)
