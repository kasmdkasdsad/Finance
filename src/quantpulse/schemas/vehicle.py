"""Vehicle asset lifecycle schemas: profile, telemetry, fuel, maintenance and cost analytics."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from quantpulse.schemas.common import StrictModel

FuelGrade = Literal["regular", "midgrade", "premium", "diesel"]


class EPARating(StrictModel):
    vehicle_id: int | None = None
    city_mpg: float = Field(gt=0)
    highway_mpg: float = Field(gt=0)
    combined_mpg: float = Field(gt=0)
    source: str


class Pricing(StrictModel):
    base_msrp: float = Field(gt=0)
    destination: float = Field(ge=0)
    source: str

    @property
    def total_msrp(self) -> float:
        return self.base_msrp + self.destination


class DepreciationParams(StrictModel):
    first_year_rate: float = Field(ge=0, lt=1)
    annual_rate: float = Field(ge=0, lt=1)
    annual_miles_baseline: float = Field(gt=0)
    mileage_adjustment_per_1000: float = Field(ge=0, le=0.05)
    max_mileage_adjustment: float = Field(ge=0, le=0.9)
    floor_fraction: float = Field(ge=0, lt=1)
    source: str


class MaintenanceItem(StrictModel):
    code: str = Field(pattern=r"^[a-z0-9_]{2,40}$")
    name: str
    interval_miles: float | None = Field(default=None, gt=0)
    interval_months: int | None = Field(default=None, gt=0)
    estimated_cost: float = Field(ge=0)

    @model_validator(mode="after")
    def _has_interval(self) -> MaintenanceItem:
        if self.interval_miles is None and self.interval_months is None:
            raise ValueError("a maintenance item needs a mileage and/or time interval")
        return self


class VehicleProfile(StrictModel):
    profile_id: str
    year: int
    make: str
    model: str
    trim: str
    engine: str
    transmission: str
    horsepower: int | None = None
    torque_lb_ft: int | None = None
    fuel_grade: FuelGrade
    tank_gallons: float = Field(gt=0)
    epa: EPARating
    pricing: Pricing
    depreciation: DepreciationParams
    maintenance_source: str
    maintenance: list[MaintenanceItem] = Field(min_length=1)


class VehicleCreate(StrictModel):
    nickname: str = Field(min_length=1, max_length=80)
    profile_id: str = "hyundai-elantra-2025-limited"
    purchase_price: float | None = Field(default=None, gt=0, description="Defaults to MSRP + destination.")
    purchase_date: date
    purchase_odometer: float = Field(default=0.0, ge=0)
    annual_miles: float = Field(default=12000.0, gt=0, le=200000)
    city_share: float = Field(default=0.55, ge=0, le=1, description="Share of miles driven in the city.")
    fuel_region: str | None = Field(default=None, description="EIA duoarea code, e.g. NUS, R1Z, SFL.")


class VehicleOut(VehicleCreate):
    id: int
    created_at: AwareDatetime


class TelemetryIn(StrictModel):
    recorded_at: AwareDatetime
    odometer: float = Field(ge=0, le=2_000_000)
    fuel_level_pct: float | None = Field(default=None, ge=0, le=100)
    source: str = Field(default="manual", max_length=40)


class TelemetryOut(TelemetryIn):
    id: int


class FuelLogIn(StrictModel):
    filled_at: AwareDatetime
    odometer: float = Field(ge=0, le=2_000_000)
    gallons: float = Field(gt=0, le=100)
    price_per_gallon: float = Field(gt=0, le=20)
    full_tank: bool = True
    station: str | None = Field(default=None, max_length=120)


class FuelLogOut(FuelLogIn):
    id: int
    total_cost: float


class MaintenanceRecordIn(StrictModel):
    service_code: str = Field(pattern=r"^[a-z0-9_]{2,40}$")
    performed_on: date
    odometer: float = Field(ge=0, le=2_000_000)
    cost: float = Field(default=0.0, ge=0)
    notes: str | None = Field(default=None, max_length=500)


class MaintenanceRecordOut(MaintenanceRecordIn):
    id: int


class FuelPrice(StrictModel):
    region: str
    region_name: str
    grade: FuelGrade
    price: float = Field(gt=0, description="USD per gallon")
    period: date
    series_id: str | None = None


class FuelPriceSeries(StrictModel):
    region: str
    region_name: str
    grade: FuelGrade
    latest: FuelPrice
    history: list[FuelPrice]


class MaintenanceDue(StrictModel):
    code: str
    name: str
    status: Literal["overdue", "due_soon", "ok"]
    last_service_odometer: float
    last_service_date: date
    next_due_odometer: float | None
    next_due_date: date | None
    miles_remaining: float | None
    days_remaining: int | None
    projected_due_date: date | None
    estimated_cost: float


class DepreciationPoint(StrictModel):
    on: date
    age_years: float
    expected_odometer: float
    value: float


class CostPerMile(StrictModel):
    fuel: float
    depreciation: float
    maintenance: float
    total: float


class VehicleDashboard(StrictModel):
    vehicle: VehicleOut
    profile: VehicleProfile
    epa_live: EPARating
    fuel_price: FuelPrice
    odometer: float
    telemetry_source: Literal["recorded", "synthetic"]
    avg_daily_miles: float
    realized_mpg: float | None
    effective_mpg: float
    current_value: float
    total_depreciation: float
    depreciation_per_day: float
    cost_per_mile: CostPerMile
    projected_monthly_cost: dict[str, float]
    projected_annual_cost: dict[str, float]
    depreciation_curve: list[DepreciationPoint]
    maintenance: list[MaintenanceDue]
    fuel_spend_to_date: float
    warnings: list[str] = Field(default_factory=list)
