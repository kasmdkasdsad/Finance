"""Vehicle operating-cost, depreciation and maintenance analytics (pure functions)."""

from __future__ import annotations

import calendar
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from importlib import resources

import numpy as np

from quantpulse.core.errors import DomainError, NotFoundError
from quantpulse.schemas.vehicle import (
    DepreciationParams,
    DepreciationPoint,
    MaintenanceDue,
    MaintenanceItem,
    VehicleProfile,
)

PROFILE_FILES = {"hyundai-elantra-2025-limited": "elantra_2025_limited.json"}
DUE_SOON_DAYS = 30
DUE_SOON_MIN_MILES = 500.0


def load_profile(profile_id: str) -> VehicleProfile:
    filename = PROFILE_FILES.get(profile_id)
    if filename is None:
        raise NotFoundError(f"unknown vehicle profile '{profile_id}'")
    raw = resources.files("quantpulse.data").joinpath(filename).read_text(encoding="utf-8")
    return VehicleProfile.model_validate_json(raw)


def add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def blended_mpg(city_mpg: float, highway_mpg: float, city_share: float) -> float:
    """Distance-weighted harmonic mean (the EPA combines ratings the same way with a 55/45 split)."""
    if not 0 <= city_share <= 1:
        raise DomainError("city_share must be between 0 and 1")
    if city_mpg <= 0 or highway_mpg <= 0:
        raise DomainError("mpg values must be positive")
    return 1.0 / (city_share / city_mpg + (1.0 - city_share) / highway_mpg)


EPA_CITY_SHARE = 0.55


def calibrated_mpg(city_mpg: float, highway_mpg: float, combined_mpg: float, city_share: float) -> float:
    """Driving-mix fuel economy anchored to the official combined rating.

    EPA label values are rounded, so the harmonic 55/45 blend of the *rounded* city/highway figures can
    miss the published combined value by up to ~1 mpg (2025 Elantra Limited: 30/39 blends to 33.5 while
    the label says 34). Scaling the blend by ``combined / blend(55%)`` reproduces the official combined
    number exactly at the EPA mix and keeps the city/highway sensitivity for any other mix.
    """
    reference = blended_mpg(city_mpg, highway_mpg, EPA_CITY_SHARE)
    return blended_mpg(city_mpg, highway_mpg, city_share) * (combined_mpg / reference)


@dataclass(frozen=True, slots=True)
class FuelFill:
    odometer: float
    gallons: float
    full_tank: bool
    price_per_gallon: float


def realized_mpg(fills: Sequence[FuelFill]) -> float | None:
    """Fuel economy from fill-ups using the full-tank method.

    Between two consecutive full-tank fills, the miles driven divided by all gallons pumped after the
    first full fill (including partial fills) is the true consumption. Returns ``None`` if fewer than two
    full-tank fills exist.
    """
    ordered = sorted(fills, key=lambda f: f.odometer)
    total_miles = 0.0
    total_gallons = 0.0
    anchor: float | None = None
    pending_gallons = 0.0
    for fill in ordered:
        if anchor is None:
            if fill.full_tank:
                anchor = fill.odometer
            continue
        pending_gallons += fill.gallons
        if fill.full_tank:
            miles = fill.odometer - anchor
            if miles > 0 and pending_gallons > 0:
                total_miles += miles
                total_gallons += pending_gallons
            anchor = fill.odometer
            pending_gallons = 0.0
    if total_gallons <= 0:
        return None
    return total_miles / total_gallons


def average_daily_miles(readings: Sequence[tuple[datetime, float]], fallback: float) -> float:
    """Least-squares slope of odometer vs. time (miles/day) — robust to a single typo'd reading."""
    if len(readings) < 2:
        return fallback
    ordered = sorted(readings)
    t0 = ordered[0][0]
    days = np.array([(ts - t0).total_seconds() / 86400.0 for ts, _ in ordered])
    miles = np.array([odo for _, odo in ordered])
    if days[-1] - days[0] < 1.0:
        return fallback
    slope = float(np.polyfit(days, miles, 1)[0])
    return slope if slope > 0 else fallback


def depreciated_value(
    purchase_price: float, age_years: float, miles_driven: float, params: DepreciationParams
) -> float:
    """Declining-balance depreciation with a first-year step and a mileage adjustment.

    ``miles_driven`` is measured since purchase; the adjustment compares it with the baseline mileage for
    the vehicle's age (higher-than-baseline miles lower the value, capped by ``max_mileage_adjustment``).
    """
    if purchase_price <= 0:
        raise DomainError("purchase price must be positive")
    age = max(age_years, 0.0)
    if age <= 1.0:
        base = purchase_price * (1.0 - params.first_year_rate) ** age
    else:
        base = purchase_price * (1.0 - params.first_year_rate) * (1.0 - params.annual_rate) ** (age - 1.0)
    expected = params.annual_miles_baseline * age
    excess_thousands = (max(miles_driven, 0.0) - expected) / 1000.0
    adjustment = -excess_thousands * params.mileage_adjustment_per_1000
    adjustment = max(-params.max_mileage_adjustment, min(params.max_mileage_adjustment, adjustment))
    return max(base * (1.0 + adjustment), purchase_price * params.floor_fraction)


def depreciation_curve(
    purchase_price: float,
    purchase_date: date,
    purchase_odometer: float,
    annual_miles: float,
    params: DepreciationParams,
    years: int = 10,
) -> list[DepreciationPoint]:
    points: list[DepreciationPoint] = []
    for month in range(0, years * 12 + 1):
        on = add_months(purchase_date, month)
        age = month / 12.0
        driven = annual_miles * age
        points.append(
            DepreciationPoint(
                on=on,
                age_years=round(age, 4),
                expected_odometer=round(purchase_odometer + driven, 1),
                value=round(depreciated_value(purchase_price, age, driven, params), 2),
            )
        )
    return points


def annual_maintenance_cost(items: Sequence[MaintenanceItem], annual_miles: float) -> float:
    """Expected yearly maintenance spend: each item recurs at whichever interval triggers first."""
    total = 0.0
    for item in items:
        per_year = 0.0
        if item.interval_miles:
            per_year = max(per_year, annual_miles / item.interval_miles)
        if item.interval_months:
            per_year = max(per_year, 12.0 / item.interval_months)
        total += per_year * item.estimated_cost
    return total


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    code: str
    performed_on: date
    odometer: float


def maintenance_schedule(
    items: Sequence[MaintenanceItem],
    records: Sequence[ServiceRecord],
    odometer: float,
    today: date,
    avg_daily_miles: float,
    baseline_date: date,
    baseline_odometer: float,
) -> list[MaintenanceDue]:
    latest: dict[str, ServiceRecord] = {}
    for rec in records:
        seen = latest.get(rec.code)
        if seen is None or (rec.odometer, rec.performed_on) > (seen.odometer, seen.performed_on):
            latest[rec.code] = rec
    out: list[MaintenanceDue] = []
    for item in items:
        last = latest.get(item.code)
        last_odo = last.odometer if last else baseline_odometer
        last_date = last.performed_on if last else baseline_date
        next_odo = last_odo + item.interval_miles if item.interval_miles else None
        next_date = add_months(last_date, item.interval_months) if item.interval_months else None
        miles_remaining = None if next_odo is None else next_odo - odometer
        days_remaining = None if next_date is None else (next_date - today).days

        candidates: list[date] = []
        if next_date is not None:
            candidates.append(next_date)
        if miles_remaining is not None and avg_daily_miles > 0:
            days_to_miles = math.floor(max(miles_remaining, 0.0) / avg_daily_miles)
            candidates.append(today + timedelta(days=days_to_miles))
        projected = min(candidates) if candidates else None

        overdue = (miles_remaining is not None and miles_remaining <= 0) or (
            days_remaining is not None and days_remaining <= 0
        )
        soon_miles = max(DUE_SOON_MIN_MILES, 0.1 * item.interval_miles) if item.interval_miles else 0.0
        due_soon = (miles_remaining is not None and miles_remaining <= soon_miles) or (
            days_remaining is not None and days_remaining <= DUE_SOON_DAYS
        )
        status = "overdue" if overdue else "due_soon" if due_soon else "ok"
        out.append(
            MaintenanceDue(
                code=item.code,
                name=item.name,
                status=status,
                last_service_odometer=last_odo,
                last_service_date=last_date,
                next_due_odometer=next_odo,
                next_due_date=next_date,
                miles_remaining=None if miles_remaining is None else round(miles_remaining, 1),
                days_remaining=days_remaining,
                projected_due_date=projected,
                estimated_cost=item.estimated_cost,
            )
        )
    order = {"overdue": 0, "due_soon": 1, "ok": 2}
    return sorted(out, key=lambda d: (order[d.status], d.projected_due_date or date.max))


def synthetic_odometer_readings(
    start: date, start_odometer: float, annual_miles: float, end: date, seed: int = 2025
) -> list[tuple[date, float]]:
    """Deterministic weekly odometer readings used when no telemetry has been recorded."""
    if end < start:
        return [(start, start_odometer)]
    rng = np.random.default_rng(seed)
    daily = annual_miles / 365.0
    readings = [(start, start_odometer)]
    odometer = start_odometer
    cursor = start
    while cursor + timedelta(days=7) <= end:
        cursor += timedelta(days=7)
        odometer += max(0.0, rng.normal(daily * 7, daily * 7 * 0.25))
        readings.append((cursor, round(odometer, 1)))
    if cursor < end:
        odometer += daily * (end - cursor).days
        readings.append((end, round(odometer, 1)))
    return readings
