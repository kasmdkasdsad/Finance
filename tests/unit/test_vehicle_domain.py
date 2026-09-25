from datetime import UTC, date, datetime, timedelta
from itertools import pairwise

import pytest

from quantpulse.domain.vehicle import (
    FuelFill,
    ServiceRecord,
    add_months,
    annual_maintenance_cost,
    average_daily_miles,
    blended_mpg,
    calibrated_mpg,
    depreciated_value,
    depreciation_curve,
    load_profile,
    maintenance_schedule,
    realized_mpg,
    synthetic_odometer_readings,
)
from quantpulse.schemas.vehicle import MaintenanceItem


@pytest.fixture(scope="module")
def profile():
    return load_profile("hyundai-elantra-2025-limited")


def test_profile_matches_verified_reference_data(profile):
    assert (profile.year, profile.make, profile.model, profile.trim) == (
        2025,
        "Hyundai",
        "Elantra",
        "Limited",
    )
    assert (profile.epa.city_mpg, profile.epa.highway_mpg, profile.epa.combined_mpg) == (30, 39, 34)
    assert profile.epa.vehicle_id == 48019
    assert profile.pricing.total_msrp == 26525 + 1150
    assert profile.tank_gallons == 12.4


def test_blended_and_calibrated_mpg(profile):
    epa = profile.epa
    # Harmonic 55/45 blend of the *rounded* label values is within a rounding error of the label...
    assert abs(blended_mpg(epa.city_mpg, epa.highway_mpg, 0.55) - epa.combined_mpg) < 1.0
    assert blended_mpg(30, 39, 1.0) == pytest.approx(30)
    # ...and the calibrated blend reproduces the official combined figure exactly at the EPA mix.
    assert calibrated_mpg(epa.city_mpg, epa.highway_mpg, epa.combined_mpg, 0.55) == pytest.approx(34.0)
    city_heavy = calibrated_mpg(epa.city_mpg, epa.highway_mpg, epa.combined_mpg, 0.9)
    hwy_heavy = calibrated_mpg(epa.city_mpg, epa.highway_mpg, epa.combined_mpg, 0.1)
    assert city_heavy < 34.0 < hwy_heavy


def test_add_months_clamps_month_end():
    assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert add_months(date(2025, 11, 15), 3) == date(2026, 2, 15)


def test_realized_mpg_full_tank_method():
    fills = [
        FuelFill(1000, 10.0, True, 3.0),
        FuelFill(1200, 5.0, False, 3.0),  # partial fill counts toward the next full interval
        FuelFill(1340, 6.0, True, 3.0),
        FuelFill(1680, 10.0, True, 3.0),
    ]
    assert realized_mpg(fills) == pytest.approx((340 + 340) / (11 + 10))
    assert realized_mpg(fills[:1]) is None


def test_average_daily_miles_regression():
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    readings = [(t0 + timedelta(days=d), 100 + 30 * d) for d in range(0, 50, 7)]
    assert average_daily_miles(readings, fallback=1.0) == pytest.approx(30.0)
    assert average_daily_miles(readings[:1], fallback=12.5) == 12.5


def test_depreciation_behaviour(profile):
    p = profile.depreciation
    price = 27675.0
    assert depreciated_value(price, 0, 0, p) == pytest.approx(price)
    assert depreciated_value(price, 1, p.annual_miles_baseline, p) == pytest.approx(
        price * (1 - p.first_year_rate)
    )
    assert depreciated_value(price, 3, 36000, p) == pytest.approx(
        price * (1 - p.first_year_rate) * (1 - p.annual_rate) ** 2
    )
    high_miles = depreciated_value(price, 3, 60000, p)
    low_miles = depreciated_value(price, 3, 20000, p)
    assert low_miles > depreciated_value(price, 3, 36000, p) > high_miles
    assert depreciated_value(price, 40, 500000, p) == pytest.approx(price * p.floor_fraction)
    curve = depreciation_curve(price, date(2025, 6, 1), 10, 12000, p, years=5)
    assert len(curve) == 61
    values = [pt.value for pt in curve]
    assert all(b <= a for a, b in pairwise(values))


def test_maintenance_schedule_statuses():
    items = [
        MaintenanceItem(
            code="engine_oil", name="Oil", interval_miles=7500, interval_months=12, estimated_cost=80
        ),
        MaintenanceItem(code="wipers", name="Wipers", interval_months=12, estimated_cost=40),
        MaintenanceItem(code="plugs", name="Plugs", interval_miles=60000, estimated_cost=250),
    ]
    today = date(2026, 9, 25)
    records = [
        ServiceRecord("engine_oil", date(2026, 6, 1), 7000),
        ServiceRecord("engine_oil", date(2025, 12, 1), 3000),
    ]
    due = {d.code: d for d in maintenance_schedule(items, records, 14100, today, 40.0, date(2025, 6, 1), 0)}
    assert due["engine_oil"].last_service_odometer == 7000
    assert due["engine_oil"].next_due_odometer == 14500
    assert due["engine_oil"].status == "due_soon"  # 400 miles left
    assert due["engine_oil"].projected_due_date == today + timedelta(days=10)
    assert due["wipers"].status == "overdue"  # baseline 2025-06-01 + 12 months
    assert due["plugs"].status == "ok"
    assert due["plugs"].next_due_date is None


def test_annual_maintenance_cost_uses_first_trigger():
    items = [
        MaintenanceItem(
            code="engine_oil", name="Oil", interval_miles=7500, interval_months=12, estimated_cost=100
        )
    ]
    assert annual_maintenance_cost(items, 15000) == pytest.approx(200)  # mileage triggers first
    assert annual_maintenance_cost(items, 3000) == pytest.approx(100)  # time triggers first


def test_synthetic_readings_deterministic_and_monotonic():
    a = synthetic_odometer_readings(date(2025, 6, 1), 10, 12000, date(2026, 9, 25))
    assert a == synthetic_odometer_readings(date(2025, 6, 1), 10, 12000, date(2026, 9, 25))
    odos = [o for _, o in a]
    assert all(b >= x for x, b in pairwise(odos))
    assert a[-1][0] == date(2026, 9, 25)
