"""Vehicle asset-lifecycle, telemetry, fuel and maintenance endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Path, Query, Response, status

from quantpulse.api.deps import ContainerDep
from quantpulse.providers.eia import REGIONS
from quantpulse.schemas.common import CompositeEnvelope, Envelope
from quantpulse.schemas.vehicle import (
    EPARating,
    FuelGrade,
    FuelLogIn,
    FuelLogOut,
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
from quantpulse.services.container import Container

router = APIRouter(tags=["vehicle"])
VehicleId = Path(..., ge=1)


@router.get("/vehicle/profiles/{profile_id}", response_model=VehicleProfile)
async def profile(profile_id: str, c: Container = ContainerDep) -> VehicleProfile:
    return c.vehicle.profile(profile_id)


@router.get(
    "/vehicle/profiles/{profile_id}/epa", response_model=Envelope[EPARating], summary="Live EPA ratings"
)
async def epa(profile_id: str, c: Container = ContainerDep) -> Envelope[EPARating]:
    r = await c.vehicle.epa(c.vehicle.profile(profile_id))
    return Envelope[EPARating](data=r.value, meta=r.provenance)


@router.get("/fuel/regions", response_model=dict[str, str], summary="EIA region codes")
async def regions() -> dict[str, str]:
    return dict(REGIONS)


@router.get(
    "/fuel/prices", response_model=Envelope[FuelPriceSeries], summary="Weekly regional retail fuel prices"
)
async def fuel_prices(
    region: str | None = Query(None, description="EIA duoarea code; default QP_FUEL_REGION"),
    grade: FuelGrade | None = Query(None),
    c: Container = ContainerDep,
) -> Envelope[FuelPriceSeries]:
    r = await c.vehicle.fuel_prices(region, grade)
    return Envelope[FuelPriceSeries](data=r.value, meta=r.provenance)


@router.get("/vehicles", response_model=list[VehicleOut])
async def list_vehicles(c: Container = ContainerDep) -> list[VehicleOut]:
    return await c.vehicle.list_all()


@router.post("/vehicles", response_model=VehicleOut, status_code=status.HTTP_201_CREATED)
async def create_vehicle(data: VehicleCreate, c: Container = ContainerDep) -> VehicleOut:
    return await c.vehicle.create(data)


@router.get("/vehicles/{vehicle_id}", response_model=VehicleOut)
async def get_vehicle(vehicle_id: int = VehicleId, c: Container = ContainerDep) -> VehicleOut:
    return await c.vehicle.get(vehicle_id)


@router.delete("/vehicles/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vehicle(vehicle_id: int = VehicleId, c: Container = ContainerDep) -> Response:
    await c.vehicle.delete(vehicle_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/vehicles/{vehicle_id}/dashboard", response_model=CompositeEnvelope[VehicleDashboard])
async def dashboard(
    vehicle_id: int = VehicleId, c: Container = ContainerDep
) -> CompositeEnvelope[VehicleDashboard]:
    return await c.vehicle.dashboard(vehicle_id)


@router.get("/vehicles/{vehicle_id}/telemetry", response_model=list[TelemetryOut])
async def list_telemetry(vehicle_id: int = VehicleId, c: Container = ContainerDep) -> list[TelemetryOut]:
    return await c.vehicle.telemetry(vehicle_id)


@router.post(
    "/vehicles/{vehicle_id}/telemetry", response_model=TelemetryOut, status_code=status.HTTP_201_CREATED
)
async def add_telemetry(
    data: TelemetryIn, vehicle_id: int = VehicleId, c: Container = ContainerDep
) -> TelemetryOut:
    return await c.vehicle.add_telemetry(vehicle_id, data)


@router.get("/vehicles/{vehicle_id}/fuel-logs", response_model=list[FuelLogOut])
async def list_fuel_logs(vehicle_id: int = VehicleId, c: Container = ContainerDep) -> list[FuelLogOut]:
    return await c.vehicle.fuel_logs(vehicle_id)


@router.post(
    "/vehicles/{vehicle_id}/fuel-logs", response_model=FuelLogOut, status_code=status.HTTP_201_CREATED
)
async def add_fuel_log(
    data: FuelLogIn, vehicle_id: int = VehicleId, c: Container = ContainerDep
) -> FuelLogOut:
    return await c.vehicle.add_fuel_log(vehicle_id, data)


@router.get("/vehicles/{vehicle_id}/maintenance", response_model=list[MaintenanceRecordOut])
async def list_maintenance(
    vehicle_id: int = VehicleId, c: Container = ContainerDep
) -> list[MaintenanceRecordOut]:
    return await c.vehicle.maintenance_records(vehicle_id)


@router.post(
    "/vehicles/{vehicle_id}/maintenance",
    response_model=MaintenanceRecordOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_maintenance(
    data: MaintenanceRecordIn, vehicle_id: int = VehicleId, c: Container = ContainerDep
) -> MaintenanceRecordOut:
    return await c.vehicle.add_maintenance(vehicle_id, data)


@router.delete("/vehicles/{vehicle_id}/{kind}/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_record(
    kind: Literal["telemetry", "fuel-logs", "maintenance"],
    vehicle_id: int = VehicleId,
    record_id: int = Path(..., ge=1),
    c: Container = ContainerDep,
) -> Response:
    await c.vehicle.delete_record(vehicle_id, kind, record_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
