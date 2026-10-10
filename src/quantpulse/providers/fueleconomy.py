"""fueleconomy.gov web services (keyless): official EPA ratings for a vehicle id."""

from __future__ import annotations

from quantpulse.core.http import HttpClient
from quantpulse.providers.base import WireModel, parse_wire, require
from quantpulse.schemas.vehicle import EPARating

NAME = "fueleconomy_gov"
URL = "https://www.fueleconomy.gov/ws/rest/vehicle/{vehicle_id}"


class _Vehicle(WireModel):
    id: int
    year: int
    make: str
    model: str
    trany: str | None = None
    displ: float | None = None
    city08: float
    highway08: float
    comb08: float
    eng_dscr: str | None = None


class FuelEconomyGov:
    name = NAME

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    def configured(self) -> bool:
        return True

    async def epa_rating(self, vehicle_id: int) -> EPARating:
        payload = await self._http.get_json(
            NAME, URL.format(vehicle_id=vehicle_id), headers={"Accept": "application/json"}
        )
        v = parse_wire(NAME, _Vehicle, payload)
        require(v.city08 > 0 and v.highway08 > 0 and v.comb08 > 0, NAME, "vehicle has no MPG ratings")
        detail = ", ".join(x for x in (f"{v.displ} L" if v.displ else None, v.trany, v.eng_dscr) if x)
        return EPARating(
            vehicle_id=v.id,
            city_mpg=v.city08,
            highway_mpg=v.highway08,
            combined_mpg=v.comb08,
            source=f"fueleconomy.gov vehicle {v.id}: {v.year} {v.make} {v.model} ({detail})",
        )
