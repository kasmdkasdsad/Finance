"""U.S. Energy Information Administration (EIA) API v2: weekly retail fuel prices by region (API key)."""

from __future__ import annotations

from datetime import date

from pydantic import field_validator

from quantpulse.core.errors import DomainError, ProviderNotConfigured
from quantpulse.core.http import HttpClient
from quantpulse.providers.base import WireModel, parse_wire, require
from quantpulse.schemas.vehicle import FuelPrice, FuelPriceSeries

NAME = "eia"
URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"

REGIONS: dict[str, str] = {
    "NUS": "U.S. average",
    "R10": "East Coast (PADD 1)",
    "R1X": "New England (PADD 1A)",
    "R1Y": "Central Atlantic (PADD 1B)",
    "R1Z": "Lower Atlantic (PADD 1C)",
    "R20": "Midwest (PADD 2)",
    "R30": "Gulf Coast (PADD 3)",
    "R40": "Rocky Mountain (PADD 4)",
    "R50": "West Coast (PADD 5)",
    "R5XCA": "West Coast less California",
    "SCA": "California",
    "SCO": "Colorado",
    "SFL": "Florida",
    "SMA": "Massachusetts",
    "SMN": "Minnesota",
    "SNY": "New York",
    "SOH": "Ohio",
    "STX": "Texas",
    "SWA": "Washington",
    "Y05LA": "Los Angeles",
    "Y05SF": "San Francisco",
    "Y35NY": "New York City",
    "Y44HO": "Houston",
    "Y48SE": "Seattle",
    "YBOS": "Boston",
    "YCLE": "Cleveland",
    "YDEN": "Denver",
    "YMIA": "Miami",
    "YORD": "Chicago",
}
PRODUCTS: dict[str, str] = {"regular": "EPMR", "midgrade": "EPMM", "premium": "EPMP", "diesel": "EPD2D"}


def validate_region(region: str) -> str:
    code = region.strip().upper()
    if code not in REGIONS:
        raise DomainError(f"unknown EIA region '{region}'. Valid: {', '.join(sorted(REGIONS))}")
    return code


class _Row(WireModel):
    period: date
    duoarea: str
    product: str
    series: str | None = None
    value: float | None = None
    units: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        return None if v in ("", None) else v


class _Response(WireModel):
    data: list[_Row] = []


class _Envelope(WireModel):
    response: _Response


class EIA:
    name = NAME

    def __init__(self, http: HttpClient, api_key: str | None) -> None:
        self._http = http
        self._key = api_key

    def configured(self) -> bool:
        return bool(self._key)

    async def fuel_prices(self, region: str, grade: str, weeks: int = 104) -> FuelPriceSeries:
        if not self._key:
            raise ProviderNotConfigured(NAME, "QP_EIA_API_KEY not set")
        code = validate_region(region)
        product = PRODUCTS[grade]
        payload = await self._http.get_json(
            NAME,
            URL,
            params={
                "api_key": self._key,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[duoarea][]": code,
                "facets[product][]": product,
                "sort[0][column]": "period",
                "sort[0][direction]": "desc",
                "offset": 0,
                "length": weeks,
            },
        )
        env = parse_wire(NAME, _Envelope, payload)
        rows = [
            r
            for r in env.response.data
            if r.value is not None
            and r.value > 0
            and r.duoarea == code
            and r.product == product
            and (r.units is None or "GAL" in r.units.upper())
        ]
        require(bool(rows), NAME, f"no {grade} prices for {code}")
        history = sorted(
            (
                FuelPrice(
                    region=code,
                    region_name=REGIONS[code],
                    grade=grade,
                    price=r.value,
                    period=r.period,
                    series_id=r.series,
                )
                for r in rows
            ),
            key=lambda p: p.period,
        )
        return FuelPriceSeries(
            region=code, region_name=REGIONS[code], grade=grade, latest=history[-1], history=history
        )
