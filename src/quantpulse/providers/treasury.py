"""U.S. Department of the Treasury daily par yield curve (keyless CSV feed).

The feed is authoritative but slow (often 15-20 s), so requests use a long timeout and the background
poller keeps the curve warm in the cache.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime

from quantpulse.core.errors import ProviderNoData, ProviderParseError
from quantpulse.core.http import HttpClient
from quantpulse.quant.rates import parse_tenor_label
from quantpulse.schemas.options import YieldCurve, YieldPoint

NAME = "treasury"
URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/{ym}"
TIMEOUT_SECONDS = 45.0


def parse_curve_csv(text: str) -> YieldCurve:
    """Parse the Treasury CSV and return the most recent curve (rates converted to decimals)."""
    reader = csv.reader(io.StringIO(text.lstrip("﻿")))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ProviderNoData(NAME, "empty CSV") from exc
    if not header or header[0].strip().lower() != "date":
        raise ProviderParseError(NAME, f"unexpected CSV header: {header[:3]}")
    tenors: list[tuple[int, str, float]] = []
    for idx, label in enumerate(header[1:], start=1):
        try:
            tenors.append((idx, label.strip(), parse_tenor_label(label)))
        except ValueError:
            continue  # ignore unknown columns rather than failing the whole curve
    if not tenors:
        raise ProviderParseError(NAME, "no tenor columns found")

    best: tuple[date, list[YieldPoint]] | None = None
    for row in reader:
        if not row or not row[0].strip():
            continue
        try:
            day = datetime.strptime(row[0].strip(), "%m/%d/%Y").date()
        except ValueError as exc:
            raise ProviderParseError(NAME, f"bad date {row[0]!r}") from exc
        points: list[YieldPoint] = []
        for idx, label, years in tenors:
            if idx >= len(row):
                continue
            cell = row[idx].strip()
            if not cell or cell.upper() in {"N/A", "NA"}:
                continue
            try:
                value = float(cell)
            except ValueError:
                continue
            points.append(YieldPoint(tenor=label, years=years, rate=value / 100.0))
        if points and (best is None or day > best[0]):
            best = (day, points)
    if best is None:
        raise ProviderNoData(NAME, "no observations in CSV")
    return YieldCurve(as_of=best[0], points=best[1])


class Treasury:
    name = NAME

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    def configured(self) -> bool:
        return True

    async def curve(self, today: date | None = None) -> YieldCurve:
        today = today or datetime.now(UTC).date()
        months = [(today.year, today.month)]
        prev = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        months.append(prev)
        last_error: Exception | None = None
        for year, month in months:  # early in a month the current month may have no rows yet
            ym = f"{year}{month:02d}"
            try:
                text = await self._http.get_text(
                    NAME,
                    URL.format(ym=ym),
                    params={
                        "type": "daily_treasury_yield_curve",
                        "field_tdr_date_value_month": ym,
                        "page": "",
                        "_format": "csv",
                    },
                    headers={"Accept": "text/csv,*/*"},
                    timeout=TIMEOUT_SECONDS,
                )
                return parse_curve_csv(text)
            except ProviderNoData as exc:
                last_error = exc
                continue
        raise ProviderNoData(NAME, f"no curve data for {months}: {last_error}")
