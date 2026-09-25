"""Cross-company fundamentals from SEC XBRL frames, stored in the warehouse.

One frame request returns a single concept for *every* SEC filer in one period, so the whole market's
value/quality inputs take about 80 requests instead of one multi-megabyte download per company.
Frames are cached in ``fundamental_facts``; closed periods are re-checked every 90 days (restatements
are rare), recent ones every ``QP_TTL_FUNDAMENTALS_FRAMES``. A frame that fails to download is simply
missing (its factors become NaN): real and synthetic fundamentals are never mixed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import ProviderError
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.domain.fundamental_factors import CompanyFacts, Fact
from quantpulse.providers import synthetic
from quantpulse.providers.sec_edgar import SecEdgar
from quantpulse.schemas.common import DataStatus

logger = logging.getLogger(__name__)

CLOSED_PERIOD_TTL = timedelta(days=90)
CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class FrameSpec:
    field: str  # CompanyFacts attribute
    taxonomy: str
    tag: str
    unit: str
    kind: str  # "annual" (CY2023), "q4" (CY2023Q4I) or "quarterly" (CY2023Q1I..Q4I)

    def periods(self, years: Sequence[int]) -> list[str]:
        if self.kind == "annual":
            return [f"CY{y}" for y in years]
        if self.kind == "q4":
            return [f"CY{y}Q4I" for y in years]
        return [f"CY{y}Q{q}I" for y in years for q in (1, 2, 3, 4)]


# Earlier specs for the same field take precedence (fallback tags only fill gaps).
SPECS: tuple[FrameSpec, ...] = (
    FrameSpec("net_income", "us-gaap", "NetIncomeLoss", "USD", "annual"),
    FrameSpec("net_income", "us-gaap", "ProfitLoss", "USD", "annual"),
    FrameSpec(
        "operating_cash_flow", "us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD", "annual"
    ),
    FrameSpec("capex", "us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD", "annual"),
    FrameSpec("capex", "us-gaap", "PaymentsToAcquireProductiveAssets", "USD", "annual"),
    FrameSpec("gross_profit", "us-gaap", "GrossProfit", "USD", "annual"),
    FrameSpec("assets", "us-gaap", "Assets", "USD", "q4"),
    FrameSpec("equity", "us-gaap", "StockholdersEquity", "USD", "q4"),
    FrameSpec(
        "equity",
        "us-gaap",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "USD",
        "q4",
    ),
    FrameSpec("public_float", "dei", "EntityPublicFloat", "USD", "quarterly"),
)


@dataclass
class FactsCoverage:
    status: DataStatus
    frames_requested: int = 0
    frames_available: int = 0
    frames_missing: list[str] = field(default_factory=list)
    companies_with_facts: int = 0


def _years(first: date, last: date) -> list[int]:
    # One extra year back: asset growth and the point-in-time lag need the prior report.
    return list(range(first.year - 2, last.year + 1))


class FactsService:
    def __init__(self, settings: Settings, db: Database, clock: Clock, sec: SecEdgar) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._sec = sec
        self._locks: dict[str, asyncio.Lock] = {}

    def _marker(self, spec: FrameSpec, period: str) -> str:
        return f"frame:{spec.tag}:{period}"

    def _ttl(self, period: str) -> timedelta:
        year = int(period[2:6])
        closed = year < self._clock.now().year - 1
        return CLOSED_PERIOD_TTL if closed else timedelta(seconds=self._settings.ttl_fundamentals_frames)

    async def _ensure(self, spec: FrameSpec, period: str) -> bool:
        """Make sure the frame is in the warehouse; ``True`` when rows are available."""
        key = self._marker(spec, period)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            async with self._db.session() as s:
                marker = await repo.get_blob(s, key)
            now = self._clock.now()
            if marker is not None and now - marker[1] < self._ttl(period):
                return bool(marker[0].get("rows"))
            try:
                facts = await self._sec.frame(spec.taxonomy, spec.tag, spec.unit, period)
            except ProviderError:
                # Not published yet (future periods), or a transient failure: keep any stored rows.
                return bool(marker and marker[0].get("rows"))
            except Exception:  # defensive: one malformed frame must not sink a whole model run
                logger.exception("SEC frame %s %s failed unexpectedly", spec.tag, period)
                return bool(marker and marker[0].get("rows"))
            async with self._db.session() as s:
                n = await repo.save_frame(s, spec.tag, period, facts)
                await repo.put_blob(s, key, {"rows": n}, self._sec.name)
            return n > 0

    async def company_facts(
        self,
        symbol_cik: Mapping[str, int],
        first: date,
        last: date,
        *,
        progress: Callable[[float], None] | None = None,
        simulated: bool = False,
    ) -> tuple[dict[str, CompanyFacts], FactsCoverage]:
        """Facts for the requested companies covering ``[first, last]`` (simulated ones when live data is
        off or ``simulated`` is asked for, e.g. for a synthetic price panel)."""
        if simulated or not self._settings.enable_live_data:
            now = self._clock.now()

            def simulate() -> dict[str, CompanyFacts]:
                return {s: synthetic.synthetic_company_facts(s, now) for s in symbol_cik}

            fake = await asyncio.to_thread(simulate)
            return fake, FactsCoverage(status=DataStatus.SYNTHETIC, companies_with_facts=len(fake))
        years = _years(first, last)
        jobs = [(spec, period) for spec in SPECS for period in spec.periods(years)]
        sem = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def one(spec: FrameSpec, period: str) -> tuple[FrameSpec, str, bool]:
            nonlocal done
            async with sem:
                ok = await self._ensure(spec, period)
            done += 1
            if progress:
                progress(done / len(jobs))
            return spec, period, ok

        results = await asyncio.gather(*(one(s, p) for s, p in jobs))
        today = self._clock.now().date()
        # Only periods that should be published by now count as missing (10-K/10-Q deadlines + slack).
        missing = [
            f"{s.tag} {p}" for s, p, ok in results if not ok and _period_end(p) + timedelta(days=120) < today
        ]
        by_cik = {cik: sym for sym, cik in symbol_cik.items()}
        async with self._db.session() as s:
            rows = await repo.facts_for(s, by_cik, {spec.tag for spec in SPECS})
        priority = {spec.tag: i for i, spec in enumerate(SPECS)}
        field_of = {spec.tag: spec.field for spec in SPECS}
        chosen: dict[tuple[str, str, date], tuple[int, Fact]] = {}
        for r in rows:
            symbol = by_cik.get(r.cik)
            if symbol is None:
                continue
            k = (symbol, field_of[r.tag], r.period_end)
            rank = priority[r.tag]
            if k not in chosen or rank < chosen[k][0]:
                chosen[k] = (rank, Fact(end=r.period_end, value=r.value, start=r.period_start))
        out: dict[str, CompanyFacts] = {}
        for (symbol, fld, _), (_, fact) in chosen.items():
            getattr(out.setdefault(symbol, CompanyFacts()), fld).append(fact)
        coverage = FactsCoverage(
            status=DataStatus.STALE if missing else DataStatus.LIVE,
            frames_requested=len(jobs),
            frames_available=sum(1 for _, _, ok in results if ok),
            frames_missing=missing,
            companies_with_facts=len(out),
        )
        return out, coverage


def _period_end(period: str) -> date:
    year = int(period[2:6])
    if len(period) == 6:
        return date(year, 12, 31)
    quarter = int(period[7])
    return date(year, 3 * quarter, 30 if quarter in (2, 3) else 31)
