"""SEC EDGAR provider (keyless, requires a descriptive User-Agent): XBRL company facts and filings.

XBRL "companyfacts" pitfalls handled here:

* each fact's ``fy`` is the fiscal year of the *filing*, so a FY2025 10-K also re-reports FY2023/FY2024
  values tagged ``fy=2025`` — periods are therefore keyed by their ``end`` date, and the fiscal-year label
  comes from the filing whose primary period ends on that date;
* annual (``fp=FY``) 10-K facts also include quarterly durations — duration facts must span ~1 year;
* restatements appear in later filings — the most recently *filed* value wins;
* companies switch tags over time (e.g. ``SalesRevenueNet`` → ``RevenueFromContract…``) — concepts are
  resolved per period through an ordered list of candidates.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from quantpulse.core.errors import ProviderNoData, ProviderParseError
from quantpulse.core.http import HttpClient
from quantpulse.domain.sectors import FF12_NAMES, ff12
from quantpulse.providers.base import WireModel, parse_wire
from quantpulse.schemas.fundamentals import CompanyFundamentals, Filing, FinancialStatement
from quantpulse.schemas.reference import CompanyEvents, CompanyProfile, FrameFact

NAME = "sec_edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
FRAMES_URL = "https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
EARNINGS_ITEM = "2.02"  # Form 8-K item 2.02: Results of Operations and Financial Condition
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accn_nodash}/{doc}"
ANNUAL_FORMS = frozenset({"10-K", "10-K/A", "10-KT", "20-F", "20-F/A", "40-F", "40-F/A"})
FILING_FORMS = frozenset({"10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "10-K/A", "10-Q/A", "DEF 14A", "S-1"})
TICKER_TTL_SECONDS = 86400.0
MAX_YEARS = 10

CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "revenue": (
        "USD",
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueGoodsNet",
            "RevenuesNetOfInterestExpense",
        ],
    ),
    "gross_profit": ("USD", ["GrossProfit"]),
    "operating_income": ("USD", ["OperatingIncomeLoss"]),
    "net_income": ("USD", ["NetIncomeLoss", "ProfitLoss"]),
    "pretax_income": (
        "USD",
        [
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ],
    ),
    "income_tax": ("USD", ["IncomeTaxExpenseBenefit"]),
    "interest_expense": (
        "USD",
        ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"],
    ),
    "depreciation_amortization": (
        "USD",
        [
            "DepreciationDepletionAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            "DepreciationAndAmortization",
            "Depreciation",
        ],
    ),
    "total_assets": ("USD", ["Assets"]),
    "total_liabilities": ("USD", ["Liabilities"]),
    "stockholders_equity": (
        "USD",
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    ),
    "cash": (
        "USD",
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "Cash",
        ],
    ),
    "current_assets": ("USD", ["AssetsCurrent"]),
    "current_liabilities": ("USD", ["LiabilitiesCurrent"]),
    "operating_cash_flow": (
        "USD",
        [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
    ),
    "capital_expenditure": (
        "USD",
        ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    ),
    "diluted_eps": ("USD/shares", ["EarningsPerShareDiluted"]),
    "diluted_shares": ("shares", ["WeightedAverageNumberOfDilutedSharesOutstanding"]),
}
DEBT_CONCEPTS: dict[str, list[str]] = {
    "ltd_total": ["LongTermDebt"],
    "ltd_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "ltd_current": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "commercial_paper": ["CommercialPaper"],
    "short_term_borrowings": ["ShortTermBorrowings"],
}


class _Fact(WireModel):
    end: date
    val: float
    accn: str
    form: str | None = None
    fp: str | None = None
    fy: int | None = None
    filed: date
    start: date | None = None


class _Concept(WireModel):
    units: dict[str, list[_Fact]] = {}


class _CompanyFacts(WireModel):
    cik: int
    entityName: str | None = None
    facts: dict[str, dict[str, _Concept]] = {}


class _Recent(WireModel):
    accessionNumber: list[str] = []
    filingDate: list[date] = []
    reportDate: list[str] = []
    form: list[str] = []
    primaryDocument: list[str] = []


class _Filings(WireModel):
    recent: _Recent


class _Submissions(WireModel):
    name: str | None = None
    filings: _Filings


class _EventArrays(WireModel):
    """The parallel arrays of a submissions page (the ``recent`` block, or an older ``files`` page)."""

    filingDate: list[date] = []
    acceptanceDateTime: list[str] = []
    form: list[str] = []
    items: list[str] = []


class _SubmissionFile(WireModel):
    name: str
    filingFrom: date
    filingTo: date


class _EventFilings(WireModel):
    recent: _EventArrays
    files: list[_SubmissionFile] = []


class _EventSubmissions(WireModel):
    name: str | None = None
    sic: str | None = None
    sicDescription: str | None = None
    filings: _EventFilings


class _FrameRow(WireModel):
    accn: str
    cik: int
    start: date | None = None
    end: date
    val: float


class _Frame(WireModel):
    data: list[_FrameRow]


def earnings_times(arrays: _EventArrays, since: date) -> list[datetime]:
    """Acceptance times (UTC) of 8-K filings carrying item 2.02, on or after ``since``."""
    out: set[datetime] = set()
    for i, form in enumerate(arrays.form):
        if form != "8-K" or i >= len(arrays.filingDate):
            continue
        items = arrays.items[i] if i < len(arrays.items) else ""
        if EARNINGS_ITEM not in {x.strip() for x in items.split(",")}:
            continue
        filed = arrays.filingDate[i]
        if filed < since:
            continue
        raw = arrays.acceptanceDateTime[i] if i < len(arrays.acceptanceDateTime) else ""
        try:
            accepted = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if accepted.tzinfo is None:
                accepted = accepted.replace(tzinfo=UTC)
        except ValueError:
            # No timestamp: assume an after-close release (the most common timing).
            accepted = datetime(filed.year, filed.month, filed.day, 20, 30, tzinfo=UTC)
        out.add(accepted.astimezone(UTC))
    return sorted(out)


@dataclass(frozen=True, slots=True)
class AnnualValue:
    value: float
    accn: str
    filed: date
    form: str
    fy: int | None


def annual_values(concept: _Concept | None, unit: str) -> dict[date, AnnualValue]:
    """Annual observations keyed by period end; latest filing wins; durations must span ~1 year."""
    if concept is None:
        return {}
    out: dict[date, AnnualValue] = {}
    for fact in concept.units.get(unit, []):
        if not _is_annual(fact):
            continue
        current = out.get(fact.end)
        if current is None or fact.filed > current.filed:
            out[fact.end] = AnnualValue(fact.val, fact.accn, fact.filed, fact.form or "10-K", fact.fy)
    return out


def _first_available(gaap: dict[str, _Concept], names: list[str], unit: str) -> dict[date, AnnualValue]:
    """Merge candidate concepts per period: earlier names take precedence for each period end."""
    merged: dict[date, AnnualValue] = {}
    for name in reversed(names):
        merged.update(annual_values(gaap.get(name), unit))
    return merged


def _is_annual(fact: _Fact) -> bool:
    if fact.form not in ANNUAL_FORMS or fact.fp != "FY":
        return False
    return fact.start is None or 330 <= (fact.end - fact.start).days <= 380


def fiscal_year_ends(concepts: list[_Concept | None]) -> dict[date, _Fact]:
    """Map each annual filing's *primary* period end (the latest end it reports) to that filing's fact.

    Must run on raw facts: after "latest filing wins" de-duplication an older 10-K no longer owns its own
    period (the comparative column of the next 10-K was filed later), which would shift every label.
    The original (earliest-filed) report for a period defines its fiscal-year label.
    """
    primary: dict[str, _Fact] = {}
    for concept in concepts:
        if concept is None:
            continue
        for facts in concept.units.values():
            for fact in facts:
                if not _is_annual(fact):
                    continue
                current = primary.get(fact.accn)
                if current is None or fact.end > current.end:
                    primary[fact.accn] = fact
    ends: dict[date, _Fact] = {}
    for fact in primary.values():
        existing = ends.get(fact.end)
        if existing is None or fact.filed < existing.filed:
            ends[fact.end] = fact
    # A fiscal-year label maps to one period (keep the latest end, e.g. after a fiscal-year change).
    by_fy: dict[int, date] = {}
    for end, fact in ends.items():
        fy = fact.fy if fact.fy is not None else end.year
        if fy not in by_fy or end > by_fy[fy]:
            by_fy[fy] = end
    return {end: ends[end] for end in by_fy.values()}


def latest_shares(facts: dict[str, dict[str, _Concept]]) -> tuple[float, date] | None:
    """Most recent share count: cover-page dei value (summed across classes) or balance-sheet value."""
    candidates: list[tuple[date, int, float]] = []  # (end, priority, value) — dei preferred on ties
    dei = facts.get("dei", {}).get("EntityCommonStockSharesOutstanding")
    if dei is not None:
        grouped: dict[tuple[str, date], float] = defaultdict(float)
        for f in dei.units.get("shares", []):
            grouped[(f.accn, f.end)] += f.val
        if grouped:
            (_accn, end), total = max(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            candidates.append((end, 1, total))
    cso = facts.get("us-gaap", {}).get("CommonStockSharesOutstanding")
    if cso is not None:
        rows = cso.units.get("shares", [])
        if rows:
            latest = max(rows, key=lambda f: (f.end, f.filed))
            candidates.append((latest.end, 0, latest.val))
    candidates = [c for c in candidates if c[2] > 0]
    if not candidates:
        return None
    end, _, value = max(candidates)
    return value, end


def build_statements(payload: Any) -> tuple[str | None, list[FinancialStatement], tuple[float, date] | None]:
    data = parse_wire(NAME, _CompanyFacts, payload)
    gaap = data.facts.get("us-gaap", {})
    if not gaap:
        raise ProviderNoData(NAME, "no us-gaap facts (IFRS/foreign filers are not supported)")
    series: dict[str, dict[date, AnnualValue]] = {
        field: _first_available(gaap, names, unit) for field, (unit, names) in CONCEPTS.items()
    }
    debt = {k: _first_available(gaap, names, "USD") for k, names in DEBT_CONCEPTS.items()}
    anchor_names = [
        *CONCEPTS["revenue"][1],
        *CONCEPTS["net_income"][1],
        *CONCEPTS["total_assets"][1],
        *CONCEPTS["operating_cash_flow"][1],
    ]
    ends = fiscal_year_ends([gaap.get(name) for name in anchor_names])
    if not ends:
        raise ProviderNoData(NAME, "no annual (10-K) facts found")

    statements: list[FinancialStatement] = []
    for end in sorted(ends)[-MAX_YEARS:]:
        anchor = ends[end]
        fy = anchor.fy if anchor.fy is not None else end.year
        values = {field: (s[end].value if end in s else None) for field, s in series.items()}
        if values["capital_expenditure"] is not None:
            values["capital_expenditure"] = abs(values["capital_expenditure"])
        statements.append(
            FinancialStatement(
                fiscal_year=fy,
                period_end=end,
                form=anchor.form or "10-K",
                filed=anchor.filed,
                accession=anchor.accn,
                total_debt=_total_debt(debt, end),
                **values,
            )
        )
    if not statements:
        raise ProviderParseError(NAME, "could not assemble any annual statement")
    return data.entityName, statements, latest_shares(data.facts)


def _total_debt(debt: dict[str, dict[date, AnnualValue]], end: date) -> float | None:
    def get(key: str) -> float | None:
        av = debt[key].get(end)
        return av.value if av else None

    ltd = get("ltd_total")
    if ltd is None:
        noncurrent, current = get("ltd_noncurrent"), get("ltd_current")
        ltd = None if noncurrent is None and current is None else (noncurrent or 0.0) + (current or 0.0)
    short = (get("commercial_paper") or 0.0) + (get("short_term_borrowings") or 0.0)
    if ltd is None and short == 0.0:
        return None
    return (ltd or 0.0) + short


def build_filings(payload: Any, cik: str, limit: int = 20) -> list[Filing]:
    sub = parse_wire(NAME, _Submissions, payload)
    recent = sub.filings.recent
    out: list[Filing] = []
    for i, accn in enumerate(recent.accessionNumber):
        form = recent.form[i] if i < len(recent.form) else ""
        if form not in FILING_FORMS:
            continue
        report = recent.reportDate[i] if i < len(recent.reportDate) else ""
        doc = recent.primaryDocument[i] if i < len(recent.primaryDocument) else None
        url = (
            ARCHIVE_URL.format(cik_int=int(cik), accn_nodash=accn.replace("-", ""), doc=doc) if doc else None
        )
        out.append(
            Filing(
                form=form,
                filing_date=recent.filingDate[i],
                report_date=date.fromisoformat(report) if report else None,
                accession=accn,
                primary_document=doc,
                url=url,
            )
        )
        if len(out) >= limit:
            break
    return out


class SecEdgar:
    name = NAME

    def __init__(self, http: HttpClient, user_agent: str) -> None:
        self._http = http
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._tickers: dict[str, tuple[str, str]] = {}
        self._tickers_loaded = 0.0

    def configured(self) -> bool:
        return True

    async def resolve_cik(self, symbol: str) -> tuple[str, str]:
        if not self._tickers or time.monotonic() - self._tickers_loaded > TICKER_TTL_SECONDS:
            payload = await self._http.get_json(NAME, TICKERS_URL, headers=self._headers)
            if not isinstance(payload, dict):
                raise ProviderParseError(NAME, "unexpected company_tickers.json shape")
            mapping: dict[str, tuple[str, str]] = {}
            for row in payload.values():
                try:
                    mapping[str(row["ticker"]).upper()] = (f"{int(row['cik_str']):010d}", str(row["title"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if not mapping:
                raise ProviderParseError(NAME, "empty ticker map")
            self._tickers = mapping
            self._tickers_loaded = time.monotonic()
        key = symbol.upper().replace(".", "-")
        if key not in self._tickers:
            raise ProviderNoData(NAME, f"{symbol} is not an SEC-registered ticker")
        return self._tickers[key]

    async def company_events(self, symbol: str, since: date) -> CompanyEvents:
        """Profile (SIC → Fama-French sector) and earnings-release times since ``since``.

        Large filers push older filings into extra ``files`` pages; those are read until ``since``."""
        cik, title = await self.resolve_cik(symbol)
        payload = await self._http.get_json(NAME, SUBMISSIONS_URL.format(cik=cik), headers=self._headers)
        sub = parse_wire(NAME, _EventSubmissions, payload)
        earnings = earnings_times(sub.filings.recent, since)
        for page in sub.filings.files:
            if page.filingTo < since:
                continue
            older = await self._http.get_json(
                NAME, SUBMISSIONS_FILE_URL.format(name=page.name), headers=self._headers
            )
            earnings.extend(earnings_times(parse_wire(NAME, _EventArrays, older), since))
        sector = ff12(sub.sic)
        return CompanyEvents(
            profile=CompanyProfile(
                symbol=symbol,
                cik=cik,
                name=sub.name or title,
                sic=sub.sic or None,
                sic_description=sub.sicDescription or None,
                sector=sector,
                sector_label=FF12_NAMES[sector],
            ),
            earnings=sorted(set(earnings)),
            earnings_since=since,
        )

    async def frame(self, taxonomy: str, tag: str, unit: str, period: str) -> list[FrameFact]:
        """One XBRL concept for every filer in one period (e.g. ``CY2023`` or ``CY2023Q4I``)."""
        payload = await self._http.get_json(
            NAME,
            FRAMES_URL.format(taxonomy=taxonomy, tag=tag, unit=unit, period=period),
            headers=self._headers,
            timeout=60.0,
        )
        rows = parse_wire(NAME, _Frame, payload).data
        if not rows:
            raise ProviderNoData(NAME, f"empty frame {tag} {period}")
        return [FrameFact(cik=r.cik, start=r.start, end=r.end, value=r.val, accn=r.accn) for r in rows]

    async def fundamentals(self, symbol: str) -> CompanyFundamentals:
        cik, title = await self.resolve_cik(symbol)
        facts_payload = await self._http.get_json(
            NAME, FACTS_URL.format(cik=cik), headers=self._headers, timeout=45.0
        )
        name, statements, shares = build_statements(facts_payload)
        try:
            sub_payload = await self._http.get_json(
                NAME, SUBMISSIONS_URL.format(cik=cik), headers=self._headers
            )
            filings = build_filings(sub_payload, cik)
        except (ProviderNoData, ProviderParseError):
            filings = []
        return CompanyFundamentals(
            symbol=symbol,
            cik=cik,
            name=name or title,
            shares_outstanding=shares[0] if shares else None,
            shares_as_of=shares[1] if shares else None,
            statements=statements,
            recent_filings=filings,
        )
