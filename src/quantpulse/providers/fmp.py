"""Financial Modeling Prep provider (API key): consensus analyst estimates and price targets.

Supports both the current ``/stable`` field names (``revenueAvg``…) and the legacy ``/api/v3`` names
(``estimatedRevenueAvg``…) so either response shape validates.
"""

from __future__ import annotations

from datetime import date

from pydantic import AliasChoices, Field

from quantpulse.core.errors import ProviderNotConfigured
from quantpulse.core.http import HttpClient
from quantpulse.providers.base import WireModel, finite_or_none, parse_wire, require
from quantpulse.schemas.fundamentals import AnalystEstimates, AnalystPeriodEstimate

NAME = "fmp"
BASE = "https://financialmodelingprep.com/stable"


class _Estimate(WireModel):
    date: date
    revenue_avg: float | None = Field(
        default=None, validation_alias=AliasChoices("revenueAvg", "estimatedRevenueAvg")
    )
    revenue_low: float | None = Field(
        default=None, validation_alias=AliasChoices("revenueLow", "estimatedRevenueLow")
    )
    revenue_high: float | None = Field(
        default=None, validation_alias=AliasChoices("revenueHigh", "estimatedRevenueHigh")
    )
    eps_avg: float | None = Field(default=None, validation_alias=AliasChoices("epsAvg", "estimatedEpsAvg"))
    analysts_revenue: int | None = Field(
        default=None, validation_alias=AliasChoices("numAnalystsRevenue", "numberAnalystEstimatedRevenue")
    )
    analysts_eps: int | None = Field(
        default=None, validation_alias=AliasChoices("numAnalystsEps", "numberAnalystsEstimatedEps")
    )


class _Target(WireModel):
    targetHigh: float | None = None
    targetLow: float | None = None
    targetConsensus: float | None = None


class FinancialModelingPrep:
    name = NAME

    def __init__(self, http: HttpClient, api_key: str | None) -> None:
        self._http = http
        self._key = api_key

    def configured(self) -> bool:
        return bool(self._key)

    async def estimates(self, symbol: str) -> AnalystEstimates:
        if not self._key:
            raise ProviderNotConfigured(NAME, "QP_FMP_API_KEY not set")
        payload = await self._http.get_json(
            NAME,
            f"{BASE}/analyst-estimates",
            params={"symbol": symbol, "period": "annual", "page": 0, "limit": 10, "apikey": self._key},
        )
        require(isinstance(payload, list) and bool(payload), NAME, f"no estimates for {symbol}")
        rows = sorted((parse_wire(NAME, _Estimate, item) for item in payload), key=lambda r: r.date)
        periods = [
            AnalystPeriodEstimate(
                period=r.date.isoformat(),
                end_date=r.date,
                revenue_avg=finite_or_none(r.revenue_avg),
                revenue_low=finite_or_none(r.revenue_low),
                revenue_high=finite_or_none(r.revenue_high),
                eps_avg=finite_or_none(r.eps_avg),
                analysts=r.analysts_revenue or r.analysts_eps,
            )
            for r in rows
        ]
        target = _Target()
        try:
            tp = await self._http.get_json(
                NAME, f"{BASE}/price-target-consensus", params={"symbol": symbol, "apikey": self._key}
            )
            if isinstance(tp, list) and tp:
                target = parse_wire(NAME, _Target, tp[0])
        except Exception:  # price targets are optional enrichment
            target = _Target()
        counts = [p.analysts for p in periods if p.analysts]
        return AnalystEstimates(
            symbol=symbol,
            target_mean_price=finite_or_none(target.targetConsensus),
            target_high_price=finite_or_none(target.targetHigh),
            target_low_price=finite_or_none(target.targetLow),
            analyst_count=max(counts) if counts else None,
            periods=periods,
        )
