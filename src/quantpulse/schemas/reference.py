"""Reference data: company profiles, earnings events, index membership and fundamentals facts."""

from __future__ import annotations

from datetime import date

from pydantic import AwareDatetime, Field

from quantpulse.schemas.common import StrictModel


class CompanyProfile(StrictModel):
    symbol: str
    cik: str
    name: str
    sic: str | None = None
    sic_description: str | None = None
    sector: str = Field(description="Fama-French 12 industry code derived from the SIC code")
    sector_label: str
    gics_sector: str | None = Field(default=None, description="GICS sector for current S&P 500 members")


class CompanyEvents(StrictModel):
    profile: CompanyProfile
    earnings: list[AwareDatetime] = Field(
        description="Earnings releases: acceptance times of 8-K filings with item 2.02, ascending"
    )
    earnings_since: date = Field(description="Filings before this date were not scanned")


class EarningsReaction(StrictModel):
    announced_at: AwareDatetime
    reaction_date: date
    stock_return: float
    benchmark_return: float | None
    abnormal_return: float | None


class EarningsOut(StrictModel):
    symbol: str
    last: AwareDatetime | None
    next_date: date | None
    next_source: str | None = Field(
        description="'scheduled' (vendor calendar) or 'estimated' (quarterly cadence)"
    )
    next_reaction_date: date | None = Field(
        default=None, description="The session expected to price the next release (the day, or the next one)"
    )
    days_to_next: int | None
    typical_move: float | None = Field(description="RMS of past earnings-day returns")
    reactions: list[EarningsReaction]


class FrameFact(StrictModel):
    cik: int
    start: date | None = None
    end: date
    value: float
    accn: str
