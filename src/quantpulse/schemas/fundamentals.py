"""Fundamental data, analyst estimates, DCF and Monte Carlo schemas."""

from __future__ import annotations

from datetime import date

from pydantic import Field, model_validator

from quantpulse.schemas.common import StrictModel


class FinancialStatement(StrictModel):
    """One fiscal year of normalised statement line items (USD)."""

    fiscal_year: int
    period_end: date
    form: str = "10-K"
    filed: date | None = None
    accession: str | None = None
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    pretax_income: float | None = None
    income_tax: float | None = None
    interest_expense: float | None = None
    depreciation_amortization: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    stockholders_equity: float | None = None
    cash: float | None = None
    total_debt: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    diluted_eps: float | None = None
    diluted_shares: float | None = None

    @property
    def free_cash_flow(self) -> float | None:
        if self.operating_cash_flow is None or self.capital_expenditure is None:
            return None
        return self.operating_cash_flow - self.capital_expenditure

    @property
    def net_working_capital(self) -> float | None:
        if self.current_assets is None or self.current_liabilities is None:
            return None
        return self.current_assets - self.current_liabilities


class Filing(StrictModel):
    form: str
    filing_date: date
    report_date: date | None = None
    accession: str
    primary_document: str | None = None
    url: str | None = None


class CompanyFundamentals(StrictModel):
    symbol: str
    cik: str | None = None
    name: str | None = None
    currency: str = "USD"
    shares_outstanding: float | None = Field(default=None, gt=0)
    shares_as_of: date | None = None
    statements: list[FinancialStatement] = Field(description="Annual statements, oldest first.")
    recent_filings: list[Filing] = Field(default_factory=list)

    @property
    def latest(self) -> FinancialStatement | None:
        return self.statements[-1] if self.statements else None


class AnalystPeriodEstimate(StrictModel):
    period: str = Field(description="e.g. '0y' (current FY), '+1y', or an ISO fiscal-year end date")
    end_date: date | None = None
    revenue_avg: float | None = None
    revenue_low: float | None = None
    revenue_high: float | None = None
    revenue_growth: float | None = None
    eps_avg: float | None = None
    eps_growth: float | None = None
    analysts: int | None = None


class AnalystEstimates(StrictModel):
    symbol: str
    target_mean_price: float | None = None
    target_high_price: float | None = None
    target_low_price: float | None = None
    recommendation_mean: float | None = Field(default=None, description="1 = strong buy … 5 = sell")
    recommendation_key: str | None = None
    analyst_count: int | None = None
    long_term_growth: float | None = None
    beta: float | None = None
    periods: list[AnalystPeriodEstimate] = Field(default_factory=list)


class WACCBreakdown(StrictModel):
    risk_free_rate: float
    beta: float
    beta_source: str
    equity_risk_premium: float
    cost_of_equity: float
    pre_tax_cost_of_debt: float
    cost_of_debt_source: str
    tax_rate: float
    market_value_equity: float
    debt: float
    weight_equity: float
    weight_debt: float
    wacc: float


class DCFInputs(StrictModel):
    """Fully specified DCF model inputs (all rates decimal)."""

    base_revenue: float = Field(gt=0)
    revenue_growth: list[float] = Field(min_length=1, max_length=30)
    ebit_margin: list[float] = Field(min_length=1, max_length=30)
    tax_rate: float = Field(ge=0, le=0.6)
    da_pct_revenue: float = Field(ge=0, le=1)
    capex_pct_revenue: float = Field(ge=0, le=1)
    nwc_pct_incremental_revenue: float = Field(ge=-1, le=1)
    wacc: float = Field(gt=0, le=0.5)
    terminal_growth: float = Field(ge=-0.05, le=0.1)
    cash: float = Field(ge=0)
    debt: float = Field(ge=0)
    shares_outstanding: float = Field(gt=0)
    mid_year_convention: bool = True
    current_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check(self) -> DCFInputs:
        if len(self.revenue_growth) != len(self.ebit_margin):
            raise ValueError("revenue_growth and ebit_margin must have one value per projection year")
        if self.terminal_growth >= self.wacc:
            raise ValueError("terminal_growth must be strictly below wacc for a finite terminal value")
        if any(g <= -1 for g in self.revenue_growth):
            raise ValueError("revenue growth must be greater than -100%")
        return self

    @property
    def years(self) -> int:
        return len(self.revenue_growth)


class DCFProjectionRow(StrictModel):
    year: int
    revenue: float
    growth: float
    ebit: float
    ebit_margin: float
    nopat: float
    depreciation_amortization: float
    capital_expenditure: float
    change_in_nwc: float
    free_cash_flow: float
    discount_factor: float
    present_value: float


class DCFOutput(StrictModel):
    projections: list[DCFProjectionRow]
    sum_pv_fcf: float
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    current_price: float | None
    upside: float | None
    terminal_value_share: float = Field(description="PV(TV) / EV")


class SensitivityGrid(StrictModel):
    wacc_values: list[float]
    growth_values: list[float]
    values_per_share: list[list[float | None]] = Field(description="rows = wacc, cols = terminal growth")


class MonteCarloConfig(StrictModel):
    paths: int = Field(default=10_000, ge=100, le=200_000)
    seed: int | None = Field(default=None, ge=0)
    wacc_sd: float = Field(default=0.01, ge=0, le=0.1)
    terminal_growth_sd: float = Field(default=0.005, ge=0, le=0.05)
    revenue_growth_sd: float = Field(default=0.03, ge=0, le=0.5)
    margin_sd: float = Field(default=0.02, ge=0, le=0.5)
    bins: int = Field(default=40, ge=5, le=200)


class MonteCarloResult(StrictModel):
    paths: int
    valid_paths: int
    seed: int
    mean: float
    median: float
    std: float
    percentiles: dict[str, float]
    prob_above_price: float | None
    current_price: float | None
    histogram_edges: list[float]
    histogram_counts: list[int]


class DCFRequest(StrictModel):
    """Live-data DCF. Every field is an optional override of the auto-derived assumption."""

    years: int = Field(default=5, ge=3, le=15)
    revenue_growth: list[float] | None = Field(default=None, description="Explicit growth per year.")
    terminal_growth: float = Field(default=0.025, ge=-0.02, le=0.06)
    ebit_margin: float | None = Field(default=None, ge=-1, le=1)
    target_ebit_margin: float | None = Field(default=None, ge=-1, le=1)
    tax_rate: float | None = Field(default=None, ge=0, le=0.6)
    wacc: float | None = Field(default=None, gt=0, le=0.5)
    equity_risk_premium: float | None = Field(default=None, ge=0, le=0.2)
    beta: float | None = Field(default=None, ge=-2, le=5)
    da_pct_revenue: float | None = Field(default=None, ge=0, le=1)
    capex_pct_revenue: float | None = Field(default=None, ge=0, le=1)
    nwc_pct_incremental_revenue: float | None = Field(default=None, ge=-1, le=1)
    mid_year_convention: bool = True
    monte_carlo: MonteCarloConfig | None = Field(default_factory=MonteCarloConfig)

    @model_validator(mode="after")
    def _growth_len(self) -> DCFRequest:
        if self.revenue_growth is not None and len(self.revenue_growth) != self.years:
            raise ValueError("revenue_growth must contain exactly `years` values")
        return self


class DCFAssumptionNote(StrictModel):
    field: str
    value: float
    source: str


class ValuationReport(StrictModel):
    symbol: str
    name: str | None
    inputs: DCFInputs
    assumptions: list[DCFAssumptionNote]
    wacc: WACCBreakdown
    dcf: DCFOutput
    sensitivity: SensitivityGrid
    monte_carlo: MonteCarloResult | None
    warnings: list[str] = Field(default_factory=list)
