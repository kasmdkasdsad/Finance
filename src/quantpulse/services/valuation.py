"""Live-data DCF valuation: assumptions are derived from SEC fundamentals, consensus estimates, the live
share price and the Treasury curve; every derived number is recorded with its source."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pandas as pd

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.quant.dcf import fade_path, growth_path, run_dcf, sensitivity_grid
from quantpulse.quant.monte_carlo import simulate_dcf
from quantpulse.quant.risk import beta as regression_beta
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta, DataStatus
from quantpulse.schemas.fundamentals import (
    AnalystEstimates,
    CompanyFundamentals,
    DCFAssumptionNote,
    DCFInputs,
    DCFRequest,
    ValuationReport,
    WACCBreakdown,
)
from quantpulse.schemas.market import PriceHistory
from quantpulse.services.fundamentals import FundamentalsService
from quantpulse.services.market import MarketService
from quantpulse.services.rates import RatesService


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def weekly_returns(history: PriceHistory) -> pd.Series:
    closes = pd.Series(
        [b.close for b in history.bars],
        index=pd.DatetimeIndex([b.timestamp for b in history.bars])
        .tz_convert("America/New_York")
        .tz_localize(None),
    )
    weekly = closes.groupby(closes.index.to_period("W-FRI")).last()
    return weekly.pct_change().dropna()


def consensus_growth(fund: CompanyFundamentals, est: AnalystEstimates, max_years: int = 3) -> list[float]:
    """Chain annual consensus revenue estimates after the last reported fiscal year into growth rates."""
    latest = fund.latest
    if latest is None or not latest.revenue:
        return []
    annual = sorted(
        (
            p
            for p in est.periods
            if p.end_date and p.revenue_avg and p.revenue_avg > 0 and not p.period.endswith("q")
        ),
        key=lambda p: p.end_date,  # type: ignore[arg-type,return-value]
    )
    growth: list[float] = []
    prev_rev, prev_end = latest.revenue, latest.period_end
    for p in annual:
        gap = (p.end_date - prev_end).days  # type: ignore[operator]
        if gap < 300:
            continue
        if gap > 430:
            break
        g = p.revenue_avg / prev_rev - 1.0  # type: ignore[operator]
        if not -0.5 <= g <= 1.0:
            break
        growth.append(g)
        prev_rev, prev_end = p.revenue_avg, p.end_date  # type: ignore[assignment]
        if len(growth) >= max_years:
            break
    return growth


class ValuationService:
    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        market: MarketService,
        rates: RatesService,
        fundamentals: FundamentalsService,
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._market = market
        self._rates = rates
        self._fund = fundamentals

    async def dcf(self, symbol: str, req: DCFRequest) -> CompositeEnvelope[ValuationReport]:
        fund_r, est_r, quote_r, curve_r = await asyncio.gather(
            self._fund.fundamentals(symbol),
            self._fund.estimates(symbol),
            self._market.quote(symbol),
            self._rates.curve(),
        )
        fund, est, quote = fund_r.value, est_r.value, quote_r.value
        sources = {
            "fundamentals": fund_r.provenance,
            "estimates": est_r.provenance,
            "quote": quote_r.provenance,
            "yield_curve": curve_r.provenance,
        }
        notes: list[DCFAssumptionNote] = []
        warnings: list[str] = []
        if fund_r.status is DataStatus.SYNTHETIC:
            warnings.append(
                "Fundamentals are synthetic (SEC EDGAR unavailable) — valuation is illustrative only."
            )
        latest = fund.latest
        if latest is None or not latest.revenue or latest.revenue <= 0:
            raise DomainError(f"no reported revenue for {symbol}; a revenue-driven DCF is not applicable")
        revenue = latest.revenue

        def note(field: str, value: float, source: str) -> float:
            notes.append(DCFAssumptionNote(field=field, value=value, source=source))
            return value

        # --- revenue growth
        if req.revenue_growth is not None:
            growth = list(req.revenue_growth)
            note("revenue_growth_y1", growth[0], "user override")
        else:
            near = consensus_growth(fund, est)
            if near:
                note(
                    "revenue_growth_y1", near[0], f"consensus revenue estimates ({est_r.provenance.provider})"
                )
            else:
                near = [self._historical_cagr(fund)]
                note("revenue_growth_y1", near[0], "historical revenue CAGR (no usable consensus estimates)")
                warnings.append(
                    "No usable consensus revenue estimates; near-term growth uses historical CAGR."
                )
            growth = growth_path(near, req.terminal_growth, req.years)

        # --- margins
        if req.ebit_margin is not None:
            start_margin = note("ebit_margin_start", req.ebit_margin, "user override")
        elif latest.operating_income is not None:
            start_margin = note(
                "ebit_margin_start",
                latest.operating_income / revenue,
                f"FY{latest.fiscal_year} operating income / revenue",
            )
        else:
            start_margin = note("ebit_margin_start", 0.15, "default (operating income not reported)")
            warnings.append("Operating income not reported; EBIT margin defaults to 15%.")
        target_margin = req.target_ebit_margin if req.target_ebit_margin is not None else start_margin
        if req.target_ebit_margin is not None:
            note("ebit_margin_target", target_margin, "user override")
        margins = fade_path(start_margin, target_margin, req.years)

        # --- tax, reinvestment
        if req.tax_rate is not None:
            tax = note("tax_rate", req.tax_rate, "user override")
        elif (
            latest.income_tax is not None
            and latest.pretax_income
            and latest.pretax_income > 0
            and 0 <= latest.income_tax / latest.pretax_income <= 0.5
        ):
            tax = note(
                "tax_rate",
                latest.income_tax / latest.pretax_income,
                f"FY{latest.fiscal_year} effective tax rate",
            )
        else:
            tax = note("tax_rate", self._settings.default_tax_rate, "default statutory rate")
        if req.da_pct_revenue is not None:
            da = note("da_pct_revenue", req.da_pct_revenue, "user override")
        elif latest.depreciation_amortization is not None:
            da = note(
                "da_pct_revenue",
                _clamp(latest.depreciation_amortization / revenue, 0, 0.3),
                f"FY{latest.fiscal_year} D&A / revenue",
            )
        else:
            da = note("da_pct_revenue", 0.04, "default (D&A not reported)")
        if req.capex_pct_revenue is not None:
            capex = note("capex_pct_revenue", req.capex_pct_revenue, "user override")
        elif latest.capital_expenditure is not None:
            capex = note(
                "capex_pct_revenue",
                _clamp(latest.capital_expenditure / revenue, 0, 0.5),
                f"FY{latest.fiscal_year} capex / revenue",
            )
        else:
            capex = note("capex_pct_revenue", da, "default: capex = D&A (capex not reported)")
        if req.nwc_pct_incremental_revenue is not None:
            nwc = note("nwc_pct_incremental_revenue", req.nwc_pct_incremental_revenue, "user override")
        elif latest.current_assets is not None and latest.current_liabilities is not None:
            op_nwc = (latest.current_assets - (latest.cash or 0.0)) - latest.current_liabilities
            nwc = note(
                "nwc_pct_incremental_revenue",
                _clamp(op_nwc / revenue, -0.15, 0.25),
                "non-cash NWC / revenue (clamped −15%…25%)",
            )
        else:
            nwc = note("nwc_pct_incremental_revenue", 0.05, "default (balance sheet incomplete)")

        # --- capital structure & WACC
        shares = fund.shares_outstanding or latest.diluted_shares or quote.shares_outstanding
        if not shares:
            raise DomainError(f"share count unavailable for {symbol}")
        cash = max(latest.cash or 0.0, 0.0)
        debt = max(latest.total_debt or 0.0, 0.0)
        wacc_info = await self._wacc(
            symbol,
            req,
            quote.price,
            shares,
            debt,
            tax,
            latest.interest_expense,
            est,
            curve_r.value,
            sources,
            warnings,
        )
        wacc = wacc_info.wacc
        if req.terminal_growth >= wacc - 0.005:
            raise DomainError(
                f"terminal growth {req.terminal_growth:.2%} must be at least 0.5pp below WACC {wacc:.2%}"
            )

        inputs = DCFInputs(
            base_revenue=revenue,
            revenue_growth=growth,
            ebit_margin=margins,
            tax_rate=tax,
            da_pct_revenue=da,
            capex_pct_revenue=capex,
            nwc_pct_incremental_revenue=nwc,
            wacc=wacc,
            terminal_growth=req.terminal_growth,
            cash=cash,
            debt=debt,
            shares_outstanding=shares,
            mid_year_convention=req.mid_year_convention,
            current_price=quote.price,
        )
        out = run_dcf(inputs)
        if out.projections[-1].free_cash_flow <= 0:
            warnings.append(
                "Terminal-year free cash flow is negative; the Gordon terminal value is not meaningful."
            )
        if out.terminal_value_share > 0.85:
            warnings.append(
                f"Terminal value is {out.terminal_value_share:.0%} of EV — results are highly sensitive to WACC and g."
            )
        mc = simulate_dcf(inputs, req.monte_carlo) if req.monte_carlo is not None else None
        report = ValuationReport(
            symbol=symbol,
            name=fund.name or quote.name,
            inputs=inputs,
            assumptions=notes,
            wacc=wacc_info,
            dcf=out,
            sensitivity=sensitivity_grid(inputs),
            monte_carlo=mc,
            warnings=warnings,
        )
        return CompositeEnvelope(data=report, meta=CompositeMeta.from_sources(sources, self._clock.now()))

    @staticmethod
    def _historical_cagr(fund: CompanyFundamentals) -> float:
        revs = [s.revenue for s in fund.statements if s.revenue and s.revenue > 0]
        if len(revs) < 2:
            return 0.03
        k = min(3, len(revs) - 1)
        return _clamp((revs[-1] / revs[-1 - k]) ** (1 / k) - 1, -0.10, 0.30)

    async def _wacc(
        self,
        symbol: str,
        req: DCFRequest,
        price: float,
        shares: float,
        debt: float,
        tax: float,
        interest_expense: float | None,
        est: AnalystEstimates,
        curve,
        sources: dict,
        warnings: list[str],
    ) -> WACCBreakdown:
        rf = RatesService.rate_from_curve(curve, 10.0).bey_rate
        erp = (
            req.equity_risk_premium
            if req.equity_risk_premium is not None
            else self._settings.equity_risk_premium
        )

        if req.beta is not None:
            beta, beta_source = req.beta, "user override"
        else:
            estimated, beta_source = await self._regression_beta(symbol, sources)
            if estimated is not None:
                beta = estimated
            elif est.beta is not None:
                beta, beta_source = est.beta, "provider-reported beta"
            else:
                beta, beta_source = 1.0, "default (market beta)"
                warnings.append("Beta could not be estimated; using 1.0.")
        cost_equity = rf + beta * erp

        if interest_expense and interest_expense > 0 and debt > 0:
            raw = interest_expense / debt
            rd = _clamp(raw, rf, rf + 0.10)
            rd_source = "interest expense / total debt" + (" (clamped to [rf, rf+10%])" if rd != raw else "")
        else:
            rd = rf + self._settings.default_credit_spread
            rd_source = f"risk-free + {self._settings.default_credit_spread:.2%} default spread (interest expense not reported)"
        equity_value = price * shares
        total = equity_value + debt
        we, wd = equity_value / total, debt / total
        wacc = we * cost_equity + wd * rd * (1 - tax)
        if req.wacc is not None:
            wacc = req.wacc
        return WACCBreakdown(
            risk_free_rate=rf,
            beta=beta,
            beta_source=beta_source,
            equity_risk_premium=erp,
            cost_of_equity=cost_equity,
            pre_tax_cost_of_debt=rd,
            cost_of_debt_source=rd_source,
            tax_rate=tax,
            market_value_equity=equity_value,
            debt=debt,
            weight_equity=we,
            weight_debt=wd,
            wacc=wacc,
        )

    async def _regression_beta(self, symbol: str, sources: dict) -> tuple[float | None, str]:
        bench = self._settings.benchmark_symbol
        if symbol == bench:
            return 1.0, f"{bench} is the benchmark"
        stock_r, bench_r = await asyncio.gather(
            self._market.history(symbol, "1d", 730), self._market.history(bench, "1d", 730)
        )
        sources["beta_history"] = stock_r.provenance
        sources["benchmark_history"] = bench_r.provenance
        joined = pd.concat(
            [weekly_returns(stock_r.value), weekly_returns(bench_r.value)], axis=1, join="inner"
        ).dropna()
        if len(joined) < 52:
            return None, "insufficient history"
        b = regression_beta(joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy())
        if b is None or not math.isfinite(b):
            return None, "degenerate regression"
        status = "live" if stock_r.status in (DataStatus.LIVE, DataStatus.CACHED) else stock_r.status.value
        return float(np.round(b, 4)), f"2y weekly regression vs {bench} ({status})"
