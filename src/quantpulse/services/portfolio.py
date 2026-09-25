"""Portfolio CRUD and the quantitative risk laboratory (VaR/CVaR, ratios, efficient frontier)."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sqlalchemy.exc import IntegrityError

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.db import repositories as repo
from quantpulse.db.models import PortfolioRow
from quantpulse.db.session import Database
from quantpulse.quant import optimization as opt
from quantpulse.quant import risk
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta
from quantpulse.schemas.market import PriceHistory
from quantpulse.schemas.portfolio import (
    AssetStats,
    Correlation,
    EfficientFrontier,
    FrontierPoint,
    HoldingIn,
    HoldingOut,
    PerformanceMetrics,
    PortfolioIn,
    PortfolioOut,
    PortfolioRiskReport,
    PositionRisk,
    RiskRequest,
    ValuePoint,
    VaRResult,
)
from quantpulse.services.market import MarketService
from quantpulse.services.rates import RatesService

MIN_OBSERVATIONS = 60


def _to_out(row: PortfolioRow) -> PortfolioOut:
    return PortfolioOut(
        id=row.id,
        name=row.name,
        created_at=row.created_at,
        updated_at=row.updated_at,
        holdings=[
            HoldingOut(symbol=h.symbol, quantity=h.quantity, cost_basis=h.cost_basis) for h in row.holdings
        ],
    )


def closes_frame(histories: dict[str, PriceHistory]) -> pd.DataFrame:
    """Align daily closes on New York trading dates (inner join)."""
    series = {}
    for symbol, h in histories.items():
        idx = (
            pd.DatetimeIndex([b.timestamp for b in h.bars])
            .tz_convert("America/New_York")
            .normalize()
            .tz_localize(None)
        )
        s = pd.Series([b.close for b in h.bars], index=idx, dtype=float)
        series[symbol] = s[~s.index.duplicated(keep="last")]
    return pd.concat(series, axis=1, join="inner").sort_index().dropna()


def _point(p: opt.PortfolioPoint, symbols: list[str]) -> FrontierPoint:
    return FrontierPoint(
        expected_return=p.expected_return,
        volatility=p.volatility,
        sharpe=p.sharpe,
        weights={s: round(float(w), 6) for s, w in zip(symbols, p.weights, strict=True)},
    )


class PortfolioService:
    def __init__(
        self, settings: Settings, db: Database, clock: Clock, market: MarketService, rates: RatesService
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._market = market
        self._rates = rates

    # ------------------------------------------------------------------ CRUD
    async def list_all(self) -> list[PortfolioOut]:
        async with self._db.session() as s:
            return [_to_out(r) for r in await repo.list_portfolios(s)]

    async def get(self, portfolio_id: int) -> PortfolioOut:
        async with self._db.session() as s:
            return _to_out(await repo.get_portfolio(s, portfolio_id))

    async def create(self, data: PortfolioIn) -> PortfolioOut:
        try:
            async with self._db.session() as s:
                if await repo.portfolio_name_exists(s, data.name):
                    raise DomainError(f"a portfolio named '{data.name}' already exists")
                row = await repo.create_portfolio(
                    s, data.name, [(h.symbol, h.quantity, h.cost_basis) for h in data.holdings]
                )
                await s.refresh(row, ["holdings"])
                return _to_out(row)
        except IntegrityError as exc:
            raise DomainError("portfolio violates a uniqueness constraint") from exc

    async def update(self, portfolio_id: int, data: PortfolioIn) -> PortfolioOut:
        async with self._db.session() as s:
            if await repo.portfolio_name_exists(s, data.name, exclude_id=portfolio_id):
                raise DomainError(f"a portfolio named '{data.name}' already exists")
            row = await repo.replace_portfolio(
                s, portfolio_id, data.name, [(h.symbol, h.quantity, h.cost_basis) for h in data.holdings]
            )
            await s.refresh(row, ["holdings"])
            return _to_out(row)

    async def delete(self, portfolio_id: int) -> None:
        async with self._db.session() as s:
            await repo.delete_portfolio(s, portfolio_id)

    async def risk_for(self, portfolio_id: int, req: RiskRequest) -> CompositeEnvelope[PortfolioRiskReport]:
        p = await self.get(portfolio_id)
        return await self.analyze(
            [HoldingIn(symbol=h.symbol, quantity=h.quantity, cost_basis=h.cost_basis) for h in p.holdings],
            req,
        )

    # ------------------------------------------------------------------ analytics
    async def analyze(
        self, holdings: Sequence[HoldingIn], req: RiskRequest
    ) -> CompositeEnvelope[PortfolioRiskReport]:
        symbols = [h.symbol for h in holdings]
        bench = req.benchmark or self._settings.benchmark_symbol
        all_symbols = list(dict.fromkeys([*symbols, bench]))
        quotes, histories, curve_r = await asyncio.gather(
            self._market.quotes(symbols),
            asyncio.gather(*(self._market.history(s, "1d", req.lookback_days) for s in all_symbols)),
            self._rates.curve(),
        )
        hist_map = dict(zip(all_symbols, histories, strict=True))
        sources = {"yield_curve": curve_r.provenance}
        for s in symbols:
            sources[f"quote:{s}"] = quotes[s].provenance
        for s in all_symbols:
            sources[f"history:{s}"] = hist_map[s].provenance
        warnings: list[str] = []

        frame = closes_frame({s: hist_map[s].value for s in all_symbols})
        if len(frame) < MIN_OBSERVATIONS + 1:
            raise DomainError(
                f"only {len(frame)} overlapping trading days across holdings; at least {MIN_OBSERVATIONS + 1} required"
            )
        rets = frame.pct_change().dropna()
        asset_rets = rets[symbols].to_numpy()
        bench_rets = rets[bench].to_numpy()

        prices = np.array([quotes[s].value.price for s in symbols])
        qty = np.array([h.quantity for h in holdings])
        values = prices * qty
        total = float(values.sum())
        weights = values / total
        port = asset_rets @ weights
        rf = RatesService.rate_from_curve(curve_r.value, 0.25).bey_rate

        mean_vec = asset_rets.mean(axis=0)
        if req.covariance == "ledoit_wolf" and len(symbols) > 1:
            cov_daily, shrinkage = risk.ledoit_wolf(asset_rets)
        else:
            cov_daily = np.atleast_2d(np.cov(asset_rets, rowvar=False, ddof=1))
            shrinkage = None

        var_results = self._var_blocks(port, mean_vec, cov_daily, weights, req, total)
        rc = risk.risk_contributions(weights, cov_daily)
        positions = []
        for i, h in enumerate(holdings):
            col = asset_rets[:, i]
            positions.append(
                PositionRisk(
                    symbol=h.symbol,
                    quantity=h.quantity,
                    price=float(prices[i]),
                    market_value=float(values[i]),
                    weight=float(weights[i]),
                    cost_basis=h.cost_basis,
                    unrealized_pnl=None
                    if h.cost_basis is None
                    else float((prices[i] - h.cost_basis) * h.quantity),
                    annual_return=risk.annualized_return(col),
                    annual_volatility=risk.annualized_volatility(col),
                    beta=risk.beta(col, bench_rets),
                    risk_contribution=float(rc[i]),
                )
            )
        dates = rets.index
        metrics = PerformanceMetrics(
            annual_return=risk.annualized_return(port),
            annual_volatility=risk.annualized_volatility(port),
            sharpe=risk.sharpe_ratio(port, rf),
            sortino=risk.sortino_ratio(port, rf),
            max_drawdown=risk.max_drawdown(port),
            beta=risk.beta(port, bench_rets),
            risk_free_rate=rf,
            benchmark=bench,
            observations=len(port),
            start=dates[0].date(),
            end=dates[-1].date(),
        )

        frontier = None
        if len(symbols) >= 2:
            try:
                frontier = self._frontier(symbols, mean_vec, cov_daily, weights, rf, req, shrinkage)
            except DomainError as exc:
                warnings.append(f"Efficient frontier unavailable: {exc}")
        else:
            warnings.append("Efficient frontier requires at least two holdings.")

        corr = np.corrcoef(asset_rets, rowvar=False) if len(symbols) > 1 else np.array([[1.0]])
        value_series = frame[symbols].to_numpy() @ qty
        history = [
            ValuePoint(on=d.date(), value=round(float(v), 2))
            for d, v in zip(frame.index, value_series, strict=True)
        ]
        report = PortfolioRiskReport(
            portfolio_value=total,
            positions=positions,
            var=var_results,
            metrics=metrics,
            frontier=frontier,
            correlation=Correlation(
                symbols=symbols, matrix=[[round(float(x), 4) for x in row] for row in np.atleast_2d(corr)]
            ),
            value_history=history,
            warnings=warnings,
        )
        return CompositeEnvelope(data=report, meta=CompositeMeta.from_sources(sources, self._clock.now()))

    @staticmethod
    def _var_blocks(port, mean_vec, cov_daily, weights, req: RiskRequest, total: float) -> list[VaRResult]:
        c, h = req.confidence, req.horizon_days
        out: list[VaRResult] = []

        def add(method: str, var: float, cvar: float | None) -> None:
            out.append(
                VaRResult(
                    method=method,
                    confidence=c,
                    horizon_days=h,
                    var_pct=var,
                    var_amount=var * total,
                    cvar_pct=cvar,
                    cvar_amount=None if cvar is None else cvar * total,
                )
            )

        with contextlib.suppress(DomainError):  # too few overlapping h-day windows
            add("historical", *risk.historical_var_cvar(port, c, h))
        mu, sigma = float(port.mean()), float(port.std(ddof=1))
        add("parametric", *risk.parametric_var_cvar(mu, sigma, c, h))
        add("cornish_fisher", risk.cornish_fisher_var(port, c, h), None)
        add(
            "monte_carlo",
            *risk.monte_carlo_var_cvar(mean_vec, cov_daily, weights, c, h, req.monte_carlo_paths, req.seed),
        )
        return out

    @staticmethod
    def _frontier(
        symbols, mean_vec, cov_daily, weights, rf, req: RiskRequest, shrinkage
    ) -> EfficientFrontier:
        mu = mean_vec * risk.TRADING_DAYS
        cov = cov_daily * risk.TRADING_DAYS
        cov = (cov + cov.T) / 2.0
        bounds = (0.0, req.max_weight)
        points = opt.efficient_frontier(mu, cov, rf, bounds, req.frontier_points)
        gmv = opt.min_variance(mu, cov, rf, bounds)
        tangency = opt.max_sharpe(mu, cov, rf, bounds)
        n = len(symbols)
        equal = opt.stats(np.full(n, 1.0 / n), mu, cov, rf)
        current = opt.stats(weights, mu, cov, rf)
        cloud = opt.random_portfolios(mu, cov, rf, count=600, max_weight=req.max_weight)
        return EfficientFrontier(
            covariance_method=req.covariance if n > 1 else "sample",
            shrinkage=shrinkage,
            risk_free_rate=rf,
            assets=[
                AssetStats(symbol=s, expected_return=float(mu[i]), volatility=math.sqrt(float(cov[i, i])))
                for i, s in enumerate(symbols)
            ],
            points=[_point(p, symbols) for p in points],
            min_variance=_point(gmv, symbols),
            max_sharpe=_point(tangency, symbols),
            current=_point(current, symbols),
            equal_weight=_point(equal, symbols),
            random_portfolios=[(round(p.volatility, 5), round(p.expected_return, 5)) for p in cloud],
        )
