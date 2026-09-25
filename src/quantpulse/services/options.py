"""Options analytics: live chains enriched with model IV/Greeks, vol surfaces and BSM pricing."""

from __future__ import annotations

import asyncio
import functools
import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Protocol

import numpy as np

from quantpulse.config import Settings
from quantpulse.core.clock import Clock
from quantpulse.core.errors import DomainError
from quantpulse.core.gateway import DataGateway, Resolved, Source
from quantpulse.db import repositories as repo
from quantpulse.db.session import Database
from quantpulse.providers import synthetic
from quantpulse.quant.black_scholes import bsm_greeks, implied_volatility, implied_volatility_vec
from quantpulse.quant.vol_surface import build_vol_surface, years_to_expiry
from quantpulse.schemas.common import CompositeEnvelope, CompositeMeta
from quantpulse.schemas.options import (
    BSMInputsUsed,
    BSMRequest,
    BSMResult,
    GreeksOut,
    OptionChain,
    VolSurface,
    YieldCurve,
)
from quantpulse.services.market import MarketService
from quantpulse.services.rates import RatesService


class OptionsProvider(Protocol):
    name: str

    def configured(self) -> bool: ...

    async def option_chain(
        self, symbol: str, expirations: Sequence[date] | None, max_expirations: int = 8
    ) -> OptionChain: ...


def _greeks_out(g) -> GreeksOut:
    return GreeksOut(
        price=g.price,
        delta=g.delta,
        gamma=g.gamma,
        vega_per_pct=g.vega / 100.0,
        theta_per_day=g.theta / 365.0,
        rho_per_pct=g.rho / 100.0,
        vanna=g.vanna,
        vomma=g.vomma,
        charm_per_day=g.charm / 365.0,
        d1=g.d1,
        d2=g.d2,
    )


class OptionsService:
    def __init__(
        self,
        settings: Settings,
        gateway: DataGateway,
        db: Database,
        clock: Clock,
        market: MarketService,
        rates: RatesService,
        providers: Sequence[OptionsProvider],
    ) -> None:
        self._settings = settings
        self._gw = gateway
        self._db = db
        self._clock = clock
        self._market = market
        self._rates = rates
        self._providers = list(providers)

    def _rate_fn(self, curve: YieldCurve):
        return lambda t: RatesService.rate_from_curve(curve, max(t, 1e-6)).continuous_rate

    async def chain(
        self,
        symbol: str,
        expirations: Sequence[date] | None = None,
        max_expirations: int = 8,
        *,
        force_refresh: bool = False,
    ) -> tuple[Resolved[OptionChain], dict[str, Resolved]]:
        """Resolve a chain; also returns the quote/curve/dividend resolutions it depended on."""
        quote_r, curve_r, div_r = await asyncio.gather(
            self._market.quote(symbol), self._rates.curve(), self._market.dividend_yield(symbol)
        )
        exp_key = ",".join(e.isoformat() for e in expirations) if expirations else "auto"
        today = self._clock.now().date()

        async def persist(chain: OptionChain, provider: str) -> None:
            async with self._db.session() as s:
                rows = await repo.insert_option_snapshot(s, chain, provider)
                await repo.record_ingestion(s, "options", symbol, provider, rows)

        async def archive() -> tuple[OptionChain, datetime, str] | None:
            async with self._db.session() as s:
                found = await repo.latest_option_snapshot(s, symbol, expirations)
            if found is None:
                return None
            chain, at, provider = found
            if self._clock.now() - at > timedelta(days=7):
                return None
            live = [c for c in chain.contracts if c.expiration > today]
            if not live:
                return None
            return chain.model_copy(update={"contracts": live}), at, provider

        sources = [
            Source(
                p.name,
                functools.partial(p.option_chain, symbol, expirations, max_expirations),
                configured=p.configured(),
            )
            for p in self._providers
        ]
        chain_r = await self._gw.resolve(
            f"options:{symbol}:{exp_key}:{max_expirations}",
            sources,
            lambda: synthetic.synthetic_option_chain(
                symbol,
                quote_r.value.price,
                self._clock.now(),
                self._rate_fn(curve_r.value),
                div_r.value,
                list(expirations) if expirations else None,
                max_expirations,
            ),
            self._settings.ttl_options_chain,
            as_of=lambda c: c.as_of,
            archive=archive,
            on_live=persist,
            force_refresh=force_refresh,
        )
        return chain_r, {"quote": quote_r, "yield_curve": curve_r, "dividend_yield": div_r}

    def _as_of(self, chain: OptionChain) -> datetime:
        return min(chain.as_of, self._clock.now())

    async def analyzed_chain(
        self, symbol: str, expirations: Sequence[date] | None = None, max_expirations: int = 8
    ) -> CompositeEnvelope[OptionChain]:
        chain_r, deps = await self.chain(symbol, expirations, max_expirations)
        chain = chain_r.value
        curve = deps["yield_curve"].value
        q = float(deps["dividend_yield"].value)
        as_of = self._as_of(chain)
        rate_fn = self._rate_fn(curve)
        spot = chain.underlying_price

        contracts = list(chain.contracts)
        ts = (
            np.array([years_to_expiry(c.expiration, as_of) for c in contracts]) if contracts else np.array([])
        )
        rs = np.array([rate_fn(t) if t > 0 else 0.0 for t in ts])
        prices = np.array([c.reference_price or np.nan for c in contracts])
        is_call = np.array([c.kind == "call" for c in contracts])
        strikes = np.array([c.strike for c in contracts])
        ivs = implied_volatility_vec(prices, spot, strikes, ts, rs, q, is_call) if contracts else np.array([])

        enriched = []
        for i, c in enumerate(contracts):
            t = float(ts[i])
            iv = float(ivs[i]) if math.isfinite(ivs[i]) else None
            sigma = iv or c.implied_volatility
            update: dict[str, object] = {"mid": c.reference_price, "model_iv": iv}
            if t > 0 and sigma and sigma > 0:
                g = bsm_greeks(spot, c.strike, t, float(rs[i]), sigma, q, c.kind)
                update.update(
                    delta=g.delta, gamma=g.gamma, theta_per_day=g.theta / 365.0, vega_per_pct=g.vega / 100.0
                )
            enriched.append(c.model_copy(update=update))
        data = chain.model_copy(update={"contracts": enriched})
        sources = {"option_chain": chain_r.provenance, **{k: v.provenance for k, v in deps.items()}}
        return CompositeEnvelope(data=data, meta=CompositeMeta.from_sources(sources, self._clock.now()))

    async def surface(
        self,
        symbol: str,
        max_expirations: int = 8,
        moneyness_range: tuple[float, float] = (0.7, 1.3),
        grid_points: int = 25,
    ) -> CompositeEnvelope[VolSurface]:
        chain_r, deps = await self.chain(symbol, None, max_expirations)
        chain = chain_r.value
        surface = build_vol_surface(
            symbol,
            chain.contracts,
            chain.underlying_price,
            self._as_of(chain),
            self._rate_fn(deps["yield_curve"].value),
            float(deps["dividend_yield"].value),
            moneyness_range=moneyness_range,
            grid_points=grid_points,
        )
        sources = {"option_chain": chain_r.provenance, **{k: v.provenance for k, v in deps.items()}}
        return CompositeEnvelope(data=surface, meta=CompositeMeta.from_sources(sources, self._clock.now()))

    async def price(self, req: BSMRequest) -> CompositeEnvelope[BSMResult]:
        now = self._clock.now()
        sources = {}
        if req.days_to_expiry is not None:
            t = req.days_to_expiry / 365.0
        else:
            t = years_to_expiry(req.expiration, now)  # type: ignore[arg-type]
        if t <= 0:
            raise DomainError("the option has already expired")

        spot, spot_source = req.spot, "user"
        if spot is None:
            quote_r = await self._market.quote(req.symbol)  # type: ignore[arg-type]
            spot, spot_source = (
                quote_r.value.price,
                f"{quote_r.provenance.provider} quote ({quote_r.status.value})",
            )
            sources["quote"] = quote_r.provenance

        if req.rate is not None:
            rate, rate_source = req.rate, "user"
        else:
            rate_info, curve_r = await self._rates.rate_at(t)
            rate = rate_info.continuous_rate
            rate_source = f"Treasury curve {rate_info.curve_date} ({curve_r.status.value}), continuous"
            sources["yield_curve"] = curve_r.provenance

        if req.dividend_yield is not None:
            q = req.dividend_yield
        elif req.symbol:
            div_r = await self._market.dividend_yield(req.symbol)
            q = float(div_r.value)
            sources["dividend_yield"] = div_r.provenance
        else:
            q = 0.0

        implied: float | None = None
        if req.volatility is not None:
            sigma, vol_source = req.volatility, "user"
        elif req.market_price is not None:
            implied = implied_volatility(req.market_price, spot, req.strike, t, rate, q, req.kind)
            if implied is None:
                raise DomainError(
                    "market_price is outside BSM no-arbitrage bounds; no implied volatility exists"
                )
            sigma, vol_source = implied, "implied from market_price"
        else:
            sigma, vol_source, extra = await self._live_vol(req.symbol, req.strike, t, spot)  # type: ignore[arg-type]
            sources.update(extra)

        if req.market_price is not None and implied is None:
            implied = implied_volatility(req.market_price, spot, req.strike, t, rate, q, req.kind)

        g = bsm_greeks(spot, req.strike, t, rate, sigma, q, req.kind)
        other = bsm_greeks(spot, req.strike, t, rate, sigma, q, "put" if req.kind == "call" else "call")
        result = BSMResult(
            inputs=BSMInputsUsed(
                spot=spot,
                strike=req.strike,
                years_to_expiry=t,
                rate=rate,
                volatility=sigma,
                dividend_yield=q,
                kind=req.kind,
                volatility_source=vol_source,
                rate_source=rate_source,
                spot_source=spot_source,
            ),
            greeks=_greeks_out(g),
            implied_volatility=implied,
            counterpart_price=other.price,
        )
        if not sources:
            from quantpulse.schemas.common import DataStatus, Provenance

            sources["inputs"] = Provenance(
                status=DataStatus.LIVE,
                provider="user",
                as_of=now,
                fetched_at=now,
                message="all inputs supplied by the caller",
            )
        return CompositeEnvelope(data=result, meta=CompositeMeta.from_sources(sources, now))

    async def _live_vol(self, symbol: str, strike: float, t: float, spot: float) -> tuple[float, str, dict]:
        """Volatility for a strike/maturity from the live smile, falling back to 3-month realised vol."""
        try:
            chain_r, deps = await self.chain(symbol, None, 12)
            chain = chain_r.value
            as_of = self._as_of(chain)
            target = min(chain.expirations, key=lambda e: abs(years_to_expiry(e, as_of) - t), default=None)
            if target is not None:
                sub = [c for c in chain.contracts if c.expiration == target]
                if not sub:
                    chain_r, deps = await self.chain(symbol, [target], 1)
                    chain = chain_r.value
                    sub = [c for c in chain.contracts if c.expiration == target]
                surface = build_vol_surface(
                    symbol,
                    sub,
                    chain.underlying_price,
                    as_of,
                    self._rate_fn(deps["yield_curve"].value),
                    float(deps["dividend_yield"].value),
                )
                if surface.smiles:
                    smile = surface.smiles[0]
                    k = math.log(strike / smile.forward)
                    pts = sorted(smile.points, key=lambda p: p.log_moneyness)
                    xs = np.array([p.log_moneyness for p in pts])
                    ys = np.array([p.iv for p in pts])
                    iv = float(np.interp(k, xs, ys))  # flat beyond the quoted wings
                    return (
                        iv,
                        f"live smile ({target}, {chain_r.status.value})",
                        {"option_chain": chain_r.provenance},
                    )
        except DomainError:
            pass
        hist_r = await self._market.history(symbol, "1d", 120)
        closes = np.array(hist_r.value.closes)
        if closes.size < 22:
            raise DomainError("not enough data to estimate volatility; pass 'volatility' explicitly")
        rets = np.diff(np.log(closes[-64:]))
        sigma = float(rets.std(ddof=1) * math.sqrt(252))
        if not sigma > 0:
            raise DomainError("realised volatility is zero; pass 'volatility' explicitly")
        return sigma, f"realised 3m ({hist_r.status.value})", {"price_history": hist_r.provenance}
