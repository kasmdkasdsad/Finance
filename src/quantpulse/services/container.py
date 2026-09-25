"""Composition root: builds providers, the gateway, the database and all services from settings."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from quantpulse import __version__
from quantpulse.config import Settings
from quantpulse.core.cache import TTLCache
from quantpulse.core.clock import Clock, SystemClock
from quantpulse.core.gateway import DataGateway
from quantpulse.core.http import HttpClient
from quantpulse.core.jobs import JobRegistry
from quantpulse.core.market_calendar import next_open, session_at
from quantpulse.core.rate_limit import TokenBucket
from quantpulse.db import migrate
from quantpulse.db.session import Database
from quantpulse.providers.alpaca import Alpaca
from quantpulse.providers.alpaca_trading import AlpacaPaperBroker
from quantpulse.providers.eia import EIA
from quantpulse.providers.espn import ESPN
from quantpulse.providers.fmp import FinancialModelingPrep
from quantpulse.providers.fueleconomy import FuelEconomyGov
from quantpulse.providers.odds_api import OddsAPI
from quantpulse.providers.polygon import Polygon
from quantpulse.providers.sec_edgar import SecEdgar
from quantpulse.providers.sp500 import SP500Wikipedia
from quantpulse.providers.treasury import Treasury
from quantpulse.providers.yahoo import YahooFinance
from quantpulse.services.backfill import BackfillService
from quantpulse.services.facts import FactsService
from quantpulse.services.forecast import ForecastService
from quantpulse.services.fundamentals import FundamentalsService
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.notifications import EmailNotifier
from quantpulse.services.options import OptionsService
from quantpulse.services.picks import PicksService
from quantpulse.services.portfolio import PortfolioService
from quantpulse.services.predictions import PredictionService
from quantpulse.services.rates import RatesService
from quantpulse.services.reference import ReferenceService
from quantpulse.services.sandbox import SandboxService
from quantpulse.services.sports import SportsService
from quantpulse.services.stocks import StockReportService
from quantpulse.services.trading import TradingService
from quantpulse.services.trading_data import TradingDataLoader
from quantpulse.services.valuation import ValuationService
from quantpulse.services.vehicle import VehicleService

logger = logging.getLogger(__name__)


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None


def build_http(
    settings: Settings, clock: Clock, transport: httpx.AsyncBaseTransport | None = None
) -> HttpClient:
    wait = settings.rate_limit_max_wait_seconds
    per_min = settings.polygon_requests_per_minute
    limits = {
        # name: (tokens per second, burst capacity, max in-flight)
        "yahoo": (2.0, 4, 2),
        "polygon": (per_min / 60.0, max(1.0, per_min), 1),
        "alpaca": (3.0, 10, 4),
        "treasury": (1.0, 2, 1),
        "sec_edgar": (8.0, 8, 4),
        "fmp": (3.0, 5, 2),
        "eia": (2.0, 5, 2),
        "fueleconomy_gov": (2.0, 4, 2),
        "espn": (8.0, 16, 4),
        "odds_api": (1.0, 2, 1),
        "wikipedia": (1.0, 2, 1),
    }
    return HttpClient(
        timeout=settings.http_timeout_seconds,
        max_retries=settings.http_max_retries,
        limiters={
            n: TokenBucket(n, rate, cap, max_wait=wait, clock=clock) for n, (rate, cap, _) in limits.items()
        },
        concurrency={n: c for n, (_, _, c) in limits.items()},
        transport=transport,
    )


class Container:
    def __init__(
        self,
        settings: Settings,
        clock: Clock | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        http: HttpClient | None = None,
        broker: AlpacaPaperBroker | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock or SystemClock()
        self.cache = TTLCache(settings.cache_max_entries, self.clock)
        self.gateway = DataGateway(
            self.cache,
            live_enabled=settings.enable_live_data,
            failure_threshold=settings.circuit_failure_threshold,
            cooldown_seconds=settings.circuit_cooldown_seconds,
            stale_grace_seconds=settings.stale_grace_seconds,
            clock=self.clock,
        )
        # Rate limiting is about real elapsed time, so limiters always use the system clock.
        self.http = http or build_http(settings, SystemClock(), transport)
        self.db = Database(settings.database_url)
        self.jobs = JobRegistry(self.clock)

        # providers
        self.yahoo = YahooFinance(self.http)
        self.polygon = Polygon(self.http, _secret(settings.polygon_api_key), settings.polygon_base_url)
        self.alpaca = Alpaca(
            self.http,
            _secret(settings.alpaca_api_key_id),
            _secret(settings.alpaca_api_secret_key),
            settings.alpaca_data_url,
            settings.alpaca_stock_feed,
            settings.alpaca_options_feed,
        )
        self.treasury = Treasury(self.http)
        self.sec = SecEdgar(self.http, settings.sec_user_agent)
        self.wikipedia = SP500Wikipedia(self.http, settings.sec_user_agent)
        self.fmp = FinancialModelingPrep(self.http, _secret(settings.fmp_api_key))
        self.eia = EIA(self.http, _secret(settings.eia_api_key))
        self.fueleconomy = FuelEconomyGov(self.http)
        self.espn = ESPN(self.http)
        self.odds = OddsAPI(self.http, _secret(settings.odds_api_key), settings.odds_bookmaker_regions)
        by_name: dict[str, YahooFinance | Polygon | Alpaca] = {
            "polygon": self.polygon,
            "alpaca": self.alpaca,
            "yahoo": self.yahoo,
        }
        market_providers = [by_name[n] for n in settings.market_providers]

        # services
        self.market = MarketService(settings, self.gateway, self.db, self.clock, market_providers, self.yahoo)
        self.rates = RatesService(settings, self.gateway, self.db, self.clock, self.treasury)
        self.options = OptionsService(
            settings, self.gateway, self.db, self.clock, self.market, self.rates, market_providers
        )
        self.fundamentals = FundamentalsService(
            settings, self.gateway, self.db, self.clock, self.sec, self.fmp, self.yahoo
        )
        self.valuation = ValuationService(settings, self.clock, self.market, self.rates, self.fundamentals)
        self.portfolio = PortfolioService(settings, self.db, self.clock, self.market, self.rates)
        self.vehicle = VehicleService(settings, self.gateway, self.db, self.clock, self.eia, self.fueleconomy)
        self.sports = SportsService(settings, self.gateway, self.db, self.clock, self.espn, self.odds)
        self.picks = PicksService(settings, self.clock, self.market)
        self.sandbox = SandboxService(settings, self.db, self.clock, self.market, self.rates)
        self.reference = ReferenceService(
            settings, self.gateway, self.db, self.clock, self.market, self.sec, self.wikipedia, self.fmp
        )
        self.facts = FactsService(settings, self.db, self.clock, self.sec)
        self.model = ModelService(
            settings, self.clock, self.cache, self.market, self.rates, self.reference, self.facts, self.jobs
        )
        self.forecast = ForecastService(
            settings, self.clock, self.cache, self.market, self.rates, self.options
        )
        self.forecast.model = self.model
        self.forecast.reference = self.reference
        self.sandbox.model = self.model
        self.picks.model = self.model
        self.picks.forecast = self.forecast
        self.predictions = PredictionService(
            settings, self.db, self.clock, self.market, self.forecast, self.model
        )
        self.backfill = BackfillService(
            settings, self.db, self.clock, self.market, self.rates, self.forecast, self.model, self.jobs
        )
        self.stocks = StockReportService(
            settings, self.clock, self.market, self.forecast, self.model, self.valuation, self.predictions
        )
        self.notifier = EmailNotifier(settings)
        # Alpaca paper trading: the broker is always built with paper=True (see providers/alpaca_trading.py).
        self.broker = broker or AlpacaPaperBroker(
            _secret(settings.alpaca_api_key_id),
            _secret(settings.alpaca_api_secret_key),
            timeout=settings.http_timeout_seconds,
        )
        self.trading = TradingService(
            settings,
            self.db,
            self.clock,
            self.broker,
            TradingDataLoader(settings, self.clock, self.market, self.model, self.options, self.reference),
            self.jobs,
        )

        from quantpulse.workers.poller import Poller  # local import avoids a cycle

        self.poller = Poller(self)
        self.started_at = self.clock.now()

    async def startup(self) -> None:
        if self.settings.auto_migrate:
            await asyncio.to_thread(migrate.upgrade, self.settings.database_url)
        if self.settings.polling_enabled:
            self.poller.start()

    async def shutdown(self) -> None:
        await self.poller.stop()
        await self.jobs.shutdown()
        await self.http.aclose()
        await self.db.dispose()

    async def status(self) -> dict[str, Any]:
        now = self.clock.now()
        try:
            revision = await asyncio.to_thread(migrate.current_revision, self.settings.database_url)
            db_ok = True
        except Exception as exc:  # pragma: no cover - reported, not raised
            revision, db_ok = f"error: {exc}", False
        s = self.settings
        return {
            "version": __version__,
            "environment": s.environment,
            "started_at": self.started_at.isoformat(),
            "now": now.isoformat(),
            "market_session": session_at(now).value,
            "next_market_open": next_open(now).isoformat(),
            "live_data_enabled": s.enable_live_data,
            "market_provider_order": list(s.market_providers),
            "credentials": {
                "polygon": s.has_credentials("polygon"),
                "alpaca": s.has_credentials("alpaca"),
                "fmp": s.has_credentials("fmp"),
                "eia": s.has_credentials("eia"),
                "odds_api": s.has_credentials("odds_api"),
                "alpaca_paper_trading": self.broker.configured(),
                "smtp": s.smtp_configured,
                "sec_user_agent_customised": "set QP_SEC_USER_AGENT" not in s.sec_user_agent,
            },
            "database": {"ok": db_ok, "revision": revision, "head": migrate.head_revision()},
            "cache": self.cache.snapshot(),
            "gateway": self.gateway.snapshot(),
            "rate_limiters": {n: b.snapshot() for n, b in self.http.limiters.items()},
            "odds_api_quota": {"remaining": self.odds.requests_remaining, "used": self.odds.requests_used},
            "poller": self.poller.snapshot(),
            "trading": {
                "paper_only": True,
                "endpoint": self.broker.base_url,
                "enabled": s.alpaca_trading_enabled,
                "dry_run": s.trading_dry_run,
                "can_submit": s.trading_can_submit,
            },
        }
