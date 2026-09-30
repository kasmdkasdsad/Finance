"""Composition root: builds providers, the gateway, the database and all services from settings."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, cast

import httpx

from quantpulse import __version__
from quantpulse.config import Settings
from quantpulse.core import runtime
from quantpulse.core.cache import TTLCache
from quantpulse.core.clock import Clock, SystemClock
from quantpulse.core.errors import ProviderNoData
from quantpulse.core.gateway import DataGateway
from quantpulse.core.http import HttpClient
from quantpulse.core.jobs import JobRegistry
from quantpulse.core.market_calendar import next_open, session_at
from quantpulse.core.rate_limit import TokenBucket
from quantpulse.db import migrate
from quantpulse.db.session import Database, PoolSettings
from quantpulse.logging_config import log_event
from quantpulse.options.data import OptionsMarketDataProvider
from quantpulse.providers.alpaca import Alpaca
from quantpulse.providers.alpaca_options import AlpacaOptionsProvider
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
from quantpulse.services.alerts import AlertService
from quantpulse.services.backfill import BackfillService
from quantpulse.services.facts import FactsService
from quantpulse.services.forecast import ForecastService
from quantpulse.services.fundamentals import FundamentalsService
from quantpulse.services.health import HealthMonitor
from quantpulse.services.lease import Lease
from quantpulse.services.market import MarketService
from quantpulse.services.model import ModelService
from quantpulse.services.notifications import EmailNotifier
from quantpulse.services.options import OptionsService
from quantpulse.services.options_lab import OptionsLabService
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
LIFECYCLE_KEY = "app_lifecycle"
LIFECYCLE_KEEP = 10
FINAL_RECONCILE_SECONDS = 15.0


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
        options_data: OptionsMarketDataProvider | None = None,
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
        self.db = Database(
            settings.database_url,
            pool=PoolSettings(
                size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                connect_timeout=settings.db_connect_timeout_seconds,
                command_timeout=settings.db_command_timeout_seconds,
            ),
        )
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
        # at most one process supervises the Brain and sends orders (a database lease)
        # production: the database server's clock decides the lease (instances' own clocks may disagree);
        # tests step a fake clock through expiries
        self.lease = Lease(
            self.db,
            self.clock,
            ttl=timedelta(seconds=settings.brain_lease_seconds),
            db_time=isinstance(self.clock, SystemClock),
        )
        trading_data = TradingDataLoader(
            settings, self.clock, self.market, self.model, self.options, self.reference
        )

        async def underlying_quote(symbol: str) -> tuple[float, datetime | None]:
            q = (await trading_data.live_quotes([symbol], consolidated=False)).get(symbol.upper())
            if q is None:
                raise ProviderNoData("alpaca_options", f"no live price for {symbol}")
            return q.price, q.timestamp

        # option chains and quotes: Alpaca's options data (the free indicative feed unless QP_ALPACA_OPTIONS_FEED
        # says opra), read with the paper keys; contracts come from the paper trading API
        self.options_data = options_data or AlpacaOptionsProvider(
            self.http,
            _secret(settings.alpaca_api_key_id),
            _secret(settings.alpaca_api_secret_key),
            feed=settings.alpaca_options_feed,
            underlying_quote=underlying_quote,
            clock=self.clock.now,
        )
        self.trading = TradingService(
            settings,
            self.db,
            self.clock,
            self.broker,
            trading_data,
            self.jobs,
            lease=self.lease,
            options_data=self.options_data if settings.options_enabled else None,
        )
        # the options research lab (background research on real prices, model-priced chains; never an order)
        self.options_lab = OptionsLabService(
            settings, self.db, self.clock, self.market, self.jobs, self.reference
        )
        # The brain: specialist agents over a read-only view of the paper account. It proposes; the trading
        # service's deterministic risk engine is the only path to an order.
        from quantpulse.brain.service import BrainService  # local import keeps the brain optional at import

        self.brain = BrainService(
            settings,
            self.clock,
            self.db,
            self.jobs,
            self.broker,
            self.trading,
            TradingDataLoader(settings, self.clock, self.market, self.model, self.options, self.reference),
            self.reference,
            self.model,
            self.options,
            self.market,
        )

        # the market evolution monitor and the versioned model registry (research; never an order)
        from quantpulse.services.evolution import EvolutionService
        from quantpulse.services.model_registry import ModelRegistryService

        self.evolution = EvolutionService(settings, self.db, self.clock, self.market, self.jobs, self.options_lab)
        self.registry = ModelRegistryService(self.db, self.clock)
        # the Options Brain: inside the Brain's cycle (shadow always; paper orders only through the trading service)
        from quantpulse.brain.options.brain import OptionsBrain

        self.options_brain = OptionsBrain(
            settings, self.clock, self.db, self.options_data, self.market, self.options_lab, self.reference
        )
        if settings.options_enabled:
            self.brain.orchestrator.options = self.options_brain

        # cloud monitoring: alerts (ntfy / webhook / heartbeat, all optional) and the health monitor, whose
        # order-critical checks (database, Alpaca, reconciliation) fail Brain orders closed at the last gate
        self.alerts = AlertService(settings, self.clock, self.db, self.http.raw)
        self.health = HealthMonitor(self)
        self.trading.health_gate = self.health.order_blockers

        from quantpulse.workers.poller import Poller  # local import avoids a cycle

        self.poller = Poller(self)
        self.started_at = self.clock.now()

    async def startup(self) -> None:
        rt = runtime.current()
        log_event(logger, "app.startup", f"QuantPulse {rt.version} starting ({self.settings.deployment})",
                  commit=rt.short_commit, branch=rt.branch, instance=rt.instance, platform=rt.platform,
                  deployment=self.settings.deployment, brain_mode=self.settings.brain_mode)  # fmt: skip
        await self._wait_for_database()
        if self.settings.auto_migrate:
            await asyncio.to_thread(migrate.upgrade, self.settings.database_url)
        else:  # migrations ran once per deploy (the pre-deploy command): the schema must be at the head
            current = await asyncio.to_thread(migrate.current_revision, self.settings.database_url)
            head = migrate.head_revision()
            if current != head:
                raise RuntimeError(
                    f"the database schema is at {current}, this version needs {head}: run quantpulse-migrate "
                    "(the pre-deploy command) — nothing was started"
                )
        await self._record_lifecycle(started_at=self.clock.now().isoformat(), stopped_at=None, clean=None)
        if self.settings.polling_enabled:
            self.poller.start()

    async def _wait_for_database(self) -> None:
        """The database must answer before anything starts; a short outage (a restart for maintenance) is
        waited out with back-off for up to ``QP_DB_STARTUP_WAIT_SECONDS``, then start-up fails (and the host
        restarts the process). Nothing trades meanwhile: nothing has started."""
        from sqlalchemy import text

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.db_startup_wait_seconds
        delay, attempt = 1.0, 0
        while True:
            attempt += 1
            try:
                async with self.db.session() as s:
                    await asyncio.wait_for(
                        s.execute(text("SELECT 1")), self.settings.db_connect_timeout_seconds
                    )
                if attempt > 1:
                    log_event(logger, "app.database.ready", f"the database answered after {attempt} attempts")
                return
            except Exception as exc:
                left = deadline - loop.time()
                if left <= 0:
                    raise RuntimeError(
                        f"the database did not answer within {self.settings.db_startup_wait_seconds:.0f} s "
                        f"({type(exc).__name__}): nothing was started"
                    ) from exc
                log_event(logger, "app.database.waiting", "the database is not answering yet: retrying",
                          level=logging.WARNING, attempt=attempt, error=type(exc).__name__,
                          retry_in_seconds=round(min(delay, left), 1))  # fmt: skip
                await asyncio.sleep(min(delay, left))
                delay = min(delay * 2, 15.0)

    async def _record_lifecycle(self, **fields: Any) -> None:
        """Each process's start and stop, in the database (``app_lifecycle``): a stop without a record is a
        crash, which the next leader also sees from the lease it had to wait for."""
        try:
            now = self.clock.now()
            store = self.brain.store
            record = await store.get_state(LIFECYCLE_KEY) or {}
            rt = runtime.current()
            mine = record.get(self.lease.holder) or {"instance": rt.instance, "commit": rt.short_commit}
            record[self.lease.holder] = {**mine, **fields}
            keep = sorted(record.items(), key=lambda kv: str(kv[1].get("started_at") or ""))[-LIFECYCLE_KEEP:]
            await store.set_state(LIFECYCLE_KEY, dict(keep), now)
        except Exception:  # bookkeeping only: never stops a start or a shutdown
            logger.warning("recording the process lifecycle failed", exc_info=True)

    async def shutdown(self) -> None:
        """A deploy or restart (SIGTERM). In order: no new Brain tick and no new order (an order already being
        sent completes: it was recorded before it left); the running tick or cycle may finish for up to
        ``QP_SHUTDOWN_DRAIN_SECONDS``, anything still running is then cancelled safely (orders are recorded
        before they are sent, and the next supervisor reconciles first); the leader reconciles once more so the
        record ends in step with Alpaca; the shutdown is recorded; the lease is handed over; network clients
        and the database pool are closed."""
        drain = self.settings.shutdown_drain_seconds
        self.trading.stopping = True  # the last gate refuses new orders from now on
        self.brain.supervisor.begin_stop()
        log_event(logger, "app.shutdown", "QuantPulse shutting down: no new orders; draining the Brain",
                  drain_seconds=drain)  # fmt: skip
        finished, still = True, cast(tuple[str, ...], ())
        try:
            finished = await self.brain.supervisor.drain(drain)
            still = await self.jobs.drain({"brain", "trading"}, drain if finished else 0.0)
            if still or not finished:
                log_event(logger, "app.shutdown.cancel", "work still running at shutdown is cancelled",
                          level=logging.WARNING, jobs=",".join(still), tick_finished=finished)  # fmt: skip
        except Exception:
            logger.warning("draining the Brain at shutdown failed", exc_info=True)
        await self.poller.stop()
        reconciled = await self._final_reconcile()
        await self._record_lifecycle(stopped_at=self.clock.now().isoformat(), clean=True,
                                     drained=finished and not still, reconciled=reconciled)  # fmt: skip
        try:  # hand the Brain over at once instead of after the lease lapses
            await self.lease.release()
        except Exception:  # the database may already be gone: the lease then lapses on its own
            logger.warning("releasing the Brain lease at shutdown failed; it lapses by itself")
        await self.jobs.shutdown()
        await self.http.aclose()
        await self.db.dispose()
        log_event(logger, "app.stopped", "QuantPulse stopped cleanly", reconciled=reconciled)

    async def _final_reconcile(self) -> str:
        """The leader brings the order record in step with Alpaca before handing over (bounded: an Alpaca
        outage never holds up a shutdown — the next supervisor reconciles first anyway)."""
        s = self.settings
        if not (s.brain_owns_account and self.broker.configured()):
            return "not applicable"
        try:
            if not await self.lease.held():
                return "skipped: not the leader"
            await asyncio.wait_for(self.trading.reconcile("shutdown"), FINAL_RECONCILE_SECONDS)
            return "done"
        except Exception as exc:
            logger.warning("the reconciliation at shutdown failed (%s): the next supervisor reconciles first",
                           type(exc).__name__)  # fmt: skip
            return f"failed: {type(exc).__name__}"

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
