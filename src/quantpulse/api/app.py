"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from quantpulse import __version__
from quantpulse.api.deps import require_api_key
from quantpulse.api.errors import install_error_handlers
from quantpulse.api.middleware import RequestContextMiddleware
from quantpulse.api.routers import (
    brain,
    forecast,
    fundamentals,
    jobs,
    market,
    model,
    options,
    picks,
    portfolio,
    predictions,
    rates,
    sandbox,
    sports,
    stocks,
    system,
    trading,
    vehicle,
)
from quantpulse.config import Settings, get_settings
from quantpulse.logging_config import configure_logging, settings_secrets
from quantpulse.services import preflight
from quantpulse.services.container import Container

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
DESCRIPTION = """
**QuantPulse Terminal** — quantitative finance & executive intelligence API.

**Alpaca paper trading** (`/trading`) runs against Alpaca's *paper* API only — simulated money. There is no
live-money path.

Every data payload carries provenance (`meta.status` = `live` | `cached` | `stale` | `synthetic`) so clients
can always tell real-time data from fallbacks. Analytics that combine several feeds return a composite
`meta` with one entry per source.
"""


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    configure_logging(
        settings.log_level, settings.log_json, secrets=settings_secrets(settings), log_dir=settings.log_dir
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # In the cloud nothing starts (no database, supervisor, poller or order) unless the preflight passes.
        try:
            report = preflight.enforce(settings)
        except preflight.PreflightFailed as exc:
            for line in exc.report.lines():
                logger.critical(line)
            raise
        if report is not None:
            logger.info("cloud preflight passed: %d checks", len(report.checks))
        c = container or Container(settings)
        app.state.container = c
        await c.startup()
        try:
            yield
        finally:
            await c.shutdown()

    app = FastAPI(
        title="QuantPulse Terminal API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": t}
            for t in (
                "system",
                "market",
                "rates",
                "options",
                "fundamentals",
                "portfolio",
                "vehicle",
                "sports",
                "picks",
                "sandbox",
                "forecast",
                "model",
                "stocks",
                "predictions",
                "trading",
            )
        ],
    )
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    app.include_router(system.public)
    auth = [Depends(require_api_key)]
    modules = (
        system,
        market,
        model,  # before any catch-all market paths, for /market/regime
        rates,
        options,
        fundamentals,
        portfolio,
        vehicle,
        sports,
        picks,
        sandbox,
        forecast,
        stocks,
        predictions,
        trading,
        brain,
        jobs,
    )
    for module in modules:
        app.include_router(module.router, prefix=API_PREFIX, dependencies=auth)
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory quantpulse.api.app:app_factory``."""
    return create_app()
