"""FastAPI application factory."""

from __future__ import annotations

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
    fundamentals,
    market,
    options,
    picks,
    portfolio,
    rates,
    sandbox,
    sports,
    system,
    vehicle,
)
from quantpulse.config import Settings, get_settings
from quantpulse.logging_config import configure_logging
from quantpulse.services.container import Container

API_PREFIX = "/api/v1"
DESCRIPTION = """
**QuantPulse Terminal** — quantitative finance & executive intelligence API.

Every data payload carries provenance (`meta.status` = `live` | `cached` | `stale` | `synthetic`) so clients
can always tell real-time data from fallbacks. Analytics that combine several feeds return a composite
`meta` with one entry per source.
"""


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    for module in (system, market, rates, options, fundamentals, portfolio, vehicle, sports, picks, sandbox):
        app.include_router(module.router, prefix=API_PREFIX, dependencies=auth)
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory quantpulse.api.app:app_factory``."""
    return create_app()
