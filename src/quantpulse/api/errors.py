"""Error boundaries: every failure leaves the API as a typed JSON ``ErrorResponse``."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from quantpulse.core.errors import DomainError, NotFoundError, ProviderError
from quantpulse.schemas.common import ErrorResponse
from quantpulse.services.notifications import NotConfiguredError, NotificationError, SyntheticDataRefused

logger = logging.getLogger(__name__)


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _respond(request: Request, status_code: int, error: str, detail: object = None) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=_rid(request))
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return _respond(request, 422, "validation_error", errors)

    @app.exception_handler(ValidationError)
    async def _model_validation(request: Request, exc: ValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return _respond(request, 422, "validation_error", errors)

    @app.exception_handler(DomainError)
    async def _domain(request: Request, exc: DomainError) -> JSONResponse:
        return _respond(request, 422, "domain_error", str(exc))

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return _respond(request, 404, "not_found", str(exc))

    @app.exception_handler(SyntheticDataRefused)
    async def _synthetic(request: Request, exc: SyntheticDataRefused) -> JSONResponse:
        return _respond(request, 409, "synthetic_data", str(exc))

    @app.exception_handler(NotConfiguredError)
    async def _not_configured(request: Request, exc: NotConfiguredError) -> JSONResponse:
        return _respond(request, 503, "not_configured", str(exc))

    @app.exception_handler(NotificationError)
    async def _notification(request: Request, exc: NotificationError) -> JSONResponse:
        return _respond(request, 502, "delivery_failed", str(exc))

    @app.exception_handler(ProviderError)
    async def _provider(
        request: Request, exc: ProviderError
    ) -> JSONResponse:  # should be absorbed by the gateway
        logger.warning("provider error escaped the gateway: %s", exc)
        return _respond(request, 502, "upstream_error", str(exc))

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        return _respond(request, exc.status_code, "http_error", exc.detail)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _respond(request, 500, "internal_error", "an unexpected error occurred")
