"""Request-scoped dependencies: the service container and API-key authentication."""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, Request, WebSocket, status
from starlette.requests import HTTPConnection

from quantpulse.services.container import Container

LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def get_container(request: Request) -> Container:
    return request.app.state.container


def _token_ok(container: Container, supplied: str | None) -> bool:
    expected = container.settings.api_token
    if expected is None:
        # locally the API may run without a token; in the cloud never (the preflight requires one, and this
        # refuses every request should one ever be missing)
        return container.settings.deployment != "cloud"
    return supplied is not None and hmac.compare_digest(supplied, expected.get_secret_value())


async def require_api_key(
    connection: HTTPConnection, x_api_key: str | None = Header(default=None, alias="X-API-Key")
) -> None:
    if connection.scope["type"] == "websocket":
        return  # WebSocket routes authenticate themselves (headers or ?api_key=) before accepting
    if not _token_ok(connection.app.state.container, x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing or invalid X-API-Key")


def websocket_authorized(websocket: WebSocket) -> bool:
    container: Container = websocket.app.state.container
    supplied = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    return _token_ok(container, supplied)


def local_only_allowed(connection: HTTPConnection, what: str) -> None:
    """Without ``QP_API_TOKEN``, ``what`` (orders, Brain controls) is accepted only from this machine; in
    the cloud never without the token — a reverse proxy on the same host would look like this machine."""
    container: Container = connection.app.state.container
    if container.settings.api_token is not None:
        return  # the token was verified for this request by require_api_key
    host = connection.client.host if connection.client else None
    if container.settings.deployment == "cloud" or host not in LOOPBACK:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{what} accept remote requests only when QP_API_TOKEN is set",
        )


ContainerDep = Depends(get_container)
