"""Thin synchronous client for the QuantPulse API used by the Streamlit frontend."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = "http://127.0.0.1:8000"


class ApiError(Exception):
    def __init__(self, status: int, error: str, detail: Any = None) -> None:
        super().__init__(f"{status} {error}: {detail}")
        self.status = status
        self.error = error
        self.detail = detail

    @property
    def message(self) -> str:
        if isinstance(self.detail, list):
            parts = []
            for item in self.detail:
                loc = ".".join(str(p) for p in item.get("loc", []) if p not in ("body", "query", "path"))
                parts.append(f"{loc}: {item.get('msg')}" if loc else str(item.get("msg")))
            return "; ".join(parts)
        return str(self.detail or self.error)


class ApiClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 90.0) -> None:
        self.base_url = (base_url or os.environ.get("QP_API_URL") or DEFAULT_URL).rstrip("/")
        token = token if token is not None else os.environ.get("QP_API_TOKEN")
        headers = {"X-API-Key": token} if token else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def _handle(self, response: httpx.Response) -> Any:
        if response.status_code == 204:
            return None
        try:
            body = response.json()
        except ValueError:
            body = {"error": "invalid_response", "detail": response.text[:300]}
        if response.status_code >= 400:
            if isinstance(body, dict):
                raise ApiError(response.status_code, str(body.get("error", "error")), body.get("detail"))
            raise ApiError(response.status_code, "error", body)
        return body

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, f"/api/v1{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(
                0, "unreachable", f"cannot reach the QuantPulse API at {self.base_url} ({type(exc).__name__})"
            ) from exc
        return self._handle(response)

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, payload: Any = None, **params: Any) -> Any:
        query = {k: v for k, v in params.items() if v is not None}
        return self.request("POST", path, json=payload, params=query or None)

    def put(self, path: str, payload: Any) -> Any:
        return self.request("PUT", path, json=payload)

    def patch(self, path: str, payload: Any) -> Any:
        return self.request("PATCH", path, json=payload)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)

    def health(self) -> dict[str, Any]:
        try:
            return self._handle(self._client.get("/health"))
        except httpx.HTTPError as exc:
            raise ApiError(0, "unreachable", f"cannot reach the QuantPulse API at {self.base_url}") from exc
