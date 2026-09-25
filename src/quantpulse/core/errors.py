"""Error hierarchy used by providers, the gateway and the API layer."""

from __future__ import annotations


class QuantPulseError(Exception):
    """Base class for all application errors."""


class DomainError(QuantPulseError):
    """Invalid input or an impossible calculation (maps to HTTP 422)."""


class NotFoundError(QuantPulseError):
    """A requested entity does not exist (maps to HTTP 404)."""


class ProviderError(QuantPulseError):
    """A live data provider failed. Always recoverable via fallback."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.message = message


class ProviderNotConfigured(ProviderError):
    """Provider lacks credentials or is disabled."""


class ProviderHTTPError(ProviderError):
    def __init__(self, provider: str, status_code: int, message: str) -> None:
        super().__init__(provider, f"HTTP {status_code}: {message}")
        self.status_code = status_code


class ProviderRateLimited(ProviderError):
    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        detail = f"rate limited (retry after {retry_after:.0f}s)" if retry_after else "rate limited"
        super().__init__(provider, detail)
        self.retry_after = retry_after


class ProviderParseError(ProviderError):
    """Payload did not match the expected schema."""


class ProviderNoData(ProviderError):
    """Provider answered successfully but had no data for the request."""


class CircuitOpenError(ProviderError):
    def __init__(self, provider: str, retry_in: float) -> None:
        super().__init__(provider, f"circuit open (retry in {retry_in:.0f}s)")
        self.retry_in = retry_in
