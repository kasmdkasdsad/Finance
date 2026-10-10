"""What the read-only key reaches: GET requests to the monitoring pages, and nothing else."""

import pytest

from quantpulse.readonly import readable


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/api/v1/system/status",
        "/api/v1/system/watchdog",
        "/api/v1/market/session",
        "/api/v1/trading/account",
        "/api/v1/trading/positions",
        "/api/v1/trading/cycles/12",
        "/api/v1/brain/status",
        "/api/v1/brain/decisions/7/audit",
        "/api/v1/brain/lab/strategies/momentum/2",
        "/api/v1/brain/research/jobs/3",
        "/api/v1/options/positions",
        "/api/v1/options/strategies/4",
        "/api/v1/evolution/changes",
        "/api/v1/registry/models",
        "/api/v1/predictions",
        "/api/v1/predictions/scorecard",
        "/api/v1/jobs/abc-123",
    ],
)
def test_the_monitoring_pages_are_readable(path):
    assert readable("GET", path)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "get"])
def test_only_get_reads(method):
    assert not readable(method, "/api/v1/brain/status")
    assert not readable(method, "/api/v1/brain/kill-switch")


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/forecast/AAPL",  # heavy or third-party market data
        "/api/v1/stocks/AAPL/report",
        "/api/v1/model/report",
        "/api/v1/picks/daily",
        "/api/v1/market/quote/AAPL",
        "/api/v1/market/stream",  # the quote streams
        "/api/v1/market/ws",
        "/api/v1/market/regime",
        "/api/v1/options/AAPL/chain",
        "/api/v1/options/status/chain",  # a page name used as a symbol
        "/api/v1/options/chains",
        "/api/v1/sandbox/accounts",
        "/api/v1/predictionsx",  # a look-alike prefix
        "/api/v1/brainx/status",
        "/api/v1/brain",
        "/api/v1/brain/../trading/test-order",  # no climbing out of a section
        "/api/v1/brain/./status",
        "/api/v1/brain/.hidden",
        "/api/v1/brain//status",
        "/api/v1/brain/a/b/c/d/e",
        "/api/v1/brain/status/",
        "/api/v1/brain/st%61tus",
        "/docs",
        "/openapi.json",
        "",
    ],
)
def test_everything_else_is_not(path):
    assert not readable("GET", path)
