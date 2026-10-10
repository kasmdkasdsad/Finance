"""What the read-only key may reach: GET requests to the monitoring pages, nothing else.

The read-only key (``QP_API_READ_TOKEN``) lets someone watch the system without the power to change it. One
rule, shared by the API and the reader gateway (``quantpulse.reader``), decides what it reaches:

* the method must be GET: every order, control, switch, setting and promotion is a POST, PUT, PATCH or DELETE;
* the path must be one of the monitoring pages below: the system's health, the trading account, the Brain, its
  research, the options book, market changes and the prediction record. The heavy or third-party market-data
  pages (forecasts, stock reports, chains, quotes) and the quote streams are left out, so a reader can neither
  slow the supervisor nor use up the market-data allowance.

None of these GET pages places, cancels or changes anything (a test calls every one of them with the key and
checks that nothing acted). Path segments may not start with a dot, so ``..`` cannot climb out of a section.
Standard library only: the reader gateway imports this without the rest of QuantPulse.
"""

from __future__ import annotations

import re

_SEG = r"[A-Za-z0-9_-][A-Za-z0-9._-]*"  # one path segment, never "." or ".."

READABLE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.ASCII)
    for p in (
        r"/health",
        r"/api/v1/system/(?:status|health|alerts|watchdog|ingestions)",
        r"/api/v1/market/session",
        rf"/api/v1/trading(?:/{_SEG}){{1,2}}",
        rf"/api/v1/brain(?:/{_SEG}){{1,4}}",
        r"/api/v1/options/(?:status|candidates|research|experiments|learning|ml|portfolio|positions|greeks"
        rf"|performance|counterfactuals|missed-opportunities|strategies(?:/{_SEG})?)",
        rf"/api/v1/(?:evolution|registry)(?:/{_SEG}){{1,2}}",
        r"/api/v1/predictions(?:/scorecard|/backfill)?",
        rf"/api/v1/jobs(?:/{_SEG})?",
    )
)


def readable(method: str, path: str) -> bool:
    """True when the read-only key may make this request: a GET to one of the monitoring pages."""
    return method == "GET" and any(p.fullmatch(path) for p in READABLE)
