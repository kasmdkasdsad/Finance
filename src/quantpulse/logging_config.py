"""Logging setup: human-readable or JSON lines, with third-party noise and every secret suppressed.

Every line — ours, uvicorn's access and error logs, exception tracebacks — passes through a redacting filter
before it is written: the configured secrets (Alpaca key and secret, the API token, provider keys, the SMTP
password, the dashboard password hash, the database password) are replaced by ``***`` wherever they appear,
and so is anything shaped like a credential (a password in a URL, ``api_key=…``/``token=…`` parameters,
Alpaca's key headers, ``X-API-Key``/``Authorization`` values).

With ``QP_LOG_DIR`` set, lines are also written to ``<dir>/quantpulse.log``, rotated at 10 MB with ten old
files kept — in the cloud that folder is a volume, so logs survive restarts and redeployments.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MASK = "***"
LOG_FILE = "quantpulse.log"
MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 10
# Anything shaped like a credential, whatever its value.
PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(://[^/\s:@]+:)[^@\s/]+@"), r"\1" + MASK + "@"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|api[_-]?token|access[_-]?token|token|secret|secret[_-]?key|password"
            r"|passwd)=([^&\s\"']+)"
        ),
        r"\1=" + MASK,
    ),
    (
        re.compile(
            r"(?i)(apca-api-key-id|apca-api-secret-key|x-api-key|authorization)(['\"]?\s*[:=]\s*['\"]?)"
            r"(?:bearer\s+|basic\s+)?[^\s'\",}]+"
        ),
        r"\1\2" + MASK,
    ),
)


class Redactor:
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        # longest first, so a secret containing another is masked whole; very short values would mask noise
        self.values = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for value in self.values:
            if value in text:
                text = text.replace(value, MASK)
        for pattern, replacement in PATTERNS:
            text = pattern.sub(replacement, text)
        return text


class RedactingFilter(logging.Filter):
    """Rewrites each record's message (and traceback) with the secrets masked, before any handler writes it."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self.redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.access" and isinstance(record.args, tuple):
            # uvicorn's access formatter needs its arguments (client, method, path, version, status)
            record.args = tuple(self.redactor(a) if isinstance(a, str) else a for a in record.args)
        else:
            try:
                message = record.getMessage()
            except Exception:  # a malformed record: keep the raw template
                message = str(record.msg)
            record.msg, record.args = self.redactor(message), ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redactor(record.exc_text)
        if record.stack_info:
            record.stack_info = self.redactor(record.stack_info)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info or record.exc_text:
            payload["exc"] = record.exc_text or self.formatException(record.exc_info)  # type: ignore[arg-type]
        return json.dumps(payload, default=str)


def settings_secrets(settings: Any) -> list[str]:
    """Every secret value in ``settings`` (SecretStr fields) and the database password."""
    from pydantic import SecretStr

    out = [
        value.get_secret_value()
        for name in type(settings).model_fields
        if isinstance(value := getattr(settings, name, None), SecretStr)
    ]
    try:
        password = urlsplit(str(getattr(settings, "database_url", "") or "")).password
    except ValueError:
        password = None
    return [*out, *([password] if password else [])]


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    *,
    secrets: Iterable[str] = (),
    log_dir: str | Path | None = None,
) -> None:
    formatter: logging.Formatter = (
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    redacting = RedactingFilter(Redactor(secrets))
    handler: logging.Handler = logging.StreamHandler(sys.stdout)
    handlers = [handler]
    if log_dir:
        folder = Path(log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                folder / LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
            )
        )
    for h in handlers:
        h.setFormatter(formatter)
        h.addFilter(redacting)
    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level.upper())
    # uvicorn's own loggers write through their own handlers (the access log carries full request paths):
    # the same filter, and the same file.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        log = logging.getLogger(name)
        for h in log.handlers:
            if not any(isinstance(f, RedactingFilter) for f in h.filters):
                h.addFilter(redacting)
            else:
                for f in h.filters:
                    if isinstance(f, RedactingFilter):
                        f.redactor = redacting.redactor
        if log_dir and log.handlers and not log.propagate:
            log.handlers = [
                h for h in log.handlers if not isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            log.handlers.append(handlers[-1])
    # httpx logs full request URLs at INFO, which would include API keys passed as query parameters.
    for noisy in ("httpx", "httpcore", "aiosqlite", "sqlalchemy.engine", "alembic.runtime.migration"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
