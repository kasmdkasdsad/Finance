"""Secrets never reach a log line: configured values, credential-shaped text, tracebacks, uvicorn's access log."""

import io
import logging

import pytest

from quantpulse.config import Settings
from quantpulse.logging_config import (
    LOG_FILE,
    RedactingFilter,
    Redactor,
    configure_logging,
    settings_secrets,
)

KEY = "PKtest-log-key-00001"
SECRET = "log-test-secret-value-00000000000001"
TOKEN = "log-test-api-token-0123456789abcdefghijkl"
DB_PASSWORD = "log-db-password-xyz"


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    access = logging.getLogger("uvicorn.access")
    access_handlers, access_propagate = access.handlers[:], access.propagate
    yield
    root.handlers[:], root.level = handlers, level
    access.handlers[:], access.propagate = access_handlers, access_propagate


def settings() -> Settings:
    return Settings(
        _env_file=None,
        alpaca_api_key_id=KEY,
        alpaca_api_secret_key=SECRET,
        api_token=TOKEN,
        database_url=f"postgresql+asyncpg://qp:{DB_PASSWORD}@db:5432/quantpulse",
    )


def test_settings_secrets_include_every_secret_and_the_database_password():
    found = settings_secrets(settings())
    assert {KEY, SECRET, TOKEN, DB_PASSWORD} <= set(found)


@pytest.mark.parametrize(
    ("text", "hidden"),
    [
        (f"connecting to postgresql+asyncpg://qp:{DB_PASSWORD}@db/quantpulse", DB_PASSWORD),
        ("GET /v1/quotes?apiKey=abc123def456&symbol=SPY", "abc123def456"),
        ("GET /api/v1/market/ws?api_key=somesecretvalue", "somesecretvalue"),
        ("headers {'APCA-API-SECRET-KEY': 'hdrsecretvalue1'}", "hdrsecretvalue1"),
        ("X-API-Key: headertokenvalue", "headertokenvalue"),
        ("Authorization: Bearer bearertokenvalue", "bearertokenvalue"),
        ("smtp password=hunter2hunter2", "hunter2hunter2"),
    ],
)
def test_credential_shaped_text_is_masked_whatever_the_value(text, hidden):
    out = Redactor()(text)
    assert hidden not in out and "***" in out


def test_configured_secrets_are_masked_anywhere():
    r = Redactor(settings_secrets(settings()))
    line = f"key {KEY} secret {SECRET} token {TOKEN} in the middle of a sentence"
    out = r(line)
    assert all(s not in out for s in (KEY, SECRET, TOKEN)) and out.count("***") == 3


def test_every_handler_writes_masked_lines_including_tracebacks_and_files(tmp_path, restore_logging):
    configure_logging("INFO", secrets=settings_secrets(settings()), log_dir=tmp_path / "logs")
    stream = io.StringIO()
    extra = logging.StreamHandler(stream)
    extra.addFilter(logging.getLogger().handlers[0].filters[0])
    logging.getLogger().addHandler(extra)
    log = logging.getLogger("quantpulse.test")
    log.warning("the key is %s and the secret %s", KEY, SECRET)
    try:
        raise RuntimeError(f"failed with token {TOKEN}")
    except RuntimeError:
        log.exception("boom at postgresql+asyncpg://qp:%s@db/quantpulse", DB_PASSWORD)
    for h in logging.getLogger().handlers:
        h.flush()
    written = (tmp_path / "logs" / LOG_FILE).read_text() + stream.getvalue()
    assert "RuntimeError" in written and "***" in written
    for secret in (KEY, SECRET, TOKEN, DB_PASSWORD):
        assert secret not in written


def test_json_lines_are_masked_too(tmp_path, restore_logging):
    configure_logging("INFO", json_logs=True, secrets=[SECRET], log_dir=tmp_path)
    logging.getLogger("quantpulse.test").error("secret=%s", SECRET)
    for h in logging.getLogger().handlers:
        h.flush()
    assert SECRET not in (tmp_path / LOG_FILE).read_text()


def test_uvicorn_access_lines_keep_their_fields_and_lose_query_secrets(restore_logging):
    from uvicorn.logging import AccessFormatter

    access = logging.getLogger("uvicorn.access")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s'))
    access.handlers[:] = [handler]
    access.propagate = False
    configure_logging("INFO", secrets=[TOKEN])
    access.info(
        '%s - "%s %s HTTP/%s" %d', "10.0.0.2:5000", "GET", f"/api/v1/market/ws?api_key={TOKEN}", "1.1", 101
    )
    line = stream.getvalue()
    assert "10.0.0.2:5000" in line and "101" in line and "/api/v1/market/ws" in line
    assert TOKEN not in line


def test_the_filter_never_drops_a_record():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "value %s", (SECRET,), None)
    assert RedactingFilter(Redactor([SECRET])).filter(record)
    assert record.getMessage() == "value ***"
