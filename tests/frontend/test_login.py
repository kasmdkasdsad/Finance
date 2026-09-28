"""The dashboard login: in the cloud nothing renders — no page, no data, no control — before the password."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import auth
from quantpulse.core.passwords import hash_password

APP = str(Path(__file__).resolve().parents[2] / "frontend" / "app.py")
PASSWORD = "a long dashboard password"
TOKEN = "dashboard-test-api-token-0123456789abcdef"


@pytest.fixture(autouse=True)
def cloud_env(monkeypatch):
    auth._guard.clear()
    monkeypatch.setenv("QP_DEPLOYMENT", "cloud")
    monkeypatch.setenv("QP_API_URL", "http://127.0.0.1:9")  # nothing listens: pages show their error
    monkeypatch.setenv("QP_API_TOKEN", TOKEN)
    monkeypatch.setenv("QP_DASHBOARD_PASSWORD_HASH", hash_password(PASSWORD, iterations=200_000))
    yield
    auth._guard.clear()


def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def sign_in(at: AppTest, password: str) -> AppTest:
    at.text_input[0].input(password)
    at.button[0].click()
    at.run()
    return at


def rendered(at: AppTest) -> str:
    return " ".join(str(getattr(e, "value", "")) for e in (*at.markdown, *at.caption, *at.error, *at.title))


def test_nothing_renders_before_the_password():
    at = app()
    assert not at.exception
    assert [t.label for t in at.text_input] == ["Password"]
    assert len(at.sidebar.button) == 0 and len(at.sidebar.text_input) == 0  # no sidebar, no controls
    assert "STOP BRAIN ORDERS" not in rendered(at) and TOKEN not in rendered(at)


def test_a_wrong_password_is_refused_and_the_right_one_opens_the_dashboard():
    at = sign_in(app(), "not the password")
    assert any("Wrong password" in e.value for e in at.error)
    assert [t.label for t in at.text_input] == ["Password"]
    at = sign_in(at, PASSWORD)
    assert not at.exception
    assert "Password" not in [t.label for t in at.text_input]
    assert any(b.label == "Sign out" for b in at.sidebar.button)
    # in the cloud the API address and token are the server's: no field to change or see them
    assert all(t.label not in ("API URL", "API token") for t in at.sidebar.text_input)
    assert TOKEN not in rendered(at)
    next(b for b in at.sidebar.button if b.label == "Sign out").click()
    at.run()
    assert [t.label for t in at.text_input] == ["Password"]


def test_repeated_wrong_passwords_lock_the_login():
    at = app()
    for _ in range(auth.FREE_ATTEMPTS):
        at = sign_in(at, "guess")
    at = sign_in(at, PASSWORD)  # even the right password waits out the lock
    assert any("Too many wrong passwords" in e.value for e in at.error)
    assert [t.label for t in at.text_input] == ["Password"]


def test_the_cloud_dashboard_stays_locked_without_a_password_hash(monkeypatch):
    monkeypatch.delenv("QP_DASHBOARD_PASSWORD_HASH")
    at = app()
    assert any("dashboard is locked" in e.value for e in at.error)
    assert len(at.text_input) == 0 and len(at.sidebar.button) == 0


def test_guard_lockout_doubles_and_resets():
    hashed = hash_password(PASSWORD, iterations=200_000)
    g = auth.Guard()
    for _ in range(auth.FREE_ATTEMPTS):
        assert g.attempt("x", hashed, now=0.0) is False
    assert g.locked_for(0.0) == auth.FIRST_LOCKOUT_SECONDS
    assert g.attempt(PASSWORD, hashed, now=1.0) is None  # locked: not even checked
    assert g.attempt("x", hashed, now=61.0) is False and g.locked_for(61.0) == 2 * auth.FIRST_LOCKOUT_SECONDS
    assert g.attempt(PASSWORD, hashed, now=500.0) is True and g.failures == 0
