"""The dashboard login. In the cloud nothing renders — no page, no data, no control — before the password.

The server checks the password (``POST /auth/login``) and keeps the sign-in in a signed, HttpOnly cookie, so
closing the tab or the browser does not sign anyone out; wrong passwords lock the login; Sign out revokes.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from streamlit.testing.v1 import AppTest
from streamlit.web.server.app_discovery import discover_asgi_app

from frontend import auth
from quantpulse.core.passwords import hash_password
from tests.frontend.conftest import _free_port

ROOT = Path(__file__).resolve().parents[2]
PAGES = str(ROOT / "frontend" / "dashboard.py")
PASSWORD = "a long dashboard password"
TOKEN = "dashboard-test-api-token-0123456789abcdef"
HASH = hash_password(PASSWORD, iterations=200_000)


@pytest.fixture(autouse=True)
def cloud_env(monkeypatch):
    auth.GUARD.reset()
    auth._revoked.clear()
    monkeypatch.setenv("QP_DEPLOYMENT", "cloud")
    monkeypatch.setenv("QP_API_URL", "http://127.0.0.1:9")  # nothing listens: pages show their error
    monkeypatch.setenv("QP_API_TOKEN", TOKEN)
    monkeypatch.setenv("QP_DASHBOARD_PASSWORD_HASH", HASH)
    yield
    auth.GUARD.reset()
    auth._revoked.clear()


def app(cookie: str | None = None, monkeypatch=None, **query) -> AppTest:
    if monkeypatch is not None:
        monkeypatch.setattr(auth, "_session_cookie", lambda: cookie)
    at = AppTest.from_file(PAGES, default_timeout=60)
    at.query_params.update(query)
    at.run()
    return at


def html(at: AppTest) -> str:
    return "\n".join(e.proto.body for e in at.get("html"))


def rendered(at: AppTest) -> str:
    parts = (*at.markdown, *at.caption, *at.error, *at.title)
    return " ".join(str(getattr(e, "value", "")) for e in parts) + html(at)


def client() -> TestClient:
    return TestClient(Starlette(routes=auth.routes()), base_url="http://dashboard", follow_redirects=False)


def sign_in(c: TestClient, password: str, **form) -> httpx.Response:
    return c.post(auth.LOGIN_PATH, data={"password": password, **form})


# ------------------------------------------------------------------ the page
def test_nothing_renders_before_the_password(monkeypatch):
    at = app(None, monkeypatch)
    assert not at.exception
    body = html(at)
    assert 'action="/auth/login"' in body and 'name="password"' in body and "Keep me signed in" in body
    assert len(at.sidebar.button) == 0 and len(at.sidebar.text_input) == 0  # no sidebar, no controls
    assert not at.metric and not at.dataframe  # no page, no data
    assert "STOP BRAIN" not in rendered(at) and TOKEN not in rendered(at)


def test_a_signed_cookie_opens_the_dashboard_and_survives_a_new_tab(monkeypatch):
    token = auth.issue(HASH, time.time(), auth.REMEMBER_SECONDS)
    for _tab in range(2):  # a new tab (or a restarted browser) sends the same cookie: still signed in
        at = app(token, monkeypatch)
        assert not at.exception
        assert 'action="/auth/login"' not in html(at)
        assert any("/auth/logout" in e.proto.body for e in at.sidebar.get("html"))  # Sign out
        # in the cloud the API address and token are the server's: no field to change or see them
        assert all(t.label not in ("API URL", "API token") for t in at.sidebar.text_input)
        assert TOKEN not in rendered(at)


def test_the_cloud_navigation_is_six_pages_with_a_tab_bar_for_phones(monkeypatch):
    at = app(auth.issue(HASH, time.time(), 3600), monkeypatch)
    assert not at.exception
    assert list(at.session_state["qp_pages"]) == [
        "",
        "trading",
        "brain",
        "options-intelligence",
        "research",
        "system",
    ]
    tab_bar = [e.proto.label for e in at.get("page_link")][:5]
    assert tab_bar == ["Home", "Portfolio", "Brain", "Options", "Research"]


def test_locally_the_older_tools_are_still_listed(monkeypatch):
    monkeypatch.setenv("QP_DEPLOYMENT", "local")
    monkeypatch.delenv("QP_DASHBOARD_PASSWORD_HASH")
    at = app(None, monkeypatch)
    assert not at.exception
    assert {"overview", "stock", "picks", "model-lab", "sports"} <= set(at.session_state["qp_pages"])


@pytest.mark.parametrize(
    "case", ["forged", "expired", "other password", "revoked", "garbage", "empty", "not ascii", "odd digits"]
)
def test_forged_expired_or_revoked_cookies_are_refused(monkeypatch, case):
    now = time.time()
    token = auth.issue(HASH, now, 3600)
    bad = {
        "forged": token[:-2] + ("AA" if not token.endswith("AA") else "BB"),
        "expired": auth.issue(HASH, now - 7200, 3600),
        "other password": auth.issue(hash_password("another long password", iterations=200_000), now, 3600),
        "revoked": token,
        "garbage": "v1.99999999999.x.y",
        "empty": "",
        "not ascii": token[:-1] + "é",
        "odd digits": "v1.²³.x.y",
    }[case]
    if case == "revoked":
        auth.revoke(token, now)
    assert not auth.valid(bad, HASH, now)
    at = app(bad, monkeypatch)
    assert 'action="/auth/login"' in html(at) and not at.metric


def test_the_form_returns_to_the_page_asked_for(monkeypatch):
    monkeypatch.setattr(
        auth.st.context.__class__, "url", property(lambda _: "https://qp.example.ts.net/brain")
    )
    assert auth._next_path() == "/brain"
    monkeypatch.setattr(auth.st.context.__class__, "url", property(lambda _: None))
    assert auth._next_path() == "/"


def test_the_login_page_says_why(monkeypatch):
    assert "Wrong password." in html(app(None, monkeypatch, signin="wrong"))
    auth.GUARD.locked_until = time.time() + 60
    assert "Too many wrong passwords: try again in" in html(app(None, monkeypatch, signin="locked"))


def test_the_cloud_dashboard_stays_locked_without_a_password_hash(monkeypatch):
    monkeypatch.delenv("QP_DASHBOARD_PASSWORD_HASH")
    at = app(None, monkeypatch)
    assert any("dashboard is locked" in e.value for e in at.error)
    assert "/auth/login" not in html(at) and len(at.sidebar.button) == 0


# ------------------------------------------------------------------ the server routes
def test_the_server_checks_the_password_and_sets_an_httponly_cookie():
    c = client()
    wrong = sign_in(c, "not the password", next="/brain")
    assert wrong.status_code == 303 and wrong.headers["location"] == "/brain?signin=wrong"
    assert "set-cookie" not in wrong.headers
    ok = sign_in(c, PASSWORD, next="/brain", remember="1")
    assert ok.status_code == 303 and ok.headers["location"] == "/brain"
    cookie = ok.headers["set-cookie"]
    assert cookie.startswith(f"{auth.COOKIE}=") and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert f"Max-Age={auth.REMEMBER_SECONDS}" in cookie and "Path=/" in cookie
    assert PASSWORD not in cookie
    assert "Secure" not in cookie  # plain http (local): a Secure cookie would never come back
    token = ok.cookies[auth.COOKIE]
    assert auth.valid(token, HASH, time.time() + auth.REMEMBER_SECONDS - 60)
    assert not auth.valid(token, HASH, time.time() + auth.REMEMBER_SECONDS + 60)


def test_over_https_the_cookie_is_secure_and_without_remember_it_ends_with_the_browser():
    c = client()
    r = c.post(auth.LOGIN_PATH, data={"password": PASSWORD}, headers={"Origin": "https://qp.example.ts.net"})
    cookie = r.headers["set-cookie"]
    assert "Secure" in cookie and "Max-Age" not in cookie and "expires" not in cookie.lower()
    token = r.cookies[auth.COOKIE]
    assert not auth.valid(token, HASH, time.time() + auth.SESSION_SECONDS + 60)


@pytest.mark.parametrize(
    ("target", "lands"),
    [("/brain", "/brain"), ("//evil.example/x", "/"), ("https://evil.example/", "/"), ("\\\\evil", "/"),
     ("", "/"), ("/trading?x=1", "/trading")],
)  # fmt: skip
def test_the_redirect_never_leaves_the_dashboard(target, lands):
    r = sign_in(client(), PASSWORD, next=target)
    assert r.status_code == 303 and r.headers["location"] == lands


def test_cross_site_posts_are_refused():
    c = client()
    r = c.post(auth.LOGIN_PATH, data={"password": PASSWORD}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403 and "set-cookie" not in r.headers
    r = c.post(auth.LOGOUT_PATH, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_repeated_wrong_passwords_lock_the_login():
    c = client()
    for _ in range(auth.FREE_ATTEMPTS):
        assert sign_in(c, "guess").headers["location"] == "/?signin=wrong"
    r = sign_in(c, PASSWORD)  # even the right password waits out the lock
    assert r.headers["location"] == "/?signin=locked" and "set-cookie" not in r.headers


def test_sign_out_deletes_and_revokes_the_cookie():
    c = client()
    token = sign_in(c, PASSWORD, remember="1").cookies[auth.COOKIE]
    assert auth.valid(token, HASH, time.time())
    c.cookies.set(auth.COOKIE, token)
    r = c.post(auth.LOGOUT_PATH)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert f'{auth.COOKIE}=""' in r.headers["set-cookie"] and "Max-Age=0" in r.headers["set-cookie"]
    assert not auth.valid(token, HASH, time.time())  # a copy of the cookie no longer works either


def test_changing_the_password_signs_every_browser_out():
    token = auth.issue(HASH, time.time(), 3600)
    assert auth.valid(token, HASH, time.time())
    assert not auth.valid(token, hash_password(PASSWORD, iterations=200_000), time.time())  # a new hash


def test_guard_lockout_doubles_and_resets():
    g = auth.Guard()
    for _ in range(auth.FREE_ATTEMPTS):
        assert g.attempt("x", HASH, now=0.0) is False
    assert g.locked_for(0.0) == auth.FIRST_LOCKOUT_SECONDS
    assert g.attempt(PASSWORD, HASH, now=1.0) is None  # locked: not even checked
    assert g.attempt("x", HASH, now=61.0) is False and g.locked_for(61.0) == 2 * auth.FIRST_LOCKOUT_SECONDS
    assert g.attempt(PASSWORD, HASH, now=500.0) is True and g.failures == 0


def test_a_plain_password_from_the_hosts_secret_settings_works(monkeypatch):
    """On Render the password is a secret environment variable: hashed in memory, never written. The hash is the
    same after a restart, so a restart signs nobody out."""
    monkeypatch.delenv("QP_DASHBOARD_PASSWORD_HASH")
    monkeypatch.setenv("QP_DASHBOARD_PASSWORD", PASSWORD)
    auth._hash_of_plain.cache_clear()
    hashed = auth.password_hash()
    assert hashed and auth.verify_password(PASSWORD, hashed)
    token = sign_in(client(), PASSWORD).cookies[auth.COOKIE]
    auth._hash_of_plain.cache_clear()  # "a restart"
    assert auth.password_hash() == hashed and auth.valid(token, auth.password_hash(), time.time())


def test_a_too_short_plain_password_keeps_the_cloud_dashboard_locked(monkeypatch):
    monkeypatch.delenv("QP_DASHBOARD_PASSWORD_HASH")
    monkeypatch.setenv("QP_DASHBOARD_PASSWORD", "short")
    auth._hash_of_plain.cache_clear()
    at = app(None, monkeypatch)
    assert any("dashboard is locked" in e.value for e in at.error) and "/auth/login" not in html(at)
    assert sign_in(client(), "short").headers["location"] == "/?signin=unavailable"


# ------------------------------------------------------------------ the real entry point
def test_streamlit_run_frontend_app_serves_the_sign_in_routes(tmp_path):
    """`streamlit run frontend/app.py` (every deployment's command) serves the pages and the sign-in routes."""
    assert discover_asgi_app(ROOT / "frontend" / "app.py").is_asgi_app
    port = _free_port()
    env = {**os.environ, "QP_DEPLOYMENT": "cloud", "QP_API_URL": "http://127.0.0.1:9",
           "QP_DASHBOARD_PASSWORD_HASH": HASH, "QP_API_TOKEN": TOKEN}  # fmt: skip
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", str(port),
         "--server.headless", "true", "--server.fileWatcherType", "none"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while True:
            try:
                if httpx.get(f"{base}/_stcore/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.time() < deadline, "the dashboard did not start"
            time.sleep(0.3)
        assert httpx.get(f"{base}/", timeout=5).status_code == 200
        wrong = httpx.post(f"{base}/auth/login", data={"password": "nope"}, timeout=10)
        assert wrong.status_code == 303 and wrong.headers["location"] == "/?signin=wrong"
        ok = httpx.post(f"{base}/auth/login", data={"password": PASSWORD, "remember": "1"}, timeout=10)
        assert ok.status_code == 303 and "HttpOnly" in ok.headers["set-cookie"]
        assert httpx.get(f"{base}/app/static/Inter-latin.woff2", timeout=5).status_code == 200  # the typeface
    finally:
        proc.terminate()
        proc.wait(timeout=20)
