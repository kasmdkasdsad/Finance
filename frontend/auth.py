"""Dashboard login: nothing renders — no page, no data, no control — until the password is verified.

The password is checked against ``QP_DASHBOARD_PASSWORD_HASH`` (PBKDF2, made with
``quantpulse-hash-password``), or against ``QP_DASHBOARD_PASSWORD`` kept as a secret in the host's settings (on
Render) and hashed in memory at start; nothing is written anywhere. In the cloud (``QP_DEPLOYMENT=cloud``)
the login is mandatory: without a hash the dashboard stays locked. Locally, without a hash, it opens as
before.

The dashboard server checks the password itself (``POST /auth/login``, mounted by ``frontend/app.py``), not
the page. A right password gets a session cookie that is HttpOnly (page scripts cannot read it),
SameSite=Strict and, over HTTPS, Secure. The sign-in therefore survives closing the tab or the browser:

* for 30 days with "Keep me signed in" (the default);
* until the browser closes without it.

The cookie holds no password, only an expiry and a random id. They are signed (HMAC-SHA256) with a key derived
from the password hash, so changing the password signs every browser out. Sign out (``POST /auth/logout``)
deletes the cookie and revokes its id for the life of the dashboard process.

Repeated wrong passwords lock the login for everyone for a while (doubling each time, at most 15 minutes):
a guessing attack gets a handful of tries an hour. Failed attempts are logged without what was typed.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import html
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from threading import Lock
from urllib.parse import urlsplit

import streamlit as st
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from quantpulse.core.passwords import PasswordHashError, hash_password, verify_password

logger = logging.getLogger("quantpulse.dashboard.auth")
FREE_ATTEMPTS = 5
FIRST_LOCKOUT_SECONDS = 60.0
MAX_LOCKOUT_SECONDS = 900.0
COOKIE = "qp_session"
REMEMBER_SECONDS = 30 * 86400  # "Keep me signed in"
SESSION_SECONDS = 12 * 3600  # without it: until the browser closes, and never longer than this
MAX_PASSWORD_LENGTH = 1024
LOGIN_PATH, LOGOUT_PATH = "/auth/login", "/auth/logout"


def cloud() -> bool:
    return os.environ.get("QP_DEPLOYMENT", "").strip().lower() == "cloud"


@functools.lru_cache(maxsize=4)
def _hash_of_plain(password: str) -> str | None:
    """``QP_DASHBOARD_PASSWORD`` (a secret in the host's settings, e.g. Render): hashed once, in memory.

    The salt comes from the password itself, so a restarted dashboard derives the same hash and therefore the
    same session key: a restart does not sign anyone out. This hash never leaves the process."""
    salt = hashlib.sha256(b"quantpulse dashboard password salt\x00" + password.encode("utf-8")).digest()[:16]
    try:
        return hash_password(password, salt=salt)
    except PasswordHashError:
        return None  # shorter than the minimum: the dashboard stays locked


def password_hash() -> str | None:
    """The PBKDF2 hash to check against: ``QP_DASHBOARD_PASSWORD_HASH``, or one made in memory from
    ``QP_DASHBOARD_PASSWORD``; ``None`` when neither is usable."""
    hashed = os.environ.get("QP_DASHBOARD_PASSWORD_HASH", "").strip()
    if hashed:
        return hashed
    plain = os.environ.get("QP_DASHBOARD_PASSWORD", "")
    return _hash_of_plain(plain) if plain.strip() else None


@dataclass
class Guard:
    """Wrong-password bookkeeping shared by every browser of this dashboard process."""

    failures: int = 0
    locked_until: float = 0.0
    lock: Lock = field(default_factory=Lock)

    def locked_for(self, now: float) -> float:
        return max(0.0, self.locked_until - now)

    def attempt(self, password: str, hashed: str, now: float) -> bool | None:
        """``True``/``False`` for a checked password; ``None`` while locked out (nothing is checked)."""
        with self.lock:
            if self.locked_for(now) > 0:
                return None
            ok = verify_password(password, hashed)
            if ok:
                self.failures, self.locked_until = 0, 0.0
                return True
            self.failures += 1
            if self.failures >= FREE_ATTEMPTS:
                extra = self.failures - FREE_ATTEMPTS
                self.locked_until = now + min(MAX_LOCKOUT_SECONDS, FIRST_LOCKOUT_SECONDS * 2**extra)
            logger.warning("dashboard login failed (%d in a row)", self.failures)
            return False

    def reset(self) -> None:
        with self.lock:
            self.failures, self.locked_until = 0, 0.0


GUARD = Guard()


# ------------------------------------------------------------------ session cookies
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign(hashed: str, body: str) -> str:
    key = hmac.new(hashed.encode("utf-8"), b"quantpulse dashboard session v1", hashlib.sha256).digest()
    return _b64(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())


_revoked: dict[str, float] = {}  # session id -> its expiry (forgotten once expired)
_revoked_lock = Lock()


def issue(hashed: str, now: float, lifetime: float) -> str:
    """A session token: ``v1.<expiry>.<random id>.<signature>``."""
    body = f"v1.{int(now + lifetime)}.{secrets.token_urlsafe(16)}"
    return f"{body}.{_sign(hashed, body)}"


def _parts(token: str) -> tuple[int, str] | None:
    if not token.isascii():  # a cookie is whatever the browser sends: anything else is not ours
        return None
    parts = token.split(".")
    if len(parts) != 4 or parts[0] != "v1" or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2]


def valid(token: str | None, hashed: str, now: float) -> bool:
    """Signed with this password's key, not expired and not signed out."""
    if not token or len(token) > 256:
        return False
    parsed = _parts(token)
    if parsed is None:
        return False
    expires, sid = parsed
    body = token.rsplit(".", 1)[0]
    if not hmac.compare_digest(_sign(hashed, body), token.rsplit(".", 1)[1]):
        return False
    with _revoked_lock:
        return expires > now and sid not in _revoked


def revoke(token: str, now: float) -> None:
    parsed = _parts(token)
    if parsed is None:
        return
    with _revoked_lock:
        for sid in [s for s, exp in _revoked.items() if exp <= now]:
            del _revoked[sid]
        _revoked[parsed[1]] = float(parsed[0])


def _session_cookie() -> str | None:
    """The browser's session cookie, as sent when the page connected."""
    return st.context.cookies.get(COOKIE)


def signed_in(now: float | None = None) -> bool:
    hashed = password_hash()
    return hashed is not None and valid(_session_cookie(), hashed, now or time.time())


# ------------------------------------------------------------------ the server routes (frontend/app.py)
def _https(request: Request) -> bool:
    origin = request.headers.get("origin", "")
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https" or origin.startswith("https://")


def _safe_next(target: str) -> str:
    """Only a path on this dashboard: never another site (``//host``, ``\\``, a scheme)."""
    parts = urlsplit(target or "/")
    if parts.scheme or parts.netloc or not parts.path.startswith("/") or "\\" in parts.path:
        return "/"
    return parts.path


def _cross_site(request: Request) -> bool:
    return request.headers.get("sec-fetch-site", "") == "cross-site"


async def _login(request: Request) -> Response:
    if _cross_site(request):
        return PlainTextResponse("Sign in from the dashboard itself.", status_code=403)
    form = await request.form()
    target = _safe_next(str(form.get("next") or "/"))
    password = str(form.get("password") or "")[:MAX_PASSWORD_LENGTH]
    hashed = password_hash()
    if hashed is None:
        return RedirectResponse(f"{target}?signin=unavailable", status_code=303)
    now = time.time()
    result = await run_in_threadpool(GUARD.attempt, password, hashed, now)  # PBKDF2: off the event loop
    if not result:
        return RedirectResponse(f"{target}?signin={'locked' if result is None else 'wrong'}", status_code=303)
    remember = form.get("remember") is not None
    lifetime = REMEMBER_SECONDS if remember else SESSION_SECONDS
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        COOKIE,
        issue(hashed, now, lifetime),
        max_age=lifetime if remember else None,
        path="/",
        secure=_https(request),
        httponly=True,
        samesite="strict",
    )
    return response


async def _logout(request: Request) -> Response:
    if _cross_site(request):
        return PlainTextResponse("Sign out from the dashboard itself.", status_code=403)
    token = request.cookies.get(COOKIE)
    if token:
        revoke(token, time.time())
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie(COOKIE, path="/", secure=_https(request), httponly=True, samesite="strict")
    return response


def routes() -> list[Route]:
    return [Route(LOGIN_PATH, _login, methods=["POST"]), Route(LOGOUT_PATH, _logout, methods=["POST"])]


# ------------------------------------------------------------------ the page side
MESSAGES = {
    "wrong": "Wrong password.",
    "unavailable": "The dashboard is locked: no password is configured on the server.",
}


def _next_path() -> str:
    """Where this browser is (``st.context.url`` is the full address): back there after signing in."""
    return _safe_next(urlsplit(st.context.url or "/").path)


def _login_form(message: str | None) -> str:
    note = f'<p class="qp-login-error" role="alert">{html.escape(message)}</p>' if message else ""
    return f"""
<div class="qp-login">
  <div class="qp-login-mark">QP</div>
  <h1>QuantPulse</h1>
  <p class="qp-login-sub">Alpaca paper trading · simulated money only</p>
  {note}
  <form method="post" action="{LOGIN_PATH}">
    <input type="hidden" name="next" value="{html.escape(_next_path(), quote=True)}">
    <label for="qp-password">Password</label>
    <input id="qp-password" name="password" type="password" autocomplete="current-password" required>
    <label class="qp-login-remember"><input type="checkbox" name="remember" value="1" checked>
      Keep me signed in for 30 days</label>
    <button type="submit">Sign in</button>
  </form>
</div>"""


def require_login() -> None:
    """Stop the script run here unless this browser has signed in (or no login is configured)."""
    hashed = password_hash()
    if hashed is None:
        if cloud():
            st.error(
                "The dashboard is locked: set QP_DASHBOARD_PASSWORD (at least 12 characters) or "
                "QP_DASHBOARD_PASSWORD_HASH (`quantpulse-hash-password`) on the server, then restart it.",
                icon=":material/lock:",
            )
            st.stop()
        return
    if signed_in():
        if "signin" in st.query_params:
            del st.query_params["signin"]
        return
    reason = st.query_params.get("signin")
    wait = GUARD.locked_for(time.time())
    message = (
        f"Too many wrong passwords: try again in {int(wait) + 1} s."
        if wait > 0
        else MESSAGES.get(reason or "")
    )
    st.html(_login_form(message))
    st.stop()


def sign_out_form() -> str:
    """The Sign out button: a plain form to the server, which deletes the cookie."""
    return f'<form method="post" action="{LOGOUT_PATH}" class="qp-signout"><button type="submit">Sign out</button></form>'
