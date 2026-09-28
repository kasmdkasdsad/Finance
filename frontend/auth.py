"""Dashboard login: nothing renders — no page, no data, no control — until the password is verified.

The password is checked against ``QP_DASHBOARD_PASSWORD_HASH`` (PBKDF2, made with
``quantpulse-hash-password``), or against ``QP_DASHBOARD_PASSWORD`` kept as a secret in the host's settings (on
Render) and hashed in memory at start; nothing is written anywhere. In the cloud (``QP_DEPLOYMENT=cloud``)
the login is mandatory: without a hash the dashboard stays locked. Locally, without a hash, it opens as
before.

Repeated wrong passwords lock the login for everyone for a while (doubling each time, at most 15 minutes):
a guessing attack gets a handful of tries an hour. A session signs out by itself after 12 hours, or with
the Sign out button. Failed attempts are logged without what was typed.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock

import streamlit as st

from quantpulse.core.passwords import PasswordHashError, hash_password, verify_password

logger = logging.getLogger("quantpulse.dashboard.auth")
FREE_ATTEMPTS = 5
FIRST_LOCKOUT_SECONDS = 60.0
MAX_LOCKOUT_SECONDS = 900.0
SESSION_SECONDS = 12 * 3600


def cloud() -> bool:
    return os.environ.get("QP_DEPLOYMENT", "").strip().lower() == "cloud"


@st.cache_resource
def _hash_of_plain(password: str) -> str | None:
    """``QP_DASHBOARD_PASSWORD`` (a secret in the host's settings, e.g. Render): hashed once, in memory."""
    try:
        return hash_password(password)
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
    """Wrong-password bookkeeping shared by every browser session of this dashboard process."""

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


@st.cache_resource
def _guard() -> Guard:
    return Guard()


def signed_in(now: float | None = None) -> bool:
    at = st.session_state.get("qp_authenticated_at")
    return bool(at) and (now or time.time()) - float(at) < SESSION_SECONDS


def sign_out() -> None:
    for key in ("qp_authenticated_at", "api_token"):
        st.session_state.pop(key, None)


def require_login() -> None:
    """Stop the script run here unless this browser session has signed in (or no login is configured)."""
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
        return
    sign_out()
    st.title("QuantPulse", anchor=False)
    st.caption("Alpaca PAPER trading · simulated money only")
    guard = _guard()
    wait = guard.locked_for(time.time())
    with st.form("qp_login", clear_on_submit=True):
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if wait > 0:
        st.error(f"Too many wrong passwords: try again in {int(wait) + 1} s.", icon=":material/timer:")
    elif submitted:
        result = guard.attempt(password, hashed, time.time())
        if result:
            st.session_state["qp_authenticated_at"] = time.time()
            st.rerun()
        elif result is None:
            st.error("Too many wrong passwords: try again later.", icon=":material/timer:")
        else:
            st.error("Wrong password.", icon=":material/lock:")
    st.stop()
