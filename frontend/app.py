"""QuantPulse dashboard — the entry point: ``streamlit run frontend/app.py`` (the API must be running).

Serves the Streamlit pages (``frontend/dashboard.py``) and the two sign-in routes, ``POST /auth/login`` and
``POST /auth/logout``. The dashboard server checks the password itself and keeps the sign-in in an HttpOnly
cookie, so closing the tab or the browser does not sign anyone out (see ``frontend/auth.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `streamlit run frontend/app.py` from the repo root
    sys.path.insert(0, str(ROOT))

from frontend import auth  # noqa: E402

app = st.App(Path(__file__).with_name("dashboard.py"), routes=auth.routes())
