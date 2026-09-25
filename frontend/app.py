"""QuantPulse Terminal — Streamlit entry point.

Run with ``streamlit run frontend/app.py`` (the API must be running: ``make api``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `streamlit run frontend/app.py` from the repo root
    sys.path.insert(0, str(ROOT))

from frontend.api_client import ApiClient, ApiError  # noqa: E402
from frontend.views import (  # noqa: E402
    options,
    overview,
    picks,
    portfolio,
    sports,
    system,
    valuation,
    vehicle,
)

st.set_page_config(page_title="QuantPulse Terminal", page_icon=":material/monitoring:", layout="wide")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### QuantPulse Terminal")
        st.session_state.setdefault("api_url", ApiClient().base_url)
        url = st.text_input("API URL", st.session_state["api_url"])
        token = st.text_input(
            "API token",
            st.session_state.get("api_token", ""),
            type="password",
            help="Only if QP_API_TOKEN is set",
        )
        st.session_state["api_url"], st.session_state["api_token"] = url.rstrip("/"), token
        try:
            health = ApiClient(url, token or None, timeout=5).health()
            st.badge(f"API online · v{health['version']}", icon=":material/check_circle:", color="green")
        except ApiError:
            st.badge("API offline", icon=":material/cloud_off:", color="red")
        live = st.toggle("Auto-refresh live panels", value=True)
        seconds = st.select_slider(
            "Refresh every", [5, 10, 15, 30, 60], value=15, disabled=not live, format_func=lambda s: f"{s}s"
        )
        st.session_state["refresh_seconds"] = seconds if live else None
        st.caption(
            "Badges: LIVE = fetched now · CACHED = recent live fetch · STALE = last good value after a failure · "
            "SYNTHETIC = simulated fallback. Hover a badge for the per-provider trail."
        )


sidebar()
pages = {
    "Markets": [
        st.Page(overview.render, title="Command Center", icon=":material/dashboard:", default=True),
        st.Page(options.render, title="Options Lab", icon=":material/candlestick_chart:", url_path="options"),
        st.Page(picks.render, title="Daily Picks", icon=":material/star:", url_path="picks"),
    ],
    "Corporate & Risk": [
        st.Page(
            valuation.render, title="Valuation Suite", icon=":material/account_balance:", url_path="valuation"
        ),
        st.Page(portfolio.render, title="Risk Laboratory", icon=":material/insights:", url_path="portfolio"),
    ],
    "Operations": [
        st.Page(
            vehicle.render, title="Asset Lifecycle", icon=":material/directions_car:", url_path="vehicle"
        ),
        st.Page(sports.render, title="Sports Hub", icon=":material/sports_football:", url_path="sports"),
        st.Page(system.render, title="System Status", icon=":material/monitor_heart:", url_path="system"),
    ],
}
st.navigation(pages).run()
