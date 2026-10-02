"""QuantPulse dashboard — the pages and the navigation (served by ``frontend/app.py``).

Nothing renders before the sign-in. The trading pages come first in the top bar; the analytics tools sit
under "More". Every page header carries the PAPER pill: QuantPulse only ever trades Alpaca's paper account.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run frontend/dashboard.py` also works (without the sign-in routes)
    sys.path.insert(0, str(ROOT))

from frontend import auth, ui  # noqa: E402
from frontend.api_client import ApiClient, ApiError  # noqa: E402
from frontend.views import (  # noqa: E402
    brain,
    evolution,
    model_lab,
    options,
    options_brain,
    overview,
    picks,
    portfolio,
    remote,
    research,
    sandbox,
    sports,
    stock,
    system,
    track_record,
    trading,
    valuation,
    vehicle,
)

ICON = ROOT / "assets" / "quantpulse.png"
st.set_page_config(
    page_title="QuantPulse",
    page_icon=str(ICON) if ICON.is_file() else ":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()


def sidebar() -> None:
    """Settings, out of the way: the connection, auto-refresh and Sign out."""
    with st.sidebar:
        st.markdown("**Settings**")
        if auth.cloud():
            # the server's own API, with the server's token: neither can be changed (or seen) from a browser,
            # so the token can never be sent anywhere else
            url, token = ApiClient().base_url, ""
        else:
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
        live = st.toggle(
            "Auto-refresh", value=True, help="Home and the live market panels update by themselves"
        )
        seconds = st.select_slider(
            "Every", [15, 30, 60, 120], value=30, disabled=not live, format_func=lambda s: f"{s}s"
        )
        st.session_state["refresh_seconds"] = seconds if live else None
        if auth.password_hash():
            st.html(auth.sign_out_form())


auth.require_login()  # nothing below renders (no page, no data, no control) before the password is verified
sidebar()
if ICON.is_file():
    st.logo(str(ICON), size="large")


def page(view, title: str, icon: str, path: str, **kw) -> st.Page:  # type: ignore[no-untyped-def]
    return st.Page(view.render, title=title, icon=f":material/{icon}:", url_path=path, **kw)


tools = [
    page(system, "System status", "monitor_heart", "system"),
    page(evolution, "Market evolution", "timeline", "evolution"),
    page(overview, "Market overview", "dashboard", "overview"),
    page(stock, "Stock intelligence", "query_stats", "stock"),
    page(picks, "Daily picks", "star", "picks"),
    page(track_record, "Track record", "fact_check", "track-record"),
    page(model_lab, "Model lab", "model_training", "model-lab"),
    page(options, "Options lab", "candlestick_chart", "options"),
    page(sandbox, "Trading sandbox", "smart_toy", "sandbox"),
    page(valuation, "Valuation", "account_balance", "valuation"),
    page(portfolio, "Risk lab", "insights", "portfolio"),
]
if not auth.cloud():  # local extras, unrelated to trading
    tools += [
        page(vehicle, "Asset lifecycle", "directions_car", "vehicle"),
        page(sports, "Sports hub", "sports_football", "sports"),
    ]
pages = {
    "": [
        page(remote, "Home", "home", "remote", default=True),
        page(brain, "Brain", "psychology", "brain"),
        page(trading, "Portfolio", "account_balance_wallet", "trading"),
        page(options_brain, "Options", "stacked_line_chart", "options-intelligence"),
        page(research, "Research", "biotech", "research"),
    ],
    "More": tools,
}
st.navigation(pages, position="top").run()
