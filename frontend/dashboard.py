"""QuantPulse dashboard — the pages and the navigation (served by ``frontend/app.py``).

Nothing renders before the sign-in. Six pages in the top bar: Home, Portfolio, Brain, Options, Research and
System; on a phone the first five are also a tab bar at the bottom of the screen. The older analytics tools
are only listed when running locally. Every page header carries the Paper pill: QuantPulse only ever trades
Alpaca's paper account.
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

STATIC = Path(__file__).with_name("static")
st.set_page_config(
    page_title="QuantPulse",
    page_icon=str(STATIC / "icon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
ui.inject_css()


def sidebar() -> None:
    """Settings, out of the way: auto-refresh and Sign out (locally also the API address and token)."""
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
            st.badge(f"Connected · v{health['version']}", icon=":material/check_circle:", color="green")
        except ApiError:
            st.badge("API offline", icon=":material/cloud_off:", color="red")
        live = st.toggle("Refresh every 30 s", value=True, help="Home updates by itself")
        st.session_state["refresh_seconds"] = 30 if live else None
        if auth.password_hash():
            st.html(auth.sign_out_form())


auth.require_login()  # nothing below renders (no page, no data, no control) before the password is verified
sidebar()
st.logo(str(STATIC / "logo.png"), size="large", icon_image=str(STATIC / "logo.png"))


def page(view, title: str, icon: str, path: str, **kw) -> st.Page:  # type: ignore[no-untyped-def]
    return st.Page(view.render, title=title, icon=f":material/{icon}:", url_path=path, **kw)


main = [
    page(remote, "Home", "home", "remote", default=True),
    page(trading, "Portfolio", "account_balance_wallet", "trading"),
    page(brain, "Brain", "psychology", "brain"),
    page(options_brain, "Options", "stacked_line_chart", "options-intelligence"),
    page(research, "Research", "biotech", "research"),
]
pages: dict[str, list[st.Page]] = {"": [*main, page(system, "System", "monitor_heart", "system")]}
if not auth.cloud():  # the older analytics tools, unrelated to the paper-trading Brain: local only
    pages["Tools"] = [
        page(overview, "Market overview", "dashboard", "overview"),
        page(stock, "Stock intelligence", "query_stats", "stock"),
        page(picks, "Daily picks", "star", "picks"),
        page(track_record, "Track record", "fact_check", "track-record"),
        page(model_lab, "Model lab", "model_training", "model-lab"),
        page(options, "Options lab", "candlestick_chart", "options"),
        page(sandbox, "Trading sandbox", "smart_toy", "sandbox"),
        page(valuation, "Valuation", "account_balance", "valuation"),
        page(portfolio, "Risk lab", "insights", "portfolio"),
        page(vehicle, "Asset lifecycle", "directions_car", "vehicle"),
        page(sports, "Sports hub", "sports_football", "sports"),
    ]
st.session_state["qp_pages"] = {p.url_path: p for group in pages.values() for p in group}  # for ui.link
current = st.navigation(pages, position="top")
ui.tab_bar(main)  # on a phone: one tap to any of the five main pages
current.run()
