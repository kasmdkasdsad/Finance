"""Every terminal screen renders without exceptions against a real API, including interactive flows."""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["overview", "options", "picks", "valuation", "portfolio", "vehicle", "sports", "system"]


def page(module: str, api_url: str, state: dict | None = None) -> AppTest:
    at = AppTest.from_string(
        f"""
import streamlit as st
st.session_state.setdefault("api_url", {api_url!r})
st.session_state.setdefault("refresh_seconds", None)
from frontend.views import {module}
{module}.render()
""",
        default_timeout=120,
    )
    for key, value in (state or {}).items():
        at.session_state[key] = value
    return at


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.mark.parametrize("module", PAGES)
def test_page_renders(api_server, module):
    at = page(module, api_server)
    at.run()
    assert_clean(at)


def test_unreachable_api_shows_actionable_error():
    at = page("overview", "http://127.0.0.1:9")
    at.run()
    assert not at.exception
    assert at.error and "cannot reach the QuantPulse API" in at.error[0].value


@pytest.mark.parametrize(
    ("module", "state"),
    [
        ("options", {"opt_view": "Volatility surface"}),
        ("options", {"opt_view": "Option chain"}),
        ("valuation", {"val_view": "Financial statements"}),
        ("valuation", {"val_view": "Consensus estimates"}),
        ("sports", {"sports_view": "Power ratings"}),
    ],
)
def test_alternate_views(api_server, module, state):
    at = page(module, api_server, state)
    at.run()
    assert_clean(at)
    assert at.get("plotly_chart") or at.dataframe


def test_forms_submit(api_server):
    at = page("valuation", api_server)
    at.run()
    next(b for b in at.button if b.label == "Run valuation").click().run()
    assert_clean(at)
    assert len(at.get("plotly_chart")) == 3 and len(at.metric) == 4

    at = page("portfolio", api_server)
    at.run()
    next(b for b in at.button if b.label == "Run risk analysis").click().run()
    assert_clean(at)
    assert len(at.metric) == 5 and len(at.get("plotly_chart")) == 3

    at = page("options", api_server, {"opt_view": "BSM calculator"})
    at.run()
    next(b for b in at.button if b.label == "Price option").click().run()
    assert_clean(at)
    assert any(m.label == "Volatility" for m in at.metric)


def test_vehicle_registration_flow(api_server):
    at = page("vehicle", api_server)
    at.run()
    create = [b for b in at.button if b.label == "Create"]
    if create:  # first run in the session: register the Elantra
        create[0].click().run()
    at = page("vehicle", api_server)
    at.run()
    assert_clean(at)
    assert any(m.label == "Cost per mile" for m in at.metric)


def test_picks_email_refuses_synthetic(api_server):
    at = page("picks", api_server)
    at.run()
    next(b for b in at.button if b.label == "Send email").click().run()
    assert not at.exception
    assert any("synthetic" in w.value.lower() for w in at.warning)
