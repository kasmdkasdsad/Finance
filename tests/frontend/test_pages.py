"""Every terminal screen renders without exceptions against a real API, including interactive flows."""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    "overview",
    "options",
    "stock",
    "picks",
    "model_lab",
    "track_record",
    "sandbox",
    "valuation",
    "portfolio",
    "vehicle",
    "sports",
    "system",
]


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


def test_sandbox_account_agent_and_training_flow(api_server):
    at = page("sandbox", api_server)
    at.run()
    assert_clean(at)
    name = next(t for t in at.text_input if t.label == "Account name")
    name.set_value("UI agent")
    next(c for c in at.checkbox if c.label == "Allow synthetic prices").check()
    next(b for b in at.button if b.label == "Open account").click().run()
    assert_clean(at)
    assert any(m.label == "Equity" and m.value == "$100,000.00" for m in at.metric)
    assert any("SYNTHETIC" in w.value for w in at.warning)  # the account is flagged as not-real

    at.toggle(key="sandbox_force").set_value(True)
    next(b for b in at.button if b.label == "Run agent now").click().run()
    assert_clean(at)
    assert any("Agent ran for" in s.value for s in at.success)
    assert len(at.get("plotly_chart")) == 1  # equity vs benchmark (created + first run)

    for view in ("Learning", "Positions", "Trades", "Journal", "Manual order", "Settings"):
        at.segmented_control(key="sandbox_view").set_value(view).run()
        assert_clean(at)
    at.segmented_control(key="sandbox_view").set_value("Manual order").run()
    next(b for b in at.button if b.label == "Place paper order").click().run()
    assert_clean(at)
    assert any(s.value.startswith("Filled: buy") for s in at.success)

    next(b for b in at.button if b.label == "Train").click().run()
    assert_clean(at)
    assert at.session_state["sandbox_view"] == "Learning"
    assert any(m.label == "Weights applied" and m.value == "Yes" for m in at.metric)  # synthetic allowed
    assert len(at.get("plotly_chart")) == 3  # learned weights, replay equity, replay weights


def test_prediction_pages_interactions(api_server):
    at = page("stock", api_server, {"stock_symbol": "MSFT"})
    at.run()
    assert_clean(at)
    assert any(m.label == "P(higher in 1 month)" for m in at.metric)
    assert len(at.get("plotly_chart")) >= 1 and any("SYNTHETIC" in w.value for w in at.warning)
    assert any(m.label == "Typical earnings move" for m in at.metric)  # the earnings section
    assert any(m.label == "Used by the forecast" for m in at.metric)  # GARCH + options blend
    next(b for b in at.button if b.label == "Test the forecaster on this stock's history").click().run()
    assert_clean(at)
    assert any(m.label == "Inside 90% range" for m in at.metric)

    at = page("model_lab", api_server)
    at.run()
    assert_clean(at)
    labels = {m.label for m in at.metric}
    assert {"Mean IC (model)", "IC within industries", "Former members modelled", "Fundamentals"} <= labels
    # IC, backtest, buckets, calibration, linear weights, tree importance, stocks per industry
    assert len(at.get("plotly_chart")) == 7
    assert any("Ensemble" in str(df.value) for df in at.dataframe)  # the model comparison table
    industries = at.multiselect(key="lab_industries")
    industries.set_value(industries.options[:1]).run()
    assert_clean(at)
    at.segmented_control(key="lab_view").set_value("Signal research").run()
    assert_clean(at)
    assert len(at.get("plotly_chart")) == 1

    at = page("track_record", api_server)
    at.run()
    assert_clean(at)
    assert any("No predictions in this record yet" in i.value for i in at.info)
    next(b for b in at.button if b.label == "Run the replay").click().run()
    assert_clean(at)
    at.segmented_control(key="track_origin").set_value("Historical replay").run()
    assert_clean(at)

    at = page("picks", api_server)
    at.run()
    at.segmented_control(key="picks_method").set_value("model").run()
    assert_clean(at)
    assert any("Ranked by the stock model" in c.value for c in at.caption)


def test_training_progress_is_shown_while_a_job_runs(api_server):
    at = AppTest.from_string(
        f"""
import streamlit as st
st.session_state.setdefault("api_url", {api_server!r})
from frontend.components import job_progress
job_progress({{"id": "model-999", "kind": "model", "description": "Stock model (sp500)", "status": "running",
    "progress": 0.42, "stage": "downloading prices (120/600)", "elapsed_seconds": 12.0}})
""",
        default_timeout=60,
    )
    at.run()
    assert not at.exception
    bars = at.get("progress")
    assert bars and "downloading prices" in bars[0].proto.text and bars[0].proto.value == 42
    assert any("runs in the background" in i.value for i in at.info)
