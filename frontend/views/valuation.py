"""Valuation Suite: SEC fundamentals, consensus estimates, live DCF, sensitivity and Monte Carlo."""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, money, num, pct, status_badge


def _fundamentals(symbol: str) -> None:
    res = guarded(lambda: api().get(f"/fundamentals/{symbol}"), "fundamentals")
    if not res:
        return
    d = res["data"]
    status_badge(res["meta"], label="SEC EDGAR")
    st.caption(
        f"{d.get('name') or symbol} · CIK {d.get('cik') or '—'} · shares {num(d.get('shares_outstanding'), 0)} (as of {d.get('shares_as_of') or '—'})"
    )
    df = pd.DataFrame(d["statements"])
    if df.empty:
        return
    df["free_cash_flow"] = df["operating_cash_flow"] - df["capital_expenditure"]
    fig = go.Figure()
    for i, (col, label) in enumerate(
        [
            ("revenue", "Revenue"),
            ("operating_income", "Operating income"),
            ("free_cash_flow", "Free cash flow"),
        ]
    ):
        fig.add_trace(
            go.Bar(
                x=df["fiscal_year"].astype(str),
                y=df[col] / 1e9,
                name=label,
                marker={"color": charts.series(i), "cornerradius": 4},
                hovertemplate=label + " FY%{x}: $%{y:,.1f}B<extra></extra>",
            )
        )
    charts.base_layout(fig, "Annual results (USD billions)", height=340, barmode="group")
    fig.update_yaxes(title="USD bn", tickprefix="$")
    charts.show(fig)
    with st.expander("Table view — normalised statements (USD)"):
        numeric = df.set_index("fiscal_year").select_dtypes("number")
        st.dataframe(
            numeric.T,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="compact") for c in numeric.index},
        )
    if d.get("recent_filings"):
        with st.expander("Recent SEC filings"):
            st.dataframe(
                pd.DataFrame(d["recent_filings"]),
                hide_index=True,
                column_config={"url": st.column_config.LinkColumn("Document")},
            )


def _dcf(symbol: str) -> None:
    with st.form("dcf"):
        c = st.columns(4)
        years = c[0].slider("Projection years", 3, 15, 5)
        g = c[1].number_input("Terminal growth (%)", -2.0, 6.0, 2.5, 0.1)
        erp = c[2].number_input("Equity risk premium (%)", 0.0, 20.0, 5.0, 0.25)
        paths = c[3].select_slider("Monte Carlo paths", [1000, 5000, 10000, 25000, 50000], value=10000)
        o = st.columns(4)
        wacc = o[0].number_input(
            "WACC override (%)", 0.0, 50.0, 0.0, 0.1, help="0 = derive from CAPM + live curve"
        )
        margin = o[1].number_input(
            "Target EBIT margin (%)", -100.0, 100.0, 0.0, 0.5, help="0 = hold the current margin"
        )
        beta = o[2].number_input(
            "Beta override", -2.0, 5.0, 0.0, 0.05, help="0 = 2y weekly regression vs SPY"
        )
        seed = o[3].number_input("Seed", 0, 2**31 - 1, 42)
        submitted = st.form_submit_button("Run valuation", type="primary")
    if not submitted and "dcf_result" not in st.session_state:
        st.info(
            "Configure assumptions and run the valuation. Every blank/zero field is derived from live data."
        )
        return
    if submitted:
        payload = {
            "years": years,
            "terminal_growth": g / 100,
            "equity_risk_premium": erp / 100,
            "monte_carlo": {"paths": paths, "seed": int(seed)},
        }
        if wacc > 0:
            payload["wacc"] = wacc / 100
        if margin != 0:
            payload["target_ebit_margin"] = margin / 100
        if beta != 0:
            payload["beta"] = beta
        res = guarded(lambda: api().post(f"/valuation/{symbol}/dcf", payload), "valuation")
        if not res:
            return
        st.session_state["dcf_result"] = (symbol, res)
    sym, res = st.session_state["dcf_result"]
    if sym != symbol:
        st.info("Run the valuation for the new symbol.")
        return
    d = res["data"]
    composite_badges(res["meta"])
    for w in d["warnings"]:
        st.warning(w, icon=":material/warning:")
    dcf, w = d["dcf"], d["wacc"]
    m = st.columns(4)
    m[0].metric(
        "Intrinsic value / share",
        num(dcf["value_per_share"]),
        None if dcf["upside"] is None else f"{dcf['upside'] * 100:+.1f}% vs price",
    )
    m[1].metric("Price", num(dcf["current_price"]))
    m[2].metric(
        "WACC",
        pct(w["wacc"]),
        help=f"Re {pct(w['cost_of_equity'])} · Rd {pct(w['pre_tax_cost_of_debt'])} · β {w['beta']:.2f} ({w['beta_source']})",
    )
    m[3].metric(
        "Enterprise value",
        money(dcf["enterprise_value"]),
        help=f"PV(TV) = {pct(dcf['terminal_value_share'], 0)} of EV",
    )

    left, right = st.columns(2, gap="large")
    with left:
        proj = pd.DataFrame(dcf["projections"])
        fig = go.Figure(
            go.Bar(
                x=proj["year"],
                y=proj["free_cash_flow"] / 1e9,
                name="FCFF",
                marker={"color": charts.series(0), "cornerradius": 4},
                hovertemplate="Year %{x}: $%{y:,.2f}B<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=proj["year"],
                y=proj["present_value"] / 1e9,
                name="Present value",
                marker={"color": charts.series(1), "cornerradius": 4},
                hovertemplate="Year %{x}: $%{y:,.2f}B<extra></extra>",
            )
        )
        charts.base_layout(fig, "Free cash flow to firm (USD bn)", height=320, barmode="group")
        charts.show(fig)
    with right:
        mc = d.get("monte_carlo")
        if mc:
            edges, counts = mc["histogram_edges"], mc["histogram_counts"]
            mids = [(a + b) / 2 for a, b in pairwise(edges)]
            fig = go.Figure(
                go.Bar(
                    x=mids,
                    y=counts,
                    marker={"color": charts.series(0), "cornerradius": 4},
                    name="Paths",
                    hovertemplate="$%{x:,.2f}: %{y} paths<extra></extra>",
                )
            )
            if dcf["current_price"]:
                fig.add_vline(
                    x=dcf["current_price"],
                    line_color=charts.theme()["ink2"],
                    line_width=2,
                    annotation_text=f"price {dcf['current_price']:,.2f}",
                    annotation_position="top",
                )
            charts.base_layout(
                fig,
                f"Monte Carlo value per share ({mc['valid_paths']:,} paths)",
                height=320,
                showlegend=False,
                bargap=0.05,
            )
            fig.update_xaxes(title="Value per share", tickprefix="$")
            charts.show(fig)
            p = mc["percentiles"]
            st.caption(
                f"P5 {num(p['p5'])} · median {num(mc['median'])} · P95 {num(p['p95'])} · P(value > price) "
                f"{pct(mc['prob_above_price'], 1) if mc['prob_above_price'] is not None else '—'} · seed {mc['seed']}"
            )

    sens = d["sensitivity"]
    z = [[None if v is None else v for v in row] for row in sens["values_per_share"]]
    price = dcf["current_price"] or dcf["value_per_share"]
    flat = [v for row in z for v in row if v is not None]
    span = max(abs(max(flat) - price), abs(min(flat) - price)) if flat else 1
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{g * 100:.1f}%" for g in sens["growth_values"]],
            y=[f"{w_ * 100:.1f}%" for w_ in sens["wacc_values"]],
            colorscale=charts.diverging_scale(),
            zmid=price,
            zmin=price - span,
            zmax=price + span,
            text=[["—" if v is None else f"{v:,.0f}" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="WACC %{y} · g %{x}: $%{z:,.2f}<extra></extra>",
            colorbar={"title": "$/share", "thickness": 12},
            xgap=2,
            ygap=2,
        )
    )
    charts.base_layout(fig, "Sensitivity: value per share (blue above / red below current price)", height=320)
    fig.update_xaxes(title="Terminal growth")
    fig.update_yaxes(title="WACC", gridcolor="rgba(0,0,0,0)")
    charts.show(fig)
    with st.expander("Assumptions & projection table"):
        st.dataframe(pd.DataFrame(d["assumptions"]), hide_index=True)
        st.dataframe(pd.DataFrame(dcf["projections"]), hide_index=True)


def _estimates(symbol: str) -> None:
    res = guarded(lambda: api().get(f"/fundamentals/{symbol}/estimates"), "estimates")
    if not res:
        return
    d = res["data"]
    status_badge(res["meta"], label="Consensus")
    m = st.columns(4)
    m[0].metric("Mean target", num(d.get("target_mean_price")))
    m[1].metric("Analysts", num(d.get("analyst_count"), 0))
    m[2].metric(
        "Recommendation",
        (d.get("recommendation_key") or "—").replace("_", " ").title(),
        help="Mean on a 1 (strong buy) … 5 (sell) scale: " + num(d.get("recommendation_mean")),
    )
    m[3].metric("Long-term growth", pct(d.get("long_term_growth")))
    if d["periods"]:
        st.dataframe(pd.DataFrame(d["periods"]), hide_index=True)


def render() -> None:
    st.title("Valuation Suite")
    symbol = (
        st.text_input("Company ticker", st.session_state.get("val_symbol", "AAPL"), key="val_symbol_input")
        .strip()
        .upper()
        or "AAPL"
    )
    st.session_state["val_symbol"] = symbol
    views = {
        "DCF & Monte Carlo": _dcf,
        "Financial statements": _fundamentals,
        "Consensus estimates": _estimates,
    }
    choice = st.segmented_control(
        "View", list(views), default="DCF & Monte Carlo", key="val_view", label_visibility="collapsed"
    )
    views[choice or "DCF & Monte Carlo"](symbol)
