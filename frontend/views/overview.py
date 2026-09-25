"""Command Center: live watchlist, market session, Treasury curve and today's top pick."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, status_badge

DEFAULT_WATCHLIST = "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META"


def _tape(symbols: str) -> None:
    data = guarded(lambda: api().get("/market/quotes", symbols=symbols), "quotes")
    if not data:
        return
    quotes = data["quotes"]
    cols = st.columns(min(len(quotes), 4))
    for i, (symbol, env) in enumerate(quotes.items()):
        q = env["data"]
        with cols[i % len(cols)], st.container(border=True):
            st.metric(
                symbol,
                f"{q['price']:,.2f}",
                None if q.get("change_percent") is None else f"{q['change_percent']:+.2f}%",
            )
            status_badge(env["meta"])


def render() -> None:
    st.title("Command Center")
    session = guarded(lambda: api().get("/market/session"), "market session")
    if session:
        labels = {
            "regular": "NYSE open",
            "pre": "Pre-market",
            "post": "After hours",
            "closed": "Market closed",
        }
        st.caption(
            f"{labels.get(session['session'], session['session'])} · next open {session['next_open'][:16].replace('T', ' ')} ET"
        )

    symbols = st.text_input(
        "Watchlist", st.session_state.get("watchlist", DEFAULT_WATCHLIST), key="watchlist_input"
    )
    st.session_state["watchlist"] = symbols
    refresh = st.session_state.get("refresh_seconds")

    @st.fragment(run_every=refresh)
    def live_tape() -> None:
        _tape(symbols)

    live_tape()

    left, right = st.columns([3, 2], gap="large")
    with left:
        curve = guarded(lambda: api().get("/rates/curve"), "yield curve")
        if curve:
            pts = curve["data"]["points"]
            fig = go.Figure(
                go.Scatter(
                    x=[p["years"] for p in pts],
                    y=[p["rate"] * 100 for p in pts],
                    mode="lines+markers",
                    line={"color": charts.series(0), "width": 2},
                    marker={"size": 8},
                    text=[p["tenor"] for p in pts],
                    hovertemplate="%{text}: %{y:.2f}%<extra></extra>",
                    name="Par yield",
                )
            )
            charts.base_layout(fig, f"U.S. Treasury par curve · {curve['data']['as_of']}", height=320)
            fig.update_xaxes(
                type="log",
                title="Maturity (years, log scale)",
                tickvals=[0.083, 0.25, 0.5, 1, 2, 5, 10, 30],
                ticktext=["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"],
            )
            fig.update_yaxes(title="Yield (%)", ticksuffix="%")
            charts.show(fig)
            status_badge(curve["meta"], label="Treasury")
            with st.expander("Table view"):
                st.dataframe(
                    pd.DataFrame(pts).assign(rate=lambda d: (d["rate"] * 100).round(3)), hide_index=True
                )
    with right:
        picks = guarded(lambda: api().get("/picks/daily", top_n=5), "daily picks")
        if picks:
            d = picks["data"]
            st.subheader(f"Top picks · {d['trading_day']}")
            composite_badges(picks["meta"])
            for p in d["picks"]:
                with st.container(border=True):
                    a, b = st.columns([3, 1], vertical_alignment="center")
                    a.markdown(
                        f"**{p['rank']}. {p['symbol']}** · {p['name'] or ''}  \n"
                        f"<small>{', '.join(p['drivers']) or 'balanced profile'}</small>",
                        unsafe_allow_html=True,
                    )
                    b.markdown(
                        f"<div style='text-align:right;font-size:1.5rem;font-weight:600;white-space:nowrap'>{p['rating']}/10</div>",
                        unsafe_allow_html=True,
                    )
            st.caption(d["disclaimer"])
