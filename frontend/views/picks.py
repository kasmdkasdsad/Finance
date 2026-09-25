"""Daily Picks: factor-ranked stocks with a 1-10 rating and the email digest."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, num

FACTOR_LABELS = {
    "momentum_12_1": "12-1m momentum",
    "momentum_3m": "3m momentum",
    "trend": "Trend",
    "risk_adjusted": "Risk-adjusted",
    "low_volatility": "Low vol",
    "reversal": "Pullback",
}


def render() -> None:
    st.title("Daily Picks")
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    top_n = c1.slider("How many", 5, 30, 10)
    method = c2.segmented_control(
        "Ranking",
        ["auto", "factors", "model", "blend"],
        default="auto",
        key="picks_method",
        format_func={"auto": "Auto", "factors": "Factor rule", "model": "Stock model", "blend": "Blend"}.get,
        help="Auto uses the stock model only when it has shown out-of-sample skill.",
    )
    refresh = c3.toggle("Force fresh prices", value=False, help="Bypass the cache (slower; uses API quota)")
    with st.spinner("Ranking the universe…"):
        res = guarded(
            lambda: api().get("/picks/daily", top_n=top_n, refresh=refresh, method=method or "auto"),
            "daily picks",
        )
    if not res:
        return
    d = res["data"]
    composite_badges(res["meta"])
    if d["data_status"] == "synthetic":
        st.warning(
            "Live prices are unavailable, so these ratings are computed from SYNTHETIC data and are not real.",
            icon=":material/science:",
        )
    top = d["top_pick"]
    if top:
        a, b, c = st.columns([2, 1, 1])
        a.metric(f"Best stock for {d['trading_day']}", f"{top['symbol']}", help=top.get("name") or "")
        b.metric("Rating", f"{top['rating']}/10")
        c.metric(
            "Price",
            num(top["price"]),
            None if top["change_percent"] is None else f"{top['change_percent']:+.2f}%",
        )
        st.caption("Drivers: " + (", ".join(top["drivers"]) or "balanced profile"))

    df = pd.DataFrame(
        [
            {
                "rank": p["rank"],
                "symbol": p["symbol"],
                "name": p["name"],
                "rating": p["rating"],
                "price": p["price"],
                "day_%": p["change_percent"],
                f"P(beat {d['benchmark']})": p["prob_outperform"],
                "1-month 90% range": "—"
                if p["low_21d"] is None
                else f"${p['low_21d']:,.2f} – ${p['high_21d']:,.2f}",
                "P(up, 1 mo)": p["prob_up_21d"],
                "factor rating": p["factor_rating"],
                "model rank": p["model_rank"],
                "drivers": ", ".join(p["drivers"]),
                "data": p["data_status"],
            }
            for p in d["picks"]
        ]
    )
    fig = go.Figure(
        go.Bar(
            x=df["rating"],
            y=df["symbol"],
            orientation="h",
            marker={"color": charts.series(0), "cornerradius": 4},
            text=df["rating"].map(lambda r: f"{r}/10"),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x}/10<extra></extra>",
            name="Rating",
        )
    )
    charts.base_layout(
        fig,
        "Rating (1 = weakest … 10 = strongest in the universe)",
        height=max(260, 28 * len(df) + 60),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 10.8], dtick=1, showgrid=True, gridcolor=charts.theme()["grid"])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    charts.show(fig)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=10, format="%d/10"),
            "price": st.column_config.NumberColumn(format="%.2f"),
            "day_%": st.column_config.NumberColumn(format="%+.2f%%"),
            f"P(beat {d['benchmark']})": st.column_config.NumberColumn(format="percent"),
            "P(up, 1 mo)": st.column_config.NumberColumn(format="percent"),
        },
    )
    method_label = {"factors": "the factor rule", "model": "the stock model", "blend": "a blend of both"}
    st.caption(f"Ranked by {method_label.get(d['method'], d['method'])}.")
    for note in d["notes"]:
        st.info(note, icon=":material/info:")
    if d["model_verdict"]:
        st.caption("Stock model: " + d["model_verdict"])
    with st.expander("Factor z-scores (table view)"):
        z = pd.DataFrame(
            [
                {"symbol": p["symbol"], **{FACTOR_LABELS[k]: v for k, v in p["factor_z"].items()}}
                for p in d["picks"]
            ]
        )
        st.dataframe(z, hide_index=True)
    st.caption(d["methodology"])
    st.caption(d["disclaimer"])
    if d["skipped"]:
        st.caption("Skipped: " + "; ".join(f"{k} ({v})" for k, v in d["skipped"].items()))

    st.subheader("Email the list")
    with st.form("email"):
        recipients = st.text_input("Recipients (comma-separated; blank = QP_PICKS_RECIPIENTS)")
        allow = st.checkbox("Send even if data is synthetic (the email is clearly marked)")
        send = st.form_submit_button("Send email", icon=":material/mail:")
    if send:
        payload = {"top_n": top_n, "allow_synthetic": allow}
        emails = [e.strip() for e in recipients.split(",") if e.strip()]
        if emails:
            payload["recipients"] = emails
        out = guarded(lambda: api().post("/picks/email", payload), "email")
        if out:
            st.success(
                f"Sent “{out['subject']}” to {', '.join(out['sent_to'])}", icon=":material/mark_email_read:"
            )
