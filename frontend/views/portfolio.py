"""Risk Laboratory: holdings, VaR/CVaR, performance ratios, correlation and the efficient frontier."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, money, num, pct

DEFAULT = pd.DataFrame(
    {
        "symbol": ["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
        "quantity": [20.0, 10.0, 15.0, 12.0, 25.0],
        "cost_basis": [180.0, 380.0, 110.0, 190.0, 105.0],
    }
)


def _holdings_editor() -> list[dict]:
    saved = guarded(lambda: api().get("/portfolios"), "portfolios") or []
    names = ["(new portfolio)"] + [f"{p['id']}: {p['name']}" for p in saved]
    choice = st.selectbox("Portfolio", names)
    if choice != "(new portfolio)":
        pid = int(choice.split(":")[0])
        chosen = next(p for p in saved if p["id"] == pid)
        base = pd.DataFrame(chosen["holdings"])
        st.session_state["portfolio_id"] = pid
        name_default = chosen["name"]
    else:
        base = DEFAULT
        st.session_state["portfolio_id"] = None
        name_default = "Core holdings"
    edited = st.data_editor(
        base,
        num_rows="dynamic",
        hide_index=True,
        key=f"editor_{choice}",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", required=True, max_chars=15),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0001, required=True),
            "cost_basis": st.column_config.NumberColumn("Cost basis", min_value=0.0, format="%.2f"),
        },
    )
    holdings = [
        {
            "symbol": str(r["symbol"]).strip().upper(),
            "quantity": float(r["quantity"]),
            "cost_basis": None if pd.isna(r.get("cost_basis")) else float(r["cost_basis"]),
        }
        for _, r in edited.dropna(subset=["symbol", "quantity"]).iterrows()
        if str(r["symbol"]).strip()
    ]
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Name", name_default)
    if c2.button("Save portfolio", width="stretch"):
        payload = {"name": name, "holdings": holdings}
        pid = st.session_state.get("portfolio_id")
        res = guarded(
            lambda: api().put(f"/portfolios/{pid}", payload) if pid else api().post("/portfolios", payload),
            "save",
        )
        if res:
            st.toast(f"Saved portfolio {res['name']}", icon=":material/save:")
    return holdings


def render() -> None:
    st.title("Risk Laboratory")
    holdings = _holdings_editor()
    with st.form("risk"):
        c = st.columns(5)
        lookback = c[0].select_slider(
            "History",
            [180, 365, 730, 1095, 1825],
            value=730,
            format_func=lambda d: f"{d // 365}y" if d >= 365 else f"{d}d",
        )
        conf = c[1].select_slider(
            "Confidence", [0.90, 0.95, 0.975, 0.99], value=0.95, format_func=lambda v: f"{v:.1%}"
        )
        horizon = c[2].number_input("Horizon (days)", 1, 30, 1)
        cov = c[3].selectbox(
            "Covariance",
            ["ledoit_wolf", "sample"],
            format_func=lambda v: "Ledoit-Wolf shrinkage" if v == "ledoit_wolf" else "Sample",
        )
        max_w = c[4].slider("Max weight", 0.1, 1.0, 1.0, 0.05)
        go_ = st.form_submit_button("Run risk analysis", type="primary")
    if not go_:
        return
    if not holdings:
        st.warning("Add at least one holding.")
        return
    payload = {
        "holdings": holdings,
        "lookback_days": lookback,
        "confidence": conf,
        "horizon_days": int(horizon),
        "covariance": cov,
        "max_weight": max_w,
        "seed": 11,
    }
    res = guarded(lambda: api().post("/portfolio/analyze", payload), "risk analysis")
    if not res:
        return
    d = res["data"]
    composite_badges(res["meta"])
    for w in d["warnings"]:
        st.warning(w, icon=":material/warning:")
    m = d["metrics"]
    k = st.columns(5)
    k[0].metric("Portfolio value", money(d["portfolio_value"]))
    k[1].metric("Annual return", pct(m["annual_return"]))
    k[2].metric("Annual volatility", pct(m["annual_volatility"]))
    k[3].metric(
        "Sharpe / Sortino",
        f"{num(m['sharpe'])} / {num(m['sortino'])}",
        help=f"Risk-free {pct(m['risk_free_rate'])} (3M Treasury)",
    )
    k[4].metric("Max drawdown", pct(m["max_drawdown"]), help=f"β vs {m['benchmark']}: {num(m['beta'])}")

    var = pd.DataFrame(d["var"])
    var["method"] = var["method"].map(
        {
            "historical": "Historical",
            "parametric": "Parametric (normal)",
            "cornish_fisher": "Cornish-Fisher",
            "monte_carlo": "Monte Carlo",
        }
    )
    st.subheader(f"{conf:.1%} {int(horizon)}-day Value at Risk")
    st.dataframe(
        var[["method", "var_pct", "var_amount", "cvar_pct", "cvar_amount"]],
        hide_index=True,
        column_config={
            "var_pct": st.column_config.NumberColumn("VaR %", format="percent"),
            "cvar_pct": st.column_config.NumberColumn("CVaR %", format="percent"),
            "var_amount": st.column_config.NumberColumn("VaR $", format="dollar"),
            "cvar_amount": st.column_config.NumberColumn("CVaR $", format="dollar"),
        },
    )

    left, right = st.columns(2, gap="large")
    front = d.get("frontier")
    with left:
        if front:
            t = charts.theme()
            fig = go.Figure()
            cloud = front["random_portfolios"]
            fig.add_trace(
                go.Scatter(
                    x=[v * 100 for v, _ in cloud],
                    y=[r * 100 for _, r in cloud],
                    mode="markers",
                    name="Random portfolios",
                    marker={"size": 5, "color": t["muted"], "opacity": 0.35},
                    hoverinfo="skip",
                )
            )
            pts = front["points"]
            fig.add_trace(
                go.Scatter(
                    x=[p["volatility"] * 100 for p in pts],
                    y=[p["expected_return"] * 100 for p in pts],
                    mode="lines",
                    name="Efficient frontier",
                    line={"color": charts.series(0), "width": 2},
                    hovertemplate="σ %{x:.2f}% · μ %{y:.2f}%<extra>frontier</extra>",
                )
            )
            # Scatter-type charts carry at most three categorical hues (blue, orange, aqua validate all-pairs);
            # "Current" uses neutral ink and is identified by marker shape + direct label.
            specials = [
                ("max_sharpe", "Max Sharpe", "star", charts.series(1)),
                ("min_variance", "Min variance", "diamond", charts.series(2)),
                ("current", "Current", "circle", t["ink2"]),
            ]
            for key, label, symbol, color in specials:
                p = front[key]
                fig.add_trace(
                    go.Scatter(
                        x=[p["volatility"] * 100],
                        y=[p["expected_return"] * 100],
                        mode="markers+text",
                        name=label,
                        text=[label],
                        textposition="top center",
                        marker={
                            "size": 13,
                            "symbol": symbol,
                            "color": color,
                            "line": {"width": 2, "color": t["surface"]},
                        },
                        hovertemplate=label + ": σ %{x:.2f}% · μ %{y:.2f}%<extra></extra>",
                    )
                )
            charts.base_layout(fig, "Efficient frontier (annualised)", height=420)
            fig.update_xaxes(title="Volatility (%)", ticksuffix="%", showgrid=True, gridcolor=t["grid"])
            fig.update_yaxes(title="Expected return (%)", ticksuffix="%")
            charts.show(fig)
    with right:
        corr = d["correlation"]
        fig = go.Figure(
            go.Heatmap(
                z=corr["matrix"],
                x=corr["symbols"],
                y=corr["symbols"],
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale=charts.diverging_scale(),
                xgap=2,
                ygap=2,
                text=[[f"{v:.2f}" for v in row] for row in corr["matrix"]],
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
                colorbar={"title": "ρ", "thickness": 12},
            )
        )
        charts.base_layout(fig, "Correlation of daily returns", height=420)
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        charts.show(fig)

    hist = pd.DataFrame(d["value_history"])
    fig = go.Figure(
        go.Scatter(
            x=hist["on"],
            y=hist["value"],
            mode="lines",
            line={"color": charts.series(0), "width": 2},
            name="Value",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    charts.base_layout(
        fig,
        "Current holdings valued over the lookback (buy-and-hold)",
        height=300,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(tickprefix="$")
    charts.show(fig)

    pos = pd.DataFrame(d["positions"])
    st.dataframe(
        pos,
        hide_index=True,
        column_config={
            "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1, format="percent"),
            "risk_contribution": st.column_config.ProgressColumn(
                "Risk share", min_value=0, max_value=1, format="percent"
            ),
            "annual_return": st.column_config.NumberColumn(format="percent"),
            "annual_volatility": st.column_config.NumberColumn(format="percent"),
            "market_value": st.column_config.NumberColumn(format="dollar"),
            "unrealized_pnl": st.column_config.NumberColumn(format="dollar"),
        },
    )
    if front:
        with st.expander("Optimal weights (table view)"):
            rows = []
            for key, label in [
                ("max_sharpe", "Max Sharpe"),
                ("min_variance", "Min variance"),
                ("equal_weight", "Equal weight"),
                ("current", "Current"),
            ]:
                p = front[key]
                rows.append(
                    {
                        "portfolio": label,
                        "return": p["expected_return"],
                        "volatility": p["volatility"],
                        "sharpe": p["sharpe"],
                        **p["weights"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True)
