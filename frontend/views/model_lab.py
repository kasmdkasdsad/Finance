"""Model Lab: is the stock model any good? Out-of-sample skill, backtest, calibration and signal research."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, num, pct

FEATURE_LABELS = {
    "mom_12_1": "12-1m momentum",
    "mom_6_1": "6-1m momentum",
    "mom_3m": "3m momentum",
    "ret_1m": "1m return",
    "ret_5d": "5d return",
    "trend_50_200": "50/200-day trend",
    "px_vs_sma50": "price vs 50-day",
    "high_52w": "near 52w high",
    "vol_63": "3m volatility",
    "vol_ratio": "vol ratio 1m/3m",
    "sharpe_126": "6m Sharpe",
    "rsi_14": "RSI 14",
    "bollinger_b": "Bollinger %B",
    "beta_252": "1y beta",
    "idio_vol_63": "idiosyncratic vol",
    "max_ret_21": "max daily gain 1m",
    "skew_63": "3m skew",
    "volume_trend": "volume trend",
}


def _ic_chart(m: dict[str, Any]) -> None:
    tl = pd.DataFrame(m["ic_timeline"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tl["date"],
            y=tl["rolling"],
            name="Model (63-day avg IC)",
            line={"color": charts.series(0), "width": 2},
            hovertemplate="%{x}: %{y:+.3f}<extra>model</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tl["date"],
            y=tl["baseline_rolling"],
            name="Factor rule (63-day avg IC)",
            line={"color": charts.series(1), "width": 2},
            hovertemplate="%{x}: %{y:+.3f}<extra>factor rule</extra>",
        )
    )
    charts.reference_line(fig, 0.0, "no skill")
    charts.base_layout(fig, "Out-of-sample information coefficient over time", height=320)
    fig.update_yaxes(title="Rank correlation with next-month returns")
    charts.show(fig, key="lab_ic")


def _backtest_chart(m: dict[str, Any]) -> None:
    bt = m["backtest"]
    fig = go.Figure()
    series = (
        ("strategy", "Model's top picks"),
        ("universe", "Equal-weight universe"),
        ("benchmark", m["benchmark"]),
    )
    for i, (key, name) in enumerate(series):
        fig.add_trace(
            go.Scatter(
                x=bt["dates"],
                y=bt[key],
                name=name,
                line={"color": charts.series(i), "width": 2},
                hovertemplate="%{x}: $%{y:.3f}<extra>" + name + "</extra>",
            )
        )
    charts.base_layout(fig, "Growth of $1, out of sample, net of trading costs", height=320)
    fig.update_yaxes(title="Value of $1", tickprefix="$")
    charts.show(fig, key="lab_backtest")


def _buckets_chart(m: dict[str, Any]) -> None:
    b = m["buckets"]
    labels = ["Lowest 20%", "2nd", "3rd", "4th", "Highest 20%"][: len(b)]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[None if v is None else v * 100 for v in b],
            marker={"color": charts.series(0), "cornerradius": 4},
            name="Mean return",
            hovertemplate="%{x}: %{y:+.2f}%<extra></extra>",
        )
    )
    charts.reference_line(fig, 0.0)
    charts.base_layout(
        fig, f"Average {m['horizon']}-day return by prediction quintile", height=300, showlegend=False
    )
    fig.update_yaxes(title="Return (%)", ticksuffix="%")
    charts.show(fig, key="lab_buckets")


def _calibration_chart(bins: list[dict[str, Any]], base: float, title: str, key: str) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": charts.theme()["axis"], "width": 1},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b["probability"] for b in bins],
            y=[b["observed"] for b in bins],
            mode="lines+markers",
            name="Observed frequency",
            line={"color": charts.series(0), "width": 2},
            marker={"size": 9},
            customdata=[b["n"] for b in bins],
            hovertemplate="predicted %{x:.0%} → observed %{y:.0%} (n=%{customdata})<extra></extra>",
        )
    )
    charts.base_layout(fig, title, height=320)
    lo = max(0.0, min(min(b["probability"] for b in bins), min(b["observed"] for b in bins), base) - 0.1)
    hi = min(1.0, max(max(b["probability"] for b in bins), max(b["observed"] for b in bins), base) + 0.1)
    fig.update_xaxes(title="Predicted probability", range=[lo, hi], tickformat=".0%")
    fig.update_yaxes(title="Observed frequency", range=[lo, hi], tickformat=".0%")
    charts.show(fig, key=key)


def _importance_chart(m: dict[str, Any]) -> None:
    imp = sorted(m["importance"], key=lambda i: i["coefficient"])
    t = charts.theme()
    pos, neg = charts.DIVERGING_BLUE[-1], charts.DIVERGING_RED[0]
    fig = go.Figure(
        go.Bar(
            x=[i["coefficient"] for i in imp],
            y=[FEATURE_LABELS.get(i["feature"], i["feature"]) for i in imp],
            orientation="h",
            marker={"color": [pos if i["coefficient"] >= 0 else neg for i in imp], "cornerradius": 4},
            customdata=[i["sign_consistency"] for i in imp],
            hovertemplate="%{y}: %{x:+.4f} (same sign in %{customdata:.0%} of refits)<extra></extra>",
            name="Coefficient",
        )
    )
    charts.base_layout(
        fig,
        "What the model leans on (average coefficient; blue = higher is better)",
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=t["grid"], zeroline=True, zerolinecolor=t["axis"])
    charts.show(fig, key="lab_importance")


def _model_view() -> None:
    c = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
    horizon = c[0].selectbox("Horizon (trading days)", [5, 10, 21, 42, 63], index=2, key="lab_horizon")
    top_k = c[1].number_input("Backtest holds", 1, 15, 5, key="lab_topk")
    refresh = c[2].toggle("Recompute", value=False, key="lab_refresh", help="Ignore today's cached run")
    with st.spinner("Training walk-forward (a few seconds the first time each day)…"):
        res = guarded(
            lambda: api().get("/model/report", horizon=horizon, top_k=top_k, refresh=refresh), "model report"
        )
    if not res:
        return
    m = res["data"]
    composite_badges(res["meta"])
    for w in m["warnings"]:
        st.warning(w, icon=":material/warning:")
    (st.success if m["has_skill"] else st.info)(m["verdict"], icon=":material/model_training:")
    oos, base, bt = m["oos"], m["baseline"], m["backtest"]
    k = st.columns(5)
    k[0].metric(
        "Mean IC (model)",
        f"{oos['mean_ic']:+.3f}",
        f"t = {num(oos['t_stat'], 1)}",
        delta_color="off",
        delta_arrow="off",
    )
    k[1].metric(
        "Mean IC (factor rule)",
        f"{base['mean_ic']:+.3f}",
        f"t = {num(base['t_stat'], 1)}",
        delta_color="off",
        delta_arrow="off",
    )
    k[2].metric(
        "Hit rate vs median",
        pct(oos["hit_rate"], 1),
        help="Above-median calls that finished above the median",
    )
    k[3].metric("Top picks, annual", pct(bt["strategy_metrics"]["annual_return"], 1))
    k[4].metric("Universe, annual", pct(bt["universe_metrics"]["annual_return"], 1))
    st.caption(
        f"Out-of-sample {m['oos_start']} → {m['oos_end']} over {len(m['symbols'])} stocks; {m['retrains']} monthly refits; "
        f"{bt['periods']} non-overlapping {m['horizon']}-day holding periods; {bt['cost_bps']:.0f} bps per trade."
    )
    left, right = st.columns(2, gap="large")
    with left:
        _ic_chart(m)
        _buckets_chart(m)
    with right:
        _backtest_chart(m)
        _calibration_chart(
            m["calibration"],
            m["base_rate"],
            f"Calibration: P(beat {m['benchmark']}) vs what happened",
            "lab_calibration",
        )
    _importance_chart(m)
    st.subheader(f"Live rankings · {m['as_of']}")
    live = pd.DataFrame(m["live"])
    st.dataframe(
        live[["rank", "symbol", "rating", "prob_outperform", "expected_excess_return", "z"]],
        hide_index=True,
        column_config={
            "rating": st.column_config.ProgressColumn("rating", min_value=0, max_value=10, format="%d/10"),
            "prob_outperform": st.column_config.NumberColumn(f"P(beat {m['benchmark']})", format="percent"),
            "expected_excess_return": st.column_config.NumberColumn("expected excess", format="percent"),
            "z": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    with st.expander("Table views (IC timeline, backtest, calibration, importance)"):
        st.dataframe(pd.DataFrame(m["ic_timeline"]), hide_index=True)
        st.dataframe(
            pd.DataFrame({k: bt[k] for k in ("dates", "strategy", "universe", "benchmark")}), hide_index=True
        )
        st.dataframe(pd.DataFrame(m["calibration"]), hide_index=True)
        st.dataframe(pd.DataFrame(m["importance"]), hide_index=True)
    if m["skipped"]:
        st.caption("Left out: " + "; ".join(f"{k} ({v})" for k, v in m["skipped"].items()))
    st.caption(m["disclaimer"])


def _signals_view() -> None:
    horizon = st.selectbox("Quintile horizon (trading days)", [5, 21, 63], index=1, key="lab_signal_horizon")
    with st.spinner("Measuring every signal…"):
        res = guarded(lambda: api().get("/model/research", horizon=horizon), "signal research")
    if not res:
        return
    r = res["data"]
    composite_badges(res["meta"])
    feats = [s["feature"] for s in r["signals"]]
    z = [
        [next((h["mean_ic"] for h in s["by_horizon"] if h["horizon"] == hz), None) for hz in r["horizons"]]
        for s in r["signals"]
    ]
    bound = max(0.05, max(abs(v) for row in z for v in row if v is not None))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{h}d" for h in r["horizons"]],
            y=[FEATURE_LABELS.get(f, f) for f in feats],
            colorscale=charts.diverging_scale(),
            zmin=-bound,
            zmax=bound,
            zmid=0,
            colorbar={"title": "mean IC"},
            hovertemplate="%{y} · %{x}: IC %{z:+.3f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    charts.base_layout(
        fig, "Mean IC by signal and horizon (blue = high values led to higher returns)", height=560
    )
    charts.show(fig, key="lab_heatmap")
    rows = []
    for s in r["signals"]:
        row: dict[str, Any] = {
            "signal": FEATURE_LABELS.get(s["feature"], s["feature"]),
            "definition": s["description"],
        }
        for h in s["by_horizon"]:
            row[f"IC {h['horizon']}d"] = h["mean_ic"]
            row[f"t {h['horizon']}d"] = h["t_stat"]
        row[f"top-bottom {r['main_horizon']}d"] = s["spread"]
        rows.append(row)
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            **{f"IC {h}d": st.column_config.NumberColumn(format="%+.3f") for h in r["horizons"]},
            **{f"t {h}d": st.column_config.NumberColumn(format="%+.1f") for h in r["horizons"]},
            f"top-bottom {r['main_horizon']}d": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(f"{r['start']} → {r['end']}, {r['n_symbols']} stocks. {r['note']}")


def render() -> None:
    st.title("Model Lab")
    st.caption(
        "The stock model is trained walk-forward: every number here comes from predictions made before the returns "
        "they are scored on. If it shows no skill, the platform says so and falls back to the simple factor rule."
    )
    views = {"Stock model": _model_view, "Signal research": _signals_view}
    choice = st.segmented_control(
        "View", list(views), default="Stock model", key="lab_view", label_visibility="collapsed"
    )
    views[choice or "Stock model"]()
