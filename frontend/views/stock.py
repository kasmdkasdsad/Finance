"""Stock Intelligence: one ticker, everything the platform knows, with honest uncertainty."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, money, num, pct

HORIZON_LABEL = {5: "1 week", 21: "1 month", 63: "3 months"}
TECHNICAL_ROWS: list[tuple[str, str, str]] = [  # (key, label, kind)
    ("price", "Price", "money"),
    ("sma20", "20-day average", "money"),
    ("sma50", "50-day average", "money"),
    ("sma200", "200-day average", "money"),
    ("rsi_14", "RSI (14)", "num"),
    ("high_52w", "52-week high", "money"),
    ("low_52w", "52-week low", "money"),
    ("from_high_52w", "Below 52-week high", "pct"),
    ("return_1m", "Return 1 month", "pct"),
    ("return_3m", "Return 3 months", "pct"),
    ("return_6m", "Return 6 months", "pct"),
    ("return_1y", "Return 1 year", "pct"),
    ("volatility_3m", "Volatility (3 months, annualised)", "pct"),
    ("max_drawdown_1y", "Worst drawdown (1 year)", "pct"),
    ("beta", "Beta (Blume-adjusted)", "num"),
    ("avg_volume_20d", "Average volume (20 days)", "int"),
]


def _md(text: str) -> str:
    """Escape ``$`` so Streamlit markdown does not read a pair of prices as a LaTeX formula."""
    return text.replace("$", "\\$")


def _technicals_table(tech: dict[str, Any]) -> pd.DataFrame:
    def fmt(value: Any, kind: str) -> str:
        if value is None:
            return "—"
        if kind == "money":
            return money(value)
        if kind == "pct":
            return pct(value, 1, signed=True)
        if kind == "int":
            return f"{value:,.0f}"
        return num(value, 2)

    return pd.DataFrame(
        [{"measure": label, "value": fmt(tech.get(key), kind)} for key, label, kind in TECHNICAL_ROWS]
    )


def _fan_chart(report: dict[str, Any]) -> None:
    """Last year of closes with 50/200-day averages, then the forecast cone (50% and 90% bands)."""
    t = charts.theme()
    hist = pd.DataFrame(report["chart"])
    cone = pd.DataFrame(report["forecast"]["cone"])
    hue = charts.series(0)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["close"],
            name="Close",
            line={"color": hue, "width": 2},
            hovertemplate="%{x}: $%{y:,.2f}<extra>close</extra>",
        )
    )
    for col, name, idx in (("sma50", "50-day average", 1), ("sma200", "200-day average", 2)):
        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist[col],
                name=name,
                line={"color": charts.series(idx), "width": 2},
                hovertemplate="%{x}: $%{y:,.2f}<extra>" + name + "</extra>",
            )
        )
    # Bands: one hue, lighter = wider. Anchored to today's price so the cone starts at the last close.
    start = pd.DataFrame(
        [
            {
                "date": hist["date"].iloc[-1],
                **dict.fromkeys(("p05", "p25", "p50", "p75", "p95"), report["forecast"]["spot"]),
            }
        ]
    )
    c = pd.concat([start, cone[["date", "p05", "p25", "p50", "p75", "p95"]]], ignore_index=True)
    for lo, hi, alpha, name in (("p05", "p95", 0.14, "90% range"), ("p25", "p75", 0.28, "50% range")):
        fig.add_trace(go.Scatter(x=c["date"], y=c[hi], line={"width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=c["date"],
                y=c[lo],
                fill="tonexty",
                fillcolor=charts.rgba(hue, alpha),
                line={"width": 0},
                name=name,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=c["date"],
            y=c["p50"],
            name="Median forecast",
            line={"color": t["ink2"], "width": 2},
            customdata=c[["p05", "p95"]].to_numpy(),
            hovertemplate="%{x}: median $%{y:,.2f} · 90% $%{customdata[0]:,.2f}–$%{customdata[1]:,.2f}<extra></extra>",
        )
    )
    charts.base_layout(fig, f"{report['symbol']}: last year and the next 63 trading days", height=420)
    fig.update_yaxes(title="Price ($)", tickprefix="$")
    charts.show(fig, key="stock_fan")


def _horizon_table(report: dict[str, Any]) -> None:
    rows = []
    for h in report["forecast"]["horizons"]:
        imp = h["implied"] or {}
        rows.append(
            {
                "horizon": f"{HORIZON_LABEL.get(h['days'], str(h['days']) + ' days')} ({h['target_date']})",
                "5%": h["band"]["p05"],
                "25%": h["band"]["p25"],
                "median": h["band"]["p50"],
                "75%": h["band"]["p75"],
                "95%": h["band"]["p95"],
                "P(higher)": h["prob_up"],
                **({"P(above target)": h["prob_above_target"]} if report["forecast"]["target"] else {}),
                "options: ±1σ move": imp.get("move_1sd"),
                "options: P(higher)": imp.get("prob_up"),
            }
        )
    money_col = st.column_config.NumberColumn(format="$%.2f")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            **dict.fromkeys(("5%", "25%", "median", "75%", "95%"), money_col),
            "P(higher)": st.column_config.NumberColumn(format="percent"),
            "P(above target)": st.column_config.NumberColumn(format="percent"),
            "options: ±1σ move": st.column_config.NumberColumn(format="percent"),
            "options: P(higher)": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Model ranges come from a GARCH volatility model with bootstrapped historical shocks (blended with options-"
        "implied volatility when available), the next earnings jump and a CAPM drift. Options columns are "
        "risk-neutral (they include the price of insurance), from the expiry nearest each horizon."
    )


def _calibration(symbol: str) -> None:
    if not st.button("Test the forecaster on this stock's history", icon=":material/fact_check:"):
        return
    with st.spinner("Replaying five years day by day…"):
        res = guarded(
            lambda: api().get(f"/forecast/{symbol}", horizons="5,21", options=False, calibrate=True),
            "calibration",
        )
    if not res or not res["data"]["calibration"]:
        return
    cal = res["data"]["calibration"]
    m = st.columns(4)
    m[0].metric(
        "Inside 90% range", pct(cal["coverage_90"], 0), help="Target 90%. Much lower = ranges too narrow."
    )
    m[1].metric("Inside 50% range", pct(cal["coverage_50"], 0), help="Target 50%.")
    m[2].metric(
        "Brier skill (direction)", num(cal["brier_skill"], 3), help="> 0 beats the base rate; ≈ 0 is typical."
    )
    m[3].metric("Realised / forecast volatility", num(cal["volatility_ratio"], 2), help="1.0 = unbiased.")
    hist = cal["pit_histogram"]
    fig = go.Figure(
        go.Bar(
            x=[f"{i * 10}–{i * 10 + 10}%" for i in range(len(hist))],
            y=[v * 100 for v in hist],
            marker={"color": charts.series(0), "cornerradius": 4},
            name="Share of outcomes",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )
    charts.reference_line(fig, 100 / len(hist), "perfectly calibrated")
    charts.base_layout(
        fig,
        f"Where outcomes landed inside the {cal['horizon']}-day forecast ({cal['n']} forecasts)",
        height=300,
        showlegend=False,
    )
    fig.update_xaxes(title="Forecast percentile of the realised price")
    fig.update_yaxes(title="Share (%)", ticksuffix="%")
    charts.show(fig, key="stock_pit")
    st.caption(
        f"{cal['start']} → {cal['end']}; forecasts every 5 days overlap, so this is about {cal['effective_n']:.0f} "
        "independent tests. A flat histogram means the ranges were honest; a U shape means they were too narrow."
    )


def _earnings(symbol: str, fc: dict[str, Any]) -> None:
    st.subheader("Earnings")
    e = fc.get("earnings")
    res = guarded(lambda: api().get(f"/stocks/{symbol}/earnings"), "earnings history")
    if not e and not res:
        st.caption("No earnings history is available for this ticker.")
        return
    k = st.columns(4)
    if e and e["next_date"]:
        when = f"{e['next_date']}" + (f" (in {e['sessions_ahead']} sessions)" if e["sessions_ahead"] else "")
        k[0].metric("Next reaction day", when, e["source"], delta_color="off", delta_arrow="off")
    else:
        k[0].metric("Next reaction day", "—")
    k[1].metric(
        "Typical earnings move",
        pct(e["typical_move"] if e else None, 1),
        help="Root-mean-square of past earnings-day returns",
    )
    k[2].metric("Past releases used", e["events_used"] if e else 0)
    k[3].metric(
        "In the forecast",
        "jump simulated" if e and e["modelled"] else "not in horizon" if e else "—",
        help="Earnings days are left out of the volatility model and added back as jumps on the day they are due",
    )
    if res and res["data"]["reactions"]:
        rx = pd.DataFrame(res["data"]["reactions"])
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=rx["reaction_date"],
                y=rx["stock_return"] * 100,
                name="Stock",
                marker={"color": charts.series(0), "cornerradius": 3},
                hovertemplate="%{x}: %{y:+.1f}%<extra>stock</extra>",
            )
        )
        if rx["abnormal_return"].notna().any():
            fig.add_trace(
                go.Bar(
                    x=rx["reaction_date"],
                    y=rx["abnormal_return"] * 100,
                    name="vs market",
                    marker={"color": charts.series(1), "cornerradius": 3},
                    hovertemplate="%{x}: %{y:+.1f}%<extra>vs market</extra>",
                )
            )
        charts.reference_line(fig, 0.0)
        charts.base_layout(fig, "Price reaction on each earnings day", height=300)
        fig.update_yaxes(title="Return (%)", ticksuffix="%")
        charts.show(fig, key="stock_earnings")
        st.caption(
            "Release times come from SEC 8-K filings (item 2.02); a release after the close is priced the next day."
        )


def render() -> None:
    st.title("Stock Intelligence")
    c = st.columns([2, 1, 1, 1, 1], vertical_alignment="bottom")
    symbol = c[0].text_input("Ticker", st.session_state.get("stock_symbol", "AAPL"), key="stock_symbol_input")
    symbol = symbol.strip().upper() or "AAPL"
    st.session_state["stock_symbol"] = symbol
    target = c[1].number_input(
        "Target price (optional)", min_value=0.0, value=0.0, step=1.0, key="stock_target"
    )
    with_model = c[2].toggle("Stock model", value=True, key="stock_with_model")
    with_dcf = c[3].toggle("DCF", value=True, key="stock_with_dcf")
    with_options = c[4].toggle("Options", value=True, key="stock_with_options")
    with st.spinner(f"Building the {symbol} report…"):
        res = guarded(
            lambda: api().get(
                f"/stocks/{symbol}/report",
                target=target or None,
                model=with_model,
                valuation=with_dcf,
                options=with_options,
            ),
            "stock report",
        )
    if not res:
        return
    r = res["data"]
    composite_badges(res["meta"])
    if r["data_status"] == "synthetic":
        st.warning(
            "Live prices are unavailable: this report is built on SYNTHETIC data and is not real.",
            icon=":material/science:",
        )
    st.markdown(f"#### {r['name'] or symbol}")
    for line in r["summary"]:
        st.markdown("- " + _md(line))

    tech, fc = r["technicals"], r["forecast"]
    month = next((h for h in fc["horizons"] if h["days"] == 21), fc["horizons"][0])
    m = st.columns(5)
    m[0].metric(
        "Price",
        money(tech["price"]),
        None if tech["change_percent"] is None else f"{tech['change_percent']:+.2f}%",
    )
    m[1].metric(
        "1-month 90% range", _md(f"{money(month['band']['p05'], 0)}–{money(month['band']['p95'], 0)}")
    )
    m[2].metric("P(higher in 1 month)", pct(month["prob_up"], 0))
    model = r["model"]
    m[3].metric(
        f"P(beat {model['benchmark']}, 1 mo)" if model else "Stock model",
        pct(model["prob_outperform"], 0) if model else "—",
        None if not model else f"rank {model['rank']} of {model['universe_size']}",
        delta_color="off",
        delta_arrow="off",
    )
    val = r["valuation"]
    m[4].metric(
        "DCF upside", pct(val["upside"], 0, signed=True) if val and val["upside"] is not None else "—"
    )

    _fan_chart(r)
    _horizon_table(r)
    if target:
        st.caption("P(above target) uses the target price you entered.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Volatility & drift")
        v, d = fc["volatility"], fc["drift"]
        k = st.columns(3)
        k[0].metric("Volatility now", pct(v["current_vol_annual"], 0), help="GARCH one-day-ahead, annualised")
        k[1].metric("Long-run volatility", pct(v["long_run_vol_annual"], 0))
        hl = v["half_life_days"]
        k[2].metric(
            "Shock half-life",
            "—" if hl is None else ("<1 day" if hl < 1 else f"{hl:.0f} days"),
            help="How long a volatility shock takes to fade by half (GARCH persistence)",
        )
        k = st.columns(3)
        k[0].metric("Expected return (yr)", pct(d["annual_expected_return"], 1), help=d["method"])
        k[1].metric("Beta", num(d["beta"]))
        k[2].metric(
            "Fat tails (ν)",
            "≈ normal" if v["nu"] is None or v["nu"] > 30 else f"{v['nu']:.1f}",
            help="Student-t degrees of freedom; lower = fatter tails",
        )
        rv = fc["realized_vol"]
        st.caption(
            f"Realised 1-month volatility: close-to-close {pct(rv['close_to_close_21d'], 0)} · "
            f"Parkinson {pct(rv['parkinson_21d'], 0)} · Garman-Klass {pct(rv['garman_klass_21d'], 0)}"
        )
        if v.get("iv_weight"):
            k = st.columns(3)
            k[0].metric(
                "GARCH, 1 month", pct(v["garch_vol_annual_21d"], 0), help="Ex-earnings historical model"
            )
            k[1].metric(
                "Options, 1 month",
                pct(v["implied_vol_annual_21d"], 0),
                help="At-the-money implied volatility near 21 sessions (includes any earnings in the option's life)",
            )
            k[2].metric(
                "Used by the forecast",
                pct(v["blended_vol_annual_21d"], 0),
                help=f"{v['iv_weight']:.0%} options (ex earnings, divided by a {v['variance_premium']:.2f} variance "
                "risk premium) + the rest GARCH",
            )
    with right:
        st.subheader("Stock model")
        if model:
            if model.get("sector_label"):
                industry = model["sector_label"]
                if model.get("industry_rank"):
                    industry += f" · {model['industry_rank']} of {model['industry_size']} in its industry"
                st.caption(f"{model.get('model_label') or 'Model'} · industry: {industry}")
            k = st.columns(3)
            k[0].metric("Rank", f"{model['rank']} / {model['universe_size']}")
            k[1].metric("Model rating", f"{model['rating']}/10")
            k[2].metric(
                f"vs {model['benchmark']}, 1 mo",
                pct(model["expected_excess_return"], 2, signed=True),
                help="Calibrated expected return in excess of the benchmark over the model horizon",
            )
            (st.success if model["has_skill"] else st.info)(
                model["verdict"], icon=":material/model_training:"
            )
        else:
            st.info("The stock model was not included or could not score this ticker.")

    _earnings(symbol, fc)

    st.subheader("Technicals")
    table = _technicals_table(tech)
    st.dataframe(table, hide_index=True, height=35 * (len(table) + 1) + 3)
    if val:
        st.subheader("Valuation (DCF)")
        k = st.columns(4)
        k[0].metric("Fair value / share", money(val["value_per_share"]))
        k[1].metric("Upside", pct(val["upside"], 0, signed=True) if val["upside"] is not None else "—")
        k[2].metric("WACC", pct(val["wacc"], 1))
        k[3].metric("Terminal value share", pct(val["terminal_value_share"], 0))
        for w in val["warnings"]:
            st.caption(w)

    st.subheader("Track record for this ticker")
    tr = r["track_record"]
    if tr["recent"]:
        st.dataframe(
            pd.DataFrame(tr["recent"])[
                [
                    "made_on",
                    "target_date",
                    "source",
                    "horizon_days",
                    "reference_price",
                    "prob_up",
                    "prob_outperform",
                    "status",
                    "realized_price",
                    "outcome_up",
                    "outcome_outperform",
                    "in_90",
                ]
            ],
            hide_index=True,
            column_config={
                "prob_up": st.column_config.NumberColumn("P(up)", format="percent"),
                "prob_outperform": st.column_config.NumberColumn("P(beat)", format="percent"),
            },
        )
        st.caption(f"{tr['resolved']} graded, {tr['open']} open.")
    else:
        st.caption(
            "No predictions logged for this ticker yet. The ledger logs the universe after each close."
        )
    _calibration(symbol)
    for n in r["notes"]:
        st.caption(n)
    st.caption(r["disclaimer"])
