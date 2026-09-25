"""Options Lab: price chart, BSM calculator with live inputs, option chain, smile and 3-D surface."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, num, pct, status_badge


def _price_chart(symbol: str) -> None:
    c1, c2 = st.columns([1, 1])
    interval = c1.segmented_control("Interval", ["5m", "1h", "1d", "1wk"], default="1d", key="opt_interval")
    lookback = c2.segmented_control(
        "Lookback", ["5D", "1M", "6M", "1Y", "5Y"], default="6M", key="opt_lookback"
    )
    days = {"5D": 5, "1M": 30, "6M": 182, "1Y": 365, "5Y": 1825}[lookback or "6M"]
    if interval == "5m":
        days = min(days, 5)
    hist = guarded(
        lambda: api().get(f"/market/history/{symbol}", interval=interval or "1d", lookback_days=days),
        "history",
    )
    if not hist or not hist["data"]["bars"]:
        return
    df = pd.DataFrame(hist["data"]["bars"])
    t = charts.theme()
    fig = go.Figure(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color=t["series"][0],
            decreasing_line_color=t["series"][1],
            increasing_fillcolor=t["series"][0],
            decreasing_fillcolor=t["series"][1],
        )
    )
    charts.base_layout(
        fig, f"{symbol} · {interval}", height=380, xaxis_rangeslider_visible=False, showlegend=False
    )
    fig.update_yaxes(title="Price")
    charts.show(fig)
    status_badge(hist["meta"], label="Bars")
    with st.expander("Table view"):
        st.dataframe(df.tail(200), hide_index=True)


def _calculator(symbol: str) -> None:
    st.subheader("Black-Scholes-Merton calculator")
    st.caption(
        "Leave spot / rate / volatility / dividend blank to use the live quote, Treasury curve, live smile and trailing yield."
    )
    with st.form("bsm"):
        c = st.columns(4)
        kind = c[0].segmented_control("Type", ["call", "put"], default="call")
        strike = c[1].number_input(
            "Strike", min_value=0.01, value=float(st.session_state.get("bsm_strike", 100.0))
        )
        expiry = c[2].date_input(
            "Expiration", value=date.today() + timedelta(days=30), min_value=date.today() + timedelta(days=1)
        )
        market_price = c[3].number_input("Market price (solve IV)", min_value=0.0, value=0.0)
        o = st.columns(4)
        spot = o[0].number_input("Spot override", min_value=0.0, value=0.0)
        vol = o[1].number_input("Vol override (%)", min_value=0.0, max_value=500.0, value=0.0)
        rate = o[2].number_input(
            "Rate override (%)", min_value=-5.0, max_value=50.0, value=0.0, help="0 = use live curve"
        )
        div = o[3].number_input("Dividend yield override (%)", min_value=0.0, max_value=50.0, value=0.0)
        submitted = st.form_submit_button("Price option", type="primary")
    if not submitted:
        return
    payload = {"symbol": symbol, "kind": kind or "call", "strike": strike, "expiration": expiry.isoformat()}
    if spot > 0:
        payload["spot"] = spot
    if vol > 0:
        payload["volatility"] = vol / 100
    if rate != 0:
        payload["rate"] = rate / 100
    if div > 0:
        payload["dividend_yield"] = div / 100
    if market_price > 0:
        payload["market_price"] = market_price
    res = guarded(lambda: api().post("/options/price", payload), "pricing")
    if not res:
        return
    d = res["data"]
    g, i = d["greeks"], d["inputs"]
    composite_badges(res["meta"])
    m = st.columns(4)
    m[0].metric(f"{i['kind'].title()} value", num(g["price"], 4))
    m[1].metric("Opposite side", num(d["counterpart_price"], 4))
    m[2].metric("Volatility", pct(i["volatility"]), help=i["volatility_source"])
    m[3].metric("Implied vol (market)", pct(d["implied_volatility"]) if d["implied_volatility"] else "—")
    greeks = pd.DataFrame(
        {
            "Greek": [
                "Delta",
                "Gamma",
                "Vega (per 1 vol pt)",
                "Theta (per day)",
                "Rho (per 1%)",
                "Vanna",
                "Vomma",
                "Charm (per day)",
            ],
            "Value": [
                g["delta"],
                g["gamma"],
                g["vega_per_pct"],
                g["theta_per_day"],
                g["rho_per_pct"],
                g["vanna"],
                g["vomma"],
                g["charm_per_day"],
            ],
        }
    )
    st.dataframe(
        greeks, hide_index=True, column_config={"Value": st.column_config.NumberColumn(format="%.6f")}
    )
    st.caption(
        f"Spot {num(i['spot'])} ({i['spot_source']}) · r {pct(i['rate'], 3)} ({i['rate_source']}) · q {pct(i['dividend_yield'])} · T {i['years_to_expiry']:.4f}y"
    )


def _surface(symbol: str) -> None:
    res = guarded(lambda: api().get(f"/options/{symbol}/surface", max_expirations=8), "vol surface")
    if not res:
        return
    s = res["data"]
    composite_badges(res["meta"])
    if not s["smiles"]:
        st.info("No usable option quotes to build a surface.")
        return
    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure()
        for i, smile in enumerate(
            s["smiles"][:3]
        ):  # ≤3 hues for overlapping series; the rest via table/surface
            pts = smile["points"]
            fig.add_trace(
                go.Scatter(
                    x=[p["moneyness"] for p in pts],
                    y=[p["iv"] * 100 for p in pts],
                    mode="lines+markers",
                    name=smile["expiration"],
                    line={"color": charts.series(i), "width": 2},
                    marker={"size": 8, "symbol": ["circle", "square", "diamond"][i]},
                    hovertemplate="K/S %{x:.3f}<br>IV %{y:.2f}%<extra>" + smile["expiration"] + "</extra>",
                )
            )
        charts.base_layout(
            fig, "Volatility smile (nearest expiries, OTM quotes)", height=380, hovermode="x unified"
        )
        fig.update_xaxes(title="Moneyness K/S")
        fig.update_yaxes(title="Implied volatility (%)", ticksuffix="%")
        charts.show(fig)
    with right:
        z = [[None if v is None else v * 100 for v in row] for row in s["iv_grid"]]
        fig = go.Figure(
            go.Surface(
                x=s["moneyness_grid"],
                y=[round(y * 365) for y in s["years"]],
                z=z,
                colorscale=charts.sequential_scale(),
                colorbar={"title": "IV %", "thickness": 12},
                hovertemplate="K/S %{x:.3f}<br>%{y} days<br>IV %{z:.2f}%<extra></extra>",
            )
        )
        charts.base_layout(fig, "Implied-volatility surface", height=380)
        fig.update_layout(
            scene={"xaxis_title": "K/S", "yaxis_title": "Days to expiry", "zaxis_title": "IV %"}
        )
        charts.show(fig)
    term = pd.DataFrame(
        [
            {
                "expiration": sm["expiration"],
                "days": round(sm["years"] * 365, 1),
                "atm_iv_%": None if sm["atm_iv"] is None else round(sm["atm_iv"] * 100, 2),
                "skew_90_110_pts": None if sm["skew_90_110"] is None else round(sm["skew_90_110"] * 100, 2),
                "points": len(sm["points"]),
            }
            for sm in s["smiles"]
        ]
    )
    st.dataframe(term, hide_index=True)
    st.caption(
        f"{s['points_used']} quotes used · {s['points_rejected']} rejected (stale/one-sided/too wide/expired) · q = {pct(s['dividend_yield'])}"
    )


def _chain(symbol: str) -> None:
    res = guarded(lambda: api().get(f"/options/{symbol}/chain", max_expirations=6), "option chain")
    if not res:
        return
    chain = res["data"]
    composite_badges(res["meta"])
    exps = sorted({c["expiration"] for c in chain["contracts"]})
    if not exps:
        st.info("Empty chain.")
        return
    exp = st.selectbox("Expiration", exps)
    df = pd.DataFrame([c for c in chain["contracts"] if c["expiration"] == exp])
    spot = chain["underlying_price"]
    cols = [
        "strike",
        "bid",
        "ask",
        "mid",
        "last",
        "volume",
        "open_interest",
        "implied_volatility",
        "model_iv",
        "delta",
        "gamma",
        "theta_per_day",
        "vega_per_pct",
    ]
    calls = df[df["kind"] == "call"][cols].sort_values("strike")
    puts = df[df["kind"] == "put"][cols].sort_values("strike")
    st.caption(f"Underlying {num(spot)} · as of {chain['as_of'][:19].replace('T', ' ')} UTC")
    cfg = {
        c: st.column_config.NumberColumn(format="%.4f")
        for c in ("implied_volatility", "model_iv", "delta", "gamma", "theta_per_day", "vega_per_pct")
    }
    a, b = st.columns(2)
    a.markdown("**Calls**")
    a.dataframe(calls, hide_index=True, column_config=cfg, height=420)
    b.markdown("**Puts**")
    b.dataframe(puts, hide_index=True, column_config=cfg, height=420)


def render() -> None:
    st.title("Options Lab")
    symbol = (
        st.text_input("Underlying", st.session_state.get("opt_symbol", "AAPL"), key="opt_symbol_input")
        .strip()
        .upper()
        or "AAPL"
    )
    st.session_state["opt_symbol"] = symbol
    quote = guarded(lambda: api().get(f"/market/quote/{symbol}"), "quote")
    if quote:
        q = quote["data"]
        a, b = st.columns([1, 2], vertical_alignment="bottom")
        a.metric(
            q.get("name") or symbol,
            f"{q['price']:,.2f}",
            None if q.get("change_percent") is None else f"{q['change_percent']:+.2f}%",
        )
        b.markdown(
            f"Bid **{num(q.get('bid'))}** × Ask **{num(q.get('ask'))}** · Day range **{num(q.get('day_low'))} – {num(q.get('day_high'))}**"
            f" · Volume **{num(q.get('volume'), 0)}**"
        )
        status_badge(quote["meta"], label="Quote")
        st.session_state.setdefault("bsm_strike", round(q["price"]))
    # A segmented control (not st.tabs) so only the selected panel fetches data.
    views = {
        "Chart": _price_chart,
        "BSM calculator": _calculator,
        "Volatility surface": _surface,
        "Option chain": _chain,
    }
    choice = st.segmented_control(
        "View", list(views), default="Chart", key="opt_view", label_visibility="collapsed"
    )
    views[choice or "Chart"](symbol)
