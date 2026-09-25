"""Trading Sandbox: paper accounts traded by a self-learning agent (simulated money only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, money, num, pct
from frontend.views.picks import FACTOR_LABELS

BASE = "/sandbox/accounts"
KIND_ICON = {
    "created": ":material/add_card:",
    "decision": ":material/swap_horiz:",
    "lesson": ":material/school:",
    "skip": ":material/pause_circle:",
    "train": ":material/model_training:",
    "order": ":material/shopping_cart:",
    "reset": ":material/restart_alt:",
    "update": ":material/tune:",
}
VIEWS = ["Performance", "Learning", "Positions", "Trades", "Journal", "Manual order", "Settings"]


def _universe(text: str) -> list[str] | None:
    symbols = [s.strip().upper() for s in text.replace("\n", ",").split(",") if s.strip()]
    return symbols or None


def _strategy_inputs(prefix: str, current: dict[str, Any] | None = None) -> dict[str, Any]:
    cur = current or {}
    signals = ["factors", "model"]
    signal = st.radio(
        "Signal",
        signals,
        index=signals.index(cur.get("signal", "factors")),
        horizontal=True,
        key=f"{prefix}_signal",
        format_func={
            "factors": "Factor rule (re-weights its factors from its own results)",
            "model": "Stock model (walk-forward ridge, retrained monthly)",
        }.get,
    )
    c = st.columns(4)
    strategy: dict[str, Any] = {
        "signal": signal,
        "top_k": c[0].number_input("Names held (top k)", 1, 20, int(cur.get("top_k", 5)), key=f"{prefix}_k"),
        "max_position": c[1].slider(
            "Max per name", 0.05, 1.0, float(cur.get("max_position", 0.25)), 0.05, key=f"{prefix}_maxpos"
        ),
        "cash_buffer": c[2].slider(
            "Cash buffer", 0.0, 0.5, float(cur.get("cash_buffer", 0.02)), 0.01, key=f"{prefix}_buffer"
        ),
        "learning_rate": c[3].slider(
            "Learning rate η",
            0.0,
            5.0,
            float(cur.get("learning_rate", 0.5)),
            0.1,
            key=f"{prefix}_eta",
            help="w ← w·exp(η·IC) after every decision; 0 turns learning off.",
        ),
    }
    d = st.columns(4)
    strategy["slippage_bps"] = d[0].number_input(
        "Slippage (bps)", 0.0, 200.0, float(cur.get("slippage_bps", 5.0)), 1.0, key=f"{prefix}_slip"
    )
    strategy["commission_per_trade"] = d[1].number_input(
        "Commission / trade ($)",
        0.0,
        100.0,
        float(cur.get("commission_per_trade", 0.0)),
        0.5,
        key=f"{prefix}_fee",
    )
    strategy["min_trade_value"] = d[2].number_input(
        "Min trade ($)", 0.0, 100000.0, float(cur.get("min_trade_value", 50.0)), 10.0, key=f"{prefix}_min"
    )
    universe = d[3].text_input(
        "Universe (blank = default list)",
        ", ".join(cur.get("universe") or []),
        key=f"{prefix}_universe",
        help="Comma-separated tickers the agent may trade (at least 3).",
    )
    for field in ("prior_shrink", "weight_floor", "commission_bps"):
        if field in cur:
            strategy[field] = cur[field]
    strategy["universe"] = _universe(universe)
    return strategy


def _create_form() -> None:
    with st.form("sandbox_new"):
        c = st.columns([2, 1, 1])
        name = c[0].text_input("Account name", "Learning agent")
        mode = c[1].selectbox(
            "Mode",
            ["agent", "manual"],
            help="agent = trades by itself every day; manual = you place the orders",
        )
        cash = c[2].number_input("Starting cash ($)", 1000.0, 1e9, 100000.0, 1000.0)
        o = st.columns(2)
        auto = o[0].checkbox("Trade automatically every trading day", True)
        allow = o[1].checkbox(
            "Allow synthetic prices", False, help="Off: the account only ever trades on real market prices."
        )
        strategy = _strategy_inputs("new")
        if st.form_submit_button("Open account", type="primary", icon=":material/add_card:"):
            payload = {
                "name": name,
                "mode": mode,
                "starting_cash": cash,
                "auto_trade": auto,
                "allow_synthetic": allow,
                "strategy": strategy,
            }
            created = guarded(lambda: api().post(BASE, payload), "create account")
            if created:
                st.session_state["sandbox_account"] = created["id"]
                st.rerun()


def _step_result(step: dict[str, Any]) -> None:
    if step["executed"]:
        st.success(
            f"Agent ran for {step['trading_day']}: {len(step['trades'])} trade(s), "
            f"equity {money(step['equity'])} · data {step['data_status'].upper()}",
            icon=":material/smart_toy:",
        )
    else:
        st.info(f"Agent sat out: {step['skipped_reason']}", icon=":material/pause_circle:")
    if step["lessons"]:
        st.markdown("**What it learned** (information coefficient = rank correlation of score vs return)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "factor": lesson["label"],
                        "IC": lesson["ic"],
                        "stocks": lesson["observations"],
                        "weight before": lesson["weight_before"],
                        "weight after": lesson["weight_after"],
                    }
                    for lesson in step["lessons"]
                ]
            ),
            hide_index=True,
            column_config={
                "IC": st.column_config.NumberColumn(format="%+.3f"),
                "weight before": st.column_config.NumberColumn(format="percent"),
                "weight after": st.column_config.NumberColumn(format="percent"),
            },
        )
    if step["candidates"]:
        st.markdown("**How it ranked the universe**")
        st.dataframe(
            pd.DataFrame(step["candidates"]),
            hide_index=True,
            column_config={
                "composite": st.column_config.NumberColumn(format="%+.3f"),
                "rating": st.column_config.ProgressColumn(
                    "rating", min_value=0, max_value=10, format="%d/10"
                ),
                "target_weight": st.column_config.NumberColumn("target", format="percent"),
            },
        )
    if step["excluded"]:
        st.caption("Left out: " + "; ".join(f"{k} ({v})" for k, v in step["excluded"].items()))


def _line_chart(df: pd.DataFrame, columns: list[str], title: str, y_title: str, key: str) -> None:
    fig = go.Figure()
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line={"width": 2, "color": charts.series(i)},
                connectgaps=True,
            )
        )
    charts.base_layout(fig, title, height=340, yaxis_title=y_title, hovermode="x unified")
    charts.show(fig, key=key)


def _performance(account_id: int, benchmark: str) -> None:
    points = guarded(lambda: api().get(f"{BASE}/{account_id}/equity"), "equity history") or []
    if len(points) < 2:
        st.info(
            "The equity curve fills in as the account trades: a snapshot is taken at every agent run and after "
            "every close.",
            icon=":material/show_chart:",
        )
        return
    df = pd.DataFrame(points)
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.set_index("recorded_at")
    idx = pd.DataFrame(index=df.index)
    idx["Strategy"] = df["equity"] / df["equity"].iloc[0] * 100
    base = df["benchmark_price"].dropna()
    cols = ["Strategy"]
    if not base.empty:
        idx[benchmark] = df["benchmark_price"] / base.iloc[0] * 100
        cols.append(benchmark)
    _line_chart(idx, cols, f"Paper equity vs {benchmark} (indexed to 100)", "Index", "sandbox_equity")
    with st.expander("Equity snapshots (table view)"):
        st.dataframe(
            df.reset_index()[["recorded_at", "equity", "cash", "benchmark_price", "data_status"]],
            hide_index=True,
            column_config={
                "equity": st.column_config.NumberColumn(format="$%.2f"),
                "cash": st.column_config.NumberColumn(format="$%.2f"),
                "benchmark_price": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def _weights_chart(prior: dict[str, float], learned: dict[str, float], key: str) -> None:
    factors = list(FACTOR_LABELS)
    labels = [FACTOR_LABELS[f] for f in factors]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[prior.get(f, 0) * 100 for f in factors],
            orientation="h",
            name="Default (prior)",
            marker={"color": charts.theme()["muted"], "cornerradius": 4},
            hovertemplate="%{y}: %{x:.1f}%<extra>prior</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[learned.get(f, 0) * 100 for f in factors],
            orientation="h",
            name="Learned",
            marker={"color": charts.series(0), "cornerradius": 4},
            hovertemplate="%{y}: %{x:.1f}%<extra>learned</extra>",
        )
    )
    charts.base_layout(fig, "Factor weights: default vs learned", height=330, barmode="group")
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=charts.theme()["grid"])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    charts.show(fig, key=key)


def _evolution(rows: list[tuple[Any, dict[str, float]]], title: str, key: str) -> None:
    df = pd.DataFrame(
        [{"date": d, **{FACTOR_LABELS[f]: w * 100 for f, w in weights.items()}} for d, weights in rows]
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    _line_chart(df, [FACTOR_LABELS[f] for f in FACTOR_LABELS], title, "Weight (%)", key)
    with st.expander("Weight history (table view)"):
        st.dataframe(df.round(2))


def _learning(account: dict[str, Any], account_id: int) -> None:
    prior, learned = account["prior_weights"], account["factor_weights"]
    st.markdown(
        f"The agent has scored **{account['periods_learned']}** past decision(s) against what the market did "
        "next. Factors whose scores ranked future winners gain weight; the rest lose it (never below a floor)."
    )
    _weights_chart(prior, learned, "sandbox_weights")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "factor": FACTOR_LABELS[f],
                    "default": prior[f],
                    "learned": learned[f],
                    "change (pts)": (learned[f] - prior[f]) * 100,
                    "IC (EMA)": account["ic_ema"].get(f),
                }
                for f in FACTOR_LABELS
            ]
        ),
        hide_index=True,
        column_config={
            "default": st.column_config.NumberColumn(format="percent"),
            "learned": st.column_config.NumberColumn(format="percent"),
            "change (pts)": st.column_config.NumberColumn(format="%+.1f"),
            "IC (EMA)": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    journal = guarded(lambda: api().get(f"{BASE}/{account_id}/journal", limit=1000), "journal") or []
    lessons = [e for e in reversed(journal) if e["kind"] == "lesson"]
    if lessons:
        rows = [(lessons[0]["created_at"], lessons[0]["details"]["weights_before"])]
        rows += [(e["created_at"], e["details"]["weights_after"]) for e in lessons]
        _evolution(rows, "How the live agent's weights evolved", "sandbox_evolution")

    trained = st.session_state.get("sandbox_train")
    if trained and trained[0] == account_id:
        _train_report(trained[1])


def _train_report(res: dict[str, Any]) -> None:
    r = res["data"]
    st.subheader(f"Walk-forward training · {r['start']} → {r['end']}")
    composite_badges(res["meta"])
    for w in r["warnings"]:
        st.warning(w, icon=":material/warning:")
    s, b = r["strategy"], r["benchmark_metrics"]
    m = st.columns(4)
    m[0].metric("Agent return", pct(s["total_return"], 1, signed=True))
    m[1].metric(f"{r['benchmark']} return", pct(b["total_return"], 1, signed=True))
    m[2].metric("Agent Sharpe", num(s["sharpe"]))
    m[3].metric("Weights applied", "Yes" if r["applied"] else "No")
    curve = pd.DataFrame(r["equity_curve"])
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.set_index("date").rename(columns={"strategy": "Agent", "benchmark": r["benchmark"]})
    _line_chart(curve, ["Agent", r["benchmark"]], "Replayed equity ($)", "Equity ($)", "sandbox_train_curve")
    st.dataframe(
        pd.DataFrame(
            [
                {"": "Agent", **s},
                {"": r["benchmark"], **b},
            ]
        ),
        hide_index=True,
        column_config={
            k: st.column_config.NumberColumn(format="percent")
            for k in ("total_return", "annual_return", "annual_volatility", "max_drawdown")
        }
        | {"sharpe": st.column_config.NumberColumn(format="%.2f")},
    )
    _evolution(
        [(w["date"], w["weights"]) for w in r["weights_history"]],
        "Factor weights during the replay",
        "sandbox_train_weights",
    )
    st.caption(
        f"{r['decisions']} decisions every {r['rebalance_every']} trading day(s) over {r['trading_days']} days; "
        f"{r['trades']} simulated trades, turnover {r['turnover']:.1f}×, fees {money(r['fees_paid'])}. "
        "Signals use closes up to day t and fill at day t+1's close (no look-ahead)."
    )
    if r["skipped"]:
        st.caption("Not replayed: " + "; ".join(f"{k} ({v})" for k, v in r["skipped"].items()))


def _positions(summary: dict[str, Any]) -> None:
    if not summary["positions"]:
        st.info("No open positions — the account is all cash.", icon=":material/savings:")
        return
    st.dataframe(
        pd.DataFrame(summary["positions"]),
        hide_index=True,
        column_config={
            "quantity": st.column_config.NumberColumn(format="%.4f"),
            "avg_cost": st.column_config.NumberColumn("avg cost", format="$%.2f"),
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "market_value": st.column_config.NumberColumn("value", format="$%.2f"),
            "weight": st.column_config.NumberColumn(format="percent"),
            "unrealized_pnl": st.column_config.NumberColumn("unrealized", format="$%.2f"),
            "unrealized_pnl_pct": st.column_config.NumberColumn("unrealized %", format="percent"),
            "data_status": st.column_config.TextColumn("data"),
        },
    )


def _trades(account_id: int) -> None:
    trades = guarded(lambda: api().get(f"{BASE}/{account_id}/trades", limit=500), "trades") or []
    if not trades:
        st.info("No trades yet.", icon=":material/receipt_long:")
        return
    df = pd.DataFrame(trades)
    df["executed_at"] = pd.to_datetime(df["executed_at"])
    st.dataframe(
        df[
            [
                "executed_at",
                "side",
                "symbol",
                "quantity",
                "price",
                "notional",
                "commission",
                "realized_pnl",
                "source",
                "data_status",
                "note",
            ]
        ],
        hide_index=True,
        column_config={
            "quantity": st.column_config.NumberColumn(format="%.4f"),
            "price": st.column_config.NumberColumn(format="$%.2f"),
            "notional": st.column_config.NumberColumn(format="$%.2f"),
            "commission": st.column_config.NumberColumn(format="$%.2f"),
            "realized_pnl": st.column_config.NumberColumn("realized", format="$%.2f"),
        },
    )


def _journal(account_id: int) -> None:
    entries = guarded(lambda: api().get(f"{BASE}/{account_id}/journal", limit=200), "journal") or []
    for e in entries:
        when = pd.Timestamp(e["created_at"]).strftime("%Y-%m-%d %H:%M UTC")
        icon = KIND_ICON.get(e["kind"], ":material/notes:")
        st.markdown(f"{icon} **{e['kind'].capitalize()}** · {when}  \n{e['summary']}")


def _order_form(account_id: int, mode: str) -> None:
    if mode == "agent":
        st.caption(
            "This account is run by the agent: at its next run it rebalances back to its own targets, so manual "
            "positions outside them will be sold."
        )
    with st.form("sandbox_order"):
        c = st.columns(4)
        symbol = c[0].text_input("Symbol", "AAPL")
        side = c[1].selectbox("Side", ["buy", "sell"])
        size_by = c[2].selectbox("Size by", ["dollars", "shares"])
        amount = c[3].number_input("Amount", 0.0001, 1e9, 1000.0)
        note = st.text_input("Note (optional)")
        if st.form_submit_button("Place paper order", icon=":material/shopping_cart:"):
            payload: dict[str, Any] = {"symbol": symbol, "side": side, "note": note or None}
            payload["notional" if size_by == "dollars" else "quantity"] = amount
            fill = guarded(lambda: api().post(f"{BASE}/{account_id}/orders", payload), "order")
            if fill:
                st.success(
                    f"Filled: {fill['side']} {fill['quantity']:g} {fill['symbol']} at ${fill['price']:,.2f} "
                    f"(quote ${fill['reference_price']:,.2f} ± slippage)",
                    icon=":material/check_circle:",
                )


def _settings(account: dict[str, Any]) -> None:
    aid = account["id"]
    key = f"sandbox_edit_{aid}"  # per-account keys so switching accounts shows that account's settings
    with st.form("sandbox_settings"):
        c = st.columns([2, 1, 1, 1])
        name = c[0].text_input("Name", account["name"], key=f"{key}_name")
        mode = c[1].selectbox(
            "Mode", ["agent", "manual"], index=0 if account["mode"] == "agent" else 1, key=f"{key}_mode"
        )
        auto = c[2].checkbox("Auto-trade", account["auto_trade"], key=f"{key}_auto")
        allow = c[3].checkbox("Allow synthetic prices", account["allow_synthetic"], key=f"{key}_allow")
        strategy = _strategy_inputs(key, account["strategy"])
        if st.form_submit_button("Save settings", icon=":material/save:"):
            payload = {
                "name": name,
                "mode": mode,
                "auto_trade": auto,
                "allow_synthetic": allow,
                "strategy": strategy,
            }
            if guarded(lambda: api().patch(f"{BASE}/{aid}", payload), "update"):
                st.rerun()
    c = st.columns(2)
    with c[0].form("sandbox_reset"):
        keep = st.checkbox("Keep what the agent learned", True)
        reset = st.form_submit_button("Reset to starting cash", icon=":material/restart_alt:")
        if reset and guarded(lambda: api().post(f"{BASE}/{aid}/reset", keep_learning=keep), "reset"):
            st.session_state.pop("sandbox_step", None)
            st.rerun()
    with c[1].form("sandbox_delete"):
        sure = st.checkbox("I understand this deletes the account and its history")
        if st.form_submit_button("Delete account", icon=":material/delete:"):
            if not sure:
                st.warning("Tick the confirmation box first.")
            elif guarded(lambda: api().delete(f"{BASE}/{aid}") or True, "delete"):
                st.rerun()


def _actions(account: dict[str, Any]) -> None:
    aid = account["id"]
    c = st.columns([1, 2])
    force = c[1].toggle(
        "Force", key="sandbox_force", help="Trade even if it already traded today or the market is closed."
    )
    if c[0].button("Run agent now", type="primary", icon=":material/smart_toy:"):
        with st.spinner("Learning from the last decision, ranking and rebalancing…"):
            step = guarded(lambda: api().post(f"{BASE}/{aid}/step", force=force or None), "agent run")
        if step:
            st.session_state["sandbox_step"] = step
            st.rerun()
    with (
        st.expander("Train on history (walk-forward, no look-ahead)", icon=":material/model_training:"),
        st.form("sandbox_train_form"),
    ):
        t = st.columns(3)
        lookback = t[0].select_slider(
            "History", [450, 730, 1095, 1825], value=1095, format_func=lambda d: f"{d / 365:.1f} years"
        )
        every = t[1].slider("Decide every N trading days", 1, 21, 5)
        apply = t[2].checkbox("Adopt the learned weights", True)
        if st.form_submit_button("Train", icon=":material/school:"):
            with st.spinner("Replaying history day by day…"):
                res = guarded(
                    lambda: api().post(
                        f"{BASE}/{aid}/train",
                        {"lookback_days": lookback, "rebalance_every": every, "apply": apply},
                    ),
                    "training",
                )
            if res:
                st.session_state["sandbox_train"] = (aid, res)
                st.session_state["sandbox_view"] = "Learning"
                st.rerun()


def render() -> None:
    st.title("Trading Sandbox")
    st.caption(
        "Paper trading with simulated money. The agent ranks stocks with the Daily Picks factors, trades a "
        "paper book at live quotes ± slippage, then scores each decision against what the market did next and "
        "re-weights its factors. No real orders are ever placed."
    )
    accounts = guarded(lambda: api().get(BASE), "accounts")
    if accounts is None:
        return
    with st.expander("Open a new paper account", icon=":material/add_card:", expanded=not accounts):
        _create_form()
    if not accounts:
        return
    by_id = {a["id"]: a for a in accounts}
    if st.session_state.get("sandbox_account") not in by_id:
        st.session_state["sandbox_account"] = next(iter(by_id))
    aid = st.selectbox(
        "Account",
        list(by_id),
        key="sandbox_account",
        format_func=lambda i: f"{by_id[i]['name']} · {by_id[i]['mode']}",
    )
    res = guarded(lambda: api().get(f"{BASE}/{aid}"), "account")
    if not res:
        return
    summary = res["data"]
    account, perf = summary["account"], summary["performance"]
    composite_badges(res["meta"])
    if account["allow_synthetic"]:
        st.warning(
            "This account may trade on SYNTHETIC prices when live data is down; its record is then not real.",
            icon=":material/science:",
        )
    m = st.columns(5)
    m[0].metric("Equity", money(perf["equity"]), pct(perf["total_return"], signed=True))
    m[1].metric(f"{perf['benchmark']} since start", pct(perf["benchmark_return"], signed=True))
    m[2].metric("Cash", money(perf["cash"]))
    m[3].metric("Realized P&L", money(perf["realized_pnl"]))
    m[4].metric("Max drawdown", pct(perf["max_drawdown"]))
    last = account["last_decision_on"] or "never"
    st.caption(
        f"{account['mode'].capitalize()} account · auto-trade {'on' if account['auto_trade'] else 'off'} · "
        f"last agent decision: {last} · learned from {account['periods_learned']} period(s) · "
        f"{perf['trades']} trades · fees {money(perf['fees_paid'])}"
    )
    if account["mode"] == "agent":
        _actions(account)
    step = st.session_state.get("sandbox_step")
    if step and step["account_id"] == aid:
        _step_result(step)

    view = st.segmented_control(
        "View",
        VIEWS,
        default=None if "sandbox_view" in st.session_state else "Performance",
        key="sandbox_view",
        label_visibility="collapsed",
    )
    match view or "Performance":
        case "Performance":
            _performance(aid, perf["benchmark"])
        case "Learning":
            _learning(account, aid)
        case "Positions":
            _positions(summary)
        case "Trades":
            _trades(aid)
        case "Journal":
            _journal(aid)
        case "Manual order":
            _order_form(aid, account["mode"])
        case "Settings":
            _settings(account)
    st.caption(summary["disclaimer"])
