"""Alpaca Paper Trading: QuantPulse's automated strategy on the Alpaca *paper* account (simulated money)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, guarded, money, num, pct

BASE = "/trading"
VIEWS = ["Portfolio", "Strategy", "Orders", "Risk", "Activity", "Performance", "Controls"]
COMPONENT_LABELS = {
    "momentum": "Momentum",
    "trend": "Trend",
    "volume": "Volume",
    "volatility": "Volatility",
    "fundamental": "Fundamentals",
    "model": "Model",
    "regime": "Regime fit",
}
REGIME_ICON = {
    "bullish": ":material/trending_up:",
    "neutral": ":material/trending_flat:",
    "high_volatility": ":material/bolt:",
    "bearish": ":material/trending_down:",
    "risk_off": ":material/shield:",
}
CLOSE_ALL_PHRASE = "CLOSE ALL"


def _md(text: str) -> str:
    """Escape dollar signs: Streamlit markdown renders text between two ``$`` as a LaTeX formula."""
    return text.replace("$", "\\$")


def _banners(status: dict[str, Any]) -> None:
    st.warning(f"**{status['banner']}**", icon=":material/science:")
    if status["mode"] == "paper":
        st.error(f"**{status['mode_banner']}**", icon=":material/send:")
    else:
        st.info(f"**{status['mode_banner']}**", icon=":material/visibility:")
    ks = status["kill_switch"]
    if ks["active"]:
        st.error(
            f"**KILL SWITCH ON** — no new orders ({ks.get('reason') or ks.get('source')}).",
            icon=":material/block:",
        )
    for w in status["warnings"]:
        st.warning(w, icon=":material/warning:")


def _setup_help() -> None:
    st.info(
        "Connect the Alpaca **paper** account by setting these in `.env` (paper keys from "
        "app.alpaca.markets → Paper Trading → API Keys), then restart the API:\n\n"
        "```\nQP_ALPACA_API_KEY_ID=...\nQP_ALPACA_API_SECRET_KEY=...\n```\n"
        "Orders stay off until you also set `QP_ALPACA_TRADING_ENABLED=true` and `QP_TRADING_DRY_RUN=false`.",
        icon=":material/key:",
    )


def _status_row(status: dict[str, Any]) -> None:
    c = st.columns(5)
    c[0].metric("Mode", "Paper" if status["mode"] == "paper" else "Dry run")
    c[1].metric("Trading enabled", "yes" if status["trading_enabled"] else "no")
    c[2].metric("Kill switch", "ON" if status["kill_switch"]["active"] else "off")
    market = status.get("market") or {}
    c[3].metric(
        "Market", "open" if market.get("is_open") else "closed", market.get("source"), delta_color="off"
    )
    nxt = status.get("next_cycle_at")
    c[4].metric(
        "Next cycle",
        pd.Timestamp(nxt).tz_convert("America/New_York").strftime("%a %H:%M ET") if nxt else "—",
        f"every {status['interval_minutes']} min" if status["scheduler_enabled"] else "scheduler off",
        delta_color="off",
    )


def _account(acct: dict[str, Any]) -> None:
    c = st.columns(5)
    c[0].metric("Equity", money(acct["equity"]))
    c[1].metric("Cash", money(acct["cash"]))
    c[2].metric("Buying power", money(acct["buying_power"]))
    c[3].metric("Today's P/L", money(acct["day_pl"]), pct(acct["day_pl_pct"], signed=True))
    c[4].metric(
        "Total P/L",
        money(acct["total_pl"]),
        pct(acct["total_pl_pct"], signed=True) if acct["total_pl_pct"] is not None else None,
        help=f"Since QuantPulse first recorded {money(acct['baseline_equity'])} on {acct['baseline_at']}",
    )
    st.caption(
        f"Alpaca paper account {acct['account_number']} · {acct['status']} · exposure "
        f"{pct(acct['exposure_pct'], 1)} · day trades {acct['daytrade_count']}"
        + (" · TRADING BLOCKED BY ALPACA" if acct["trading_blocked"] else "")
    )


# ----------------------------------------------------------------------------- views
def _portfolio() -> None:
    positions = guarded(lambda: api().get(f"{BASE}/positions"), "positions")
    if positions is None:
        return
    if not positions:
        st.info("No open positions on the Alpaca paper account.", icon=":material/inventory_2:")
        return
    df = pd.DataFrame(positions)
    st.dataframe(
        df[
            [
                "symbol",
                "qty",
                "avg_entry_price",
                "current_price",
                "market_value",
                "weight",
                "unrealized_pl",
                "unrealized_plpc",
                "target_weight",
                "signal_score",
                "stop_loss_price",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "symbol": "Symbol",
            "qty": st.column_config.NumberColumn("Shares", format="%g"),
            "avg_entry_price": st.column_config.NumberColumn("Avg entry", format="dollar"),
            "current_price": st.column_config.NumberColumn("Price", format="dollar"),
            "market_value": st.column_config.NumberColumn("Market value", format="dollar"),
            "weight": st.column_config.NumberColumn("Weight", format="percent"),
            "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L", format="dollar"),
            "unrealized_plpc": st.column_config.NumberColumn("P/L %", format="percent"),
            "target_weight": st.column_config.NumberColumn("Target", format="percent"),
            "signal_score": st.column_config.NumberColumn("Score", format="%+.2f"),
            "stop_loss_price": st.column_config.NumberColumn("Stop-loss", format="dollar"),
        },
    )
    fig = go.Figure()
    fig.add_bar(x=df["symbol"], y=df["weight"], name="current", marker_color=charts.series(0))
    fig.add_bar(x=df["symbol"], y=df["target_weight"].fillna(0), name="target", marker_color=charts.series(1))
    charts.base_layout(fig, "Current vs target weight", height=300, yaxis_tickformat=".0%", barmode="group")
    charts.show(fig, key="trade_alloc")


def _trades_table(trades: list[dict[str, Any]], key: str) -> None:
    if not trades:
        st.caption("No trades proposed.")
        return
    df = pd.DataFrame(trades)
    st.dataframe(
        df[
            [
                "symbol",
                "side",
                "qty",
                "notional",
                "kind",
                "status",
                "approved",
                "risk",
                "order_type",
                "limit_price",
                "reason",
            ]
        ],
        hide_index=True,
        width="stretch",
        key=key,
        column_config={
            "qty": st.column_config.NumberColumn("Shares", format="%g"),
            "notional": st.column_config.NumberColumn("Notional", format="dollar"),
            "limit_price": st.column_config.NumberColumn("Limit", format="dollar"),
            "approved": st.column_config.CheckboxColumn("Risk OK"),
            "risk": st.column_config.TextColumn("Risk decision", width="large"),
            "reason": st.column_config.TextColumn("Why", width="large"),
        },
    )
    with st.expander("Risk checks per trade", icon=":material/fact_check:"):
        for t in trades:
            mark = "✓" if t["approved"] else "✗"
            st.markdown(f"**{mark} {t['side'].upper()} {t['qty']:g} {t['symbol']}** — {t['kind']}")
            st.caption(
                _md(
                    " · ".join(
                        f"{'✓' if c['passed'] else '✗'} {c['name']}: {c['detail']}" for c in t["checks"]
                    )
                )
            )


def _strategy() -> None:
    cycle = guarded(lambda: api().get(f"{BASE}/proposed"), "latest cycle")
    if cycle is None:
        st.info(
            "No strategy cycle has run yet. Use **Controls → Run strategy now** (a dry run unless paper execution "
            "is enabled), or wait for the scheduler.",
            icon=":material/hourglass_empty:",
        )
        return
    st.caption(
        f"Cycle {cycle['cycle_key']} · {cycle['trigger']} · {'DRY RUN' if cycle['mode'] == 'dry_run' else 'PAPER'} · "
        f"{cycle['status']} · started {cycle['started_at']} · prices {str(cycle['data_status'] or '—').upper()}"
    )
    regime = cycle.get("regime")
    if regime:
        c = st.columns(4)
        c[0].metric("Market regime", regime["label"].replace("_", " ").title())
        c[1].metric("Exposure allowed", pct(regime["exposure_share"], 0))
        c[2].metric("Extra entry score", f"+{regime['entry_penalty']:.2f}")
        c[3].metric("Target gross exposure", pct(cycle.get("gross_target"), 0))
        st.info(
            f"{regime['description']}. " + "; ".join(regime["reasons"]),
            icon=REGIME_ICON.get(regime["label"], ":material/insights:"),
        )
    for note in cycle["notes"]:
        st.caption(_md(f"• {note}"))

    st.markdown("#### Top opportunities")
    signals = cycle["signals"]
    if signals:
        rows = []
        for s in signals:
            row = {
                "rank": s["rank"],
                "symbol": s["symbol"],
                "score": s["score"],
                **{COMPONENT_LABELS[k]: v for k, v in s["components"].items()},
                "price": s["price"],
                "volatility": s["risk_vol"],
                "trend OK": s["trend_ok"],
                "target": s["target_weight"],
                "held": s["held"],
                "blocked": "; ".join(s["entry_blocks"]),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        cfg: dict[str, Any] = {
            "score": st.column_config.ProgressColumn("Score", min_value=-3, max_value=3, format="%+.2f"),
            "price": st.column_config.NumberColumn(format="dollar"),
            "volatility": st.column_config.NumberColumn("Vol (ann.)", format="percent"),
            "target": st.column_config.NumberColumn("Target", format="percent"),
        }
        for label in COMPONENT_LABELS.values():
            cfg[label] = st.column_config.NumberColumn(label, format="%+.2f")
        st.dataframe(df, hide_index=True, width="stretch", column_config=cfg, height=380)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Target portfolio")
        if cycle["targets"]:
            st.dataframe(
                pd.DataFrame(cycle["targets"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "weight": st.column_config.NumberColumn("Weight", format="percent"),
                    "score": st.column_config.NumberColumn(format="%+.2f"),
                    "conviction": st.column_config.NumberColumn(format="%.2f"),
                    "risk_vol": st.column_config.NumberColumn("Vol", format="percent"),
                },
            )
        else:
            st.caption("All cash: nothing qualified.")
    with right:
        st.markdown("#### Portfolio after the cycle")
        if cycle["positions"]:
            st.dataframe(
                pd.DataFrame(cycle["positions"])[
                    ["symbol", "qty", "market_value", "weight", "unrealized_plpc"]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "market_value": st.column_config.NumberColumn("Value", format="dollar"),
                    "weight": st.column_config.NumberColumn(format="percent"),
                    "unrealized_plpc": st.column_config.NumberColumn("P/L %", format="percent"),
                },
            )
        else:
            st.caption("No positions.")

    st.markdown("#### Proposed trades and risk decisions")
    _trades_table(cycle["trades"], "trade_proposed_table")
    if cycle["exits"] or cycle["skipped"]:
        with st.expander("Exits and names left alone", icon=":material/info:"):
            for sym, why in cycle["exits"].items():
                st.markdown(_md(f"**{sym}** exit — {why}"))
            for sym, why in cycle["skipped"].items():
                st.caption(_md(f"{sym}: {why}"))


def _orders() -> None:
    which = st.segmented_control("Orders", ["all", "open", "closed"], default="all", key="trade_orders")
    orders = guarded(lambda: api().get(f"{BASE}/orders", status=which or "all", limit=200), "orders")
    if not orders:
        st.caption("No orders on the Alpaca paper account yet.")
        return
    df = pd.DataFrame(orders)
    st.dataframe(
        df[
            [
                "symbol",
                "side",
                "qty",
                "filled_qty",
                "order_type",
                "limit_price",
                "filled_avg_price",
                "status",
                "submitted_at",
                "filled_at",
                "kind",
                "strategy",
                "reason",
                "error",
                "client_order_id",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "qty": st.column_config.NumberColumn("Qty", format="%g"),
            "filled_qty": st.column_config.NumberColumn("Filled", format="%g"),
            "limit_price": st.column_config.NumberColumn("Limit", format="dollar"),
            "filled_avg_price": st.column_config.NumberColumn("Avg fill", format="dollar"),
            "submitted_at": st.column_config.DatetimeColumn("Submitted", format="MMM D HH:mm:ss"),
            "filled_at": st.column_config.DatetimeColumn("Filled", format="MMM D HH:mm:ss"),
        },
    )


def _risk() -> None:
    r = guarded(lambda: api().get(f"{BASE}/risk"), "risk")
    if r is None:
        return
    c = st.columns(4)
    c[0].metric(
        "Daily P/L",
        money(r["day_pl"]),
        pct(r["day_pl_pct"], signed=True),
        help=f"Limit −{pct(r['daily_loss_limit_pct'], 0)}; action when hit: {r['daily_loss_action']}",
    )
    c[1].metric(
        "Daily loss limit",
        f"−{pct(r['daily_loss_limit_pct'], 0)}",
        "HIT" if r["daily_loss_limit_hit"] else "ok",
        delta_color="inverse" if r["daily_loss_limit_hit"] else "off",
    )
    c[2].metric(
        "Exposure", pct(r["exposure_pct"], 1), f"max {pct(r['max_exposure_pct'], 0)}", delta_color="off"
    )
    c[3].metric(
        "Cash",
        money(r["cash"]),
        f"{pct(r['cash_pct'], 1)} (reserve {pct(r['cash_buffer_pct'], 0)})",
        delta_color="off",
    )
    d = st.columns(4)
    d[0].metric("Positions", f"{r['positions']} / {r['max_positions']}")
    d[1].metric(
        "Largest position",
        r["largest_position"] or "—",
        f"{pct(r['largest_position_pct'], 1)} (max {pct(r['max_position_pct'], 0)})"
        if r["largest_position"]
        else None,
        delta_color="off",
    )
    d[2].metric(
        "Order size", _md(f"{money(r['min_order_notional'], 0)} – {money(r['max_order_notional'], 0)}")
    )
    d[3].metric("Stop-loss per position", f"−{pct(r['position_loss_limit_pct'], 0)}")
    st.progress(
        min(max(-r["day_pl_pct"] / r["daily_loss_limit_pct"], 0.0), 1.0) if r["day_pl_pct"] < 0 else 0.0,
        text=f"Daily loss used: {pct(max(-r['day_pl_pct'], 0.0), 2)} of {pct(r['daily_loss_limit_pct'], 0)}",
    )
    st.progress(
        min(r["exposure_pct"] / r["max_exposure_pct"], 1.0) if r["max_exposure_pct"] else 0.0,
        text=f"Exposure used: {pct(r['exposure_pct'], 1)} of {pct(r['max_exposure_pct'], 0)}",
    )
    flags = st.columns(4)
    flags[0].metric("Kill switch", "ON" if r["kill_switch"]["active"] else "off")
    flags[1].metric("Dry run", "yes" if r["dry_run"] else "no")
    flags[2].metric("Trading enabled", "yes" if r["trading_enabled"] else "no")
    flags[3].metric("Orders possible now", "yes" if r["can_submit"] and r["market_open"] else "no")
    if r["positions_at_stop"]:
        st.error(
            "At or past the stop-loss: "
            + ", ".join(f"{s} ({pct(v, 1, signed=True)})" for s, v in r["positions_at_stop"].items()),
            icon=":material/trending_down:",
        )


def _activity() -> None:
    events = guarded(lambda: api().get(f"{BASE}/events", limit=300), "events") or []
    cycles = guarded(lambda: api().get(f"{BASE}/cycles", limit=50), "cycles") or []
    st.markdown("#### Cycles")
    if cycles:
        st.dataframe(
            pd.DataFrame(cycles)[
                [
                    "cycle_key",
                    "trigger",
                    "mode",
                    "status",
                    "started_at",
                    "trades_proposed",
                    "orders_submitted",
                    "skip_reason",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={"started_at": st.column_config.DatetimeColumn("Started", format="MMM D HH:mm")},
        )
    else:
        st.caption("No cycles yet.")
    st.markdown("#### Audit trail")
    if events:
        st.dataframe(
            pd.DataFrame(events)[["created_at", "kind", "symbol", "message", "client_order_id"]],
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "created_at": st.column_config.DatetimeColumn("When", format="MMM D HH:mm:ss"),
                "message": st.column_config.TextColumn("What happened", width="large"),
            },
        )
    else:
        st.caption("No events yet.")


def _performance() -> None:
    p = guarded(lambda: api().get(f"{BASE}/performance"), "performance")
    if p is None:
        return
    c = st.columns(5)
    c[0].metric("Total return", pct(p["total_return"], signed=True))
    c[1].metric("Sharpe", num(p["sharpe"]))
    c[2].metric("Sortino", num(p["sortino"]))
    c[3].metric("Max drawdown", pct(p["max_drawdown"]))
    c[4].metric("Win rate", pct(p["win_rate"], 0), f"{p['round_trips']} round trips", delta_color="off")
    d = st.columns(5)
    d[0].metric("Avg winner", money(p["avg_winner"]))
    d[1].metric("Avg loser", money(p["avg_loser"]))
    d[2].metric("Realized P/L", money(p["realized_pl"]))
    d[3].metric("Turnover", f"{p['turnover']:.2f}×" if p["turnover"] is not None else "—")
    d[4].metric("Avg exposure", pct(p["avg_exposure"], 0))
    for note in p["notes"]:
        st.caption(f"• {note}")
    if len(p["daily"]) >= 2:
        df = pd.DataFrame(p["daily"])
        fig = go.Figure(
            go.Scatter(
                x=df["date"],
                y=df["equity"],
                mode="lines+markers",
                line={"color": charts.series(0), "width": 2},
            )
        )
        charts.base_layout(fig, "Daily equity (recorded at each cycle)", height=320, yaxis_tickprefix="$")
        charts.show(fig, key="trade_equity")
    if p["monthly"]:
        st.dataframe(
            pd.DataFrame(p["monthly"]),
            hide_index=True,
            column_config={
                "pl": st.column_config.NumberColumn("P/L", format="dollar"),
                "return": st.column_config.NumberColumn("Return", format="percent"),
            },
        )
    if p["by_symbol"] or p["by_exit"]:
        left, right = st.columns(2)
        left.markdown("**Realized P/L by symbol**")
        left.dataframe(
            pd.Series(p["by_symbol"], name="P/L").to_frame(),
            column_config={"P/L": st.column_config.NumberColumn(format="dollar")},
        )
        right.markdown("**Realized P/L by exit type**")
        right.dataframe(
            pd.Series(p["by_exit"], name="P/L").to_frame(),
            column_config={"P/L": st.column_config.NumberColumn(format="dollar")},
        )


def _controls(status: dict[str, Any]) -> None:
    st.markdown("#### Run strategy now")
    force_dry = st.checkbox(
        "Dry run only (compute everything, send nothing)",
        value=status["mode"] != "paper",
        disabled=status["mode"] != "paper",
        key="trade_force_dry",
    )
    if st.button("Run strategy now", type="primary", icon=":material/play_arrow:", key="trade_run"):
        out = guarded(lambda: api().post(f"{BASE}/run", dry_run=force_dry, wait=60), "strategy cycle")
        if out:
            sent = sum(1 for t in out["trades"] if t["client_order_id"])
            st.success(
                f"Cycle {out['cycle_key']} {out['status']} ({'dry run' if out['mode'] == 'dry_run' else 'paper'}): "
                f"{len(out['trades'])} trade(s) proposed, {sent} order(s) sent."
                + (f" Error: {out['error']}" if out["error"] else ""),
                icon=":material/check_circle:",
            )

    st.markdown("#### Kill switch")
    ks = status["kill_switch"]
    if ks["active"]:
        st.error(f"Kill switch is ON ({ks.get('reason') or ks['source']}).", icon=":material/block:")
        if ks["source"] == "env":
            st.caption("Set by QP_TRADING_KILL_SWITCH=true: change the setting and restart to release it.")
        elif st.button("Release kill switch", icon=":material/lock_open:", key="trade_release") and guarded(
            lambda: api().post(f"{BASE}/kill-switch", {"active": False}), "kill switch"
        ):
            st.rerun()
    else:
        with st.form("trade_kill"):
            reason = st.text_input("Reason (optional)", key="trade_kill_reason")
            cancel = st.checkbox("Also cancel working orders", value=True, key="trade_kill_cancel")
            body = {"active": True, "reason": reason or None, "cancel_open_orders": cancel}
            if st.form_submit_button("Activate kill switch", icon=":material/block:") and guarded(
                lambda: api().post(f"{BASE}/kill-switch", body), "kill switch"
            ):
                st.rerun()

    st.markdown("#### Reconcile with Alpaca")
    if st.button("Reconcile now", icon=":material/sync:", key="trade_reconcile"):
        rec = guarded(lambda: api().post(f"{BASE}/reconcile"), "reconciliation")
        if rec:
            st.success(
                f"{rec['positions']} position(s), {rec['open_orders']} open order(s); "
                f"{rec['orders_updated']} updated, {rec['orders_added']} added.",
                icon=":material/sync:",
            )

    with st.expander("Danger zone", icon=":material/warning:"):
        st.markdown(
            "**Cancel all open orders** on the Alpaca paper account (including ones placed elsewhere)."
        )
        confirm_cancel = st.checkbox("I want to cancel every open order", key="trade_cancel_confirm")
        if st.button("Cancel all open orders", disabled=not confirm_cancel, key="trade_cancel_all"):
            out = guarded(lambda: api().post(f"{BASE}/cancel-all", {"confirm": True}), "cancel all")
            if out:
                st.success(out["message"])
        st.divider()
        st.markdown(
            "**Close all positions** with market orders"
            + (" (a preview: dry-run mode sends nothing)" if status["mode"] != "paper" else "")
            + f". Type `{CLOSE_ALL_PHRASE}` to enable the button."
        )
        phrase = st.text_input("Confirmation", key="trade_close_phrase", placeholder=CLOSE_ALL_PHRASE)
        if st.button(
            "Close all positions",
            type="primary",
            disabled=phrase.strip() != CLOSE_ALL_PHRASE,
            key="trade_close_all",
        ):
            out = guarded(lambda: api().post(f"{BASE}/close-all", {"confirm": phrase.strip()}), "close all")
            if out:
                st.success(out["message"])
                _trades_table(out["trades"], "trade_close_table")


def render() -> None:
    st.title("Alpaca Paper Trading")
    st.caption(
        "QuantPulse's automated strategy on your Alpaca **paper** account: it ranks a liquid universe, sizes a "
        "concentrated portfolio by conviction and volatility, passes every order through the risk engine and "
        "reconciles fills with Alpaca. Simulated money only — there is no live-money path."
    )
    status = guarded(lambda: api().get(f"{BASE}/status"), "trading status")
    if status is None:
        return
    _banners(status)
    if not status["broker_configured"]:
        _setup_help()
        return
    _status_row(status)
    acct = guarded(lambda: api().get(f"{BASE}/account"), "Alpaca paper account")
    if acct:
        _account(acct)
    view = st.segmented_control("View", VIEWS, default="Portfolio", key="trade_view") or "Portfolio"
    if view == "Portfolio":
        _portfolio()
    elif view == "Strategy":
        _strategy()
    elif view == "Orders":
        _orders()
    elif view == "Risk":
        _risk()
    elif view == "Activity":
        _activity()
    elif view == "Performance":
        _performance()
    else:
        _controls(status)
