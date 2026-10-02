"""Home — the page to open first, on a desktop or a phone: is everything all right, how is the paper account
doing, what is the Brain doing, and the stop button.

One status line on top (green: trading on its own; yellow: healthy but not trading now; red: stopped or
broken, and why), the account in four cards, the positions, the Brain's latest decision and its recent trades.
The STOP button is always on the page: one tap stops new Brain orders and cancels its working orders;
releasing it needs a second, deliberate step. The technical details (every part's health, the cloud, the
switches, alerts) are folded away. Nothing here shows a key, a token or a password.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend import ui
from frontend.components import api, guarded, money, pct, today_split

BASE = "/brain"
MARK = {"ok": "✓", "warn": "!", "fail": "✗", "standby": "‖", "n/a": "–"}
WORKING = ("new", "accepted", "partially_filled", "pending_new")


def _status(cs: dict[str, Any] | None, ex: dict[str, Any] | None, report: dict[str, Any] | None) -> None:
    kill = (ex or {}).get("brain_kill_switch") or {}
    if kill.get("active"):
        ui.status(
            "red", "Brain trading is stopped", kill.get("reason") or f"switched on ({kill.get('source')})"
        )
    elif cs:
        sm = cs.get("summary") or {}
        why = (sm.get("problems") or [])[:2]
        if not why and sm.get("light") == "yellow":
            why = ((cs.get("autonomous_execution") or {}).get("reasons") or [])[1:3]
        ui.status(
            sm.get("light", ""), ui.plain(sm.get("headline") or "—"), "; ".join(map(ui.plain, why)) or None
        )
    for blocker in (report or {}).get("order_blockers") or []:
        st.warning(ui.md(f"New Brain orders held: {blocker}"), icon=":material/block:")


def _account(acct: dict[str, Any] | None) -> None:
    if not acct:
        return
    ui.kpis(
        [
            ui.Kpi("Equity", money(acct.get("equity"), 0), pct(acct.get("day_pl_pct"), 2, signed=True)),
            ui.Kpi("Today's P&L", money(acct.get("day_pl"), 0)),
            ui.Kpi(
                "Total P&L", money(acct.get("total_pl"), 0), pct(acct.get("total_pl_pct"), 2, signed=True)
            ),
            ui.Kpi("Buying power", money(acct.get("buying_power"), 0)),
        ],
        key="home_account",
    )


def _positions(
    live: list[dict[str, Any]] | None,
    acct: dict[str, Any] | None,
    theses: dict[str, Any] | None,
    orders: list[Any] | None,
) -> None:
    """Alpaca's positions, read live alongside the account above, so the two always agree. "Today" is each
    position's change today; "Total" its gain since it was bought."""
    rows = live or []
    ui.section(f"Positions ({len(rows)})" if rows else "Positions")
    if rows:
        st.dataframe(
            pd.DataFrame(
                [{"Symbol": p["symbol"], "Shares": p.get("qty"), "Value": p.get("market_value"),
                  "Today": p.get("intraday_pl"), "Total": p.get("unrealized_pl"),
                  "Return": p.get("unrealized_plpc")} for p in rows]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Shares": st.column_config.NumberColumn(format="%g"),
                "Value": st.column_config.NumberColumn(format="dollar"),
                "Today": st.column_config.NumberColumn(format="dollar", help="Change today"),
                "Total": st.column_config.NumberColumn(format="dollar", help="Gain since bought"),
                "Return": st.column_config.NumberColumn(format="percent", help="Since bought"),
            },
        )  # fmt: skip
    elif live is not None:
        st.caption("No open positions.")
    split = today_split((acct or {}).get("day_pl"), live)
    if split and (rows or abs((acct or {}).get("day_pl") or 0.0) >= 0.5):
        st.caption(split)
    unexpected = (theses or {}).get("unexpected") or []
    if unexpected:
        st.warning(
            ui.md(f"Not opened by the Brain: {', '.join(map(str, unexpected))}"), icon=":material/help:"
        )
    working = [o for o in orders or [] if o.get("status") in WORKING]
    if working:
        st.caption(
            ui.md(
                "Working orders: "
                + ", ".join(f"{o['side']} {o['qty']:g} {o['symbol']}" for o in working[:10])
            )
        )
    ui.link("trading", "Orders, performance and risk")


def _learning(learning: dict[str, Any] | None) -> str | None:
    """Graded predictions so far, the hit rate once there is one (unproven below the sample the learning
    system needs before it trusts a record), and when the next ones are graded."""
    p = (learning or {}).get("predictions") or {}
    if not p:
        return None
    need = (learning or {}).get("min_observations") or 0
    small = " (unproven)" if p.get("evaluated", 0) < need else ""
    hit = f" · {p['hit_rate']:.0%} right{small}" if p.get("hit_rate") is not None else ""
    due = f" · next graded {p['next_due']}" if p.get("next_due") else ""
    return f"{p.get('evaluated', 0)} graded{hit} · {p.get('open', 0)} open{due}"


def _brain(cs: dict[str, Any] | None, sup: dict[str, Any] | None) -> None:
    ui.section("Brain")
    sup = sup or {}
    state = (
        "paused" if sup.get("paused") else
        "standby" if sup.get("standby") else
        "waiting" if sup.get("waiting") else
        "running" if sup.get("enabled") else "off"
    )  # fmt: skip
    market = (cs or {}).get("market") or {}
    today = (cs or {}).get("today") or {}
    last = ((cs or {}).get("last_cycle") or {}).get("last_at")
    ui.facts(
        [
            ("Supervisor", state),
            ("Market", ("open" if market.get("open") else "closed") + (f" · {market['new_york_time'][11:]} NY" if market.get("new_york_time") else "")),
            ("Last cycle", ui.when(last)),
            ("Next cycle", ui.when(sup.get("next_cycle_at"))),
            ("Orders today", f"{today.get('orders', 0)} sent · {today.get('fills', 0)} filled"),
            ("Learning", _learning(guarded(lambda: api().get(f"{BASE}/learning"), "learning"))),
        ]
    )  # fmt: skip
    if sup.get("waiting"):
        st.caption(ui.md(f"Waiting: {sup['waiting']}"))
    cycles = guarded(lambda: api().get(f"{BASE}/cycles", limit=1), "cycles") or []
    cycle = guarded(lambda: api().get(f"{BASE}/cycles/{cycles[0]['id']}"), "cycle") if cycles else None
    decision = ((cycle or {}).get("summary") or {}).get("decision") or {}
    if decision:
        with st.container(border=True):
            st.markdown(ui.md(f"**Latest decision** · cycle #{cycle['id']}"))  # type: ignore[index]
            st.markdown(ui.md((decision.get("headline") or "").capitalize()))
            for o in (decision.get("orders") or [])[:5]:
                st.caption(
                    ui.md(f"{o['action']} {o['qty']:g} {o['subject']} — {'; '.join(o.get('why') or [])}")
                )
            for r in (decision.get("reasons") or [])[:3]:
                st.caption(ui.md(r))
    elif not cycles:
        st.caption("The Brain has not run a cycle yet.")
    trades = guarded(lambda: api().get(f"{BASE}/trades", limit=8), "trades") or []
    if trades:
        st.markdown("**Recent trades**")
        st.dataframe(
            pd.DataFrame(
                [{"When": ui.when(t["at"]), "Action": t["action"], "Symbol": t["subject"], "Qty": t.get("quantity"),
                  "Result": t["status"].replace("_", " "), "Note": t.get("reason") or ""} for t in trades[:8]]
            ),
            hide_index=True,
            width="stretch",
            column_config={"Qty": st.column_config.NumberColumn(format="%g")},
        )  # fmt: skip
    ui.link("brain", "Everything the Brain saw and decided")


def _stop(ex: dict[str, Any] | None) -> None:
    if not ex:
        return
    kill = ex["brain_kill_switch"]
    with st.container(border=True):
        if kill["active"]:
            st.error(
                ui.md(f"BRAIN TRADING STOPPED — {kill.get('reason') or kill['source']}"),
                icon=":material/block:",
            )
            if kill["source"] == "env":
                st.caption(
                    "Set by QP_BRAIN_KILL_SWITCH=true on the server: change it there and restart to release."
                )
                return
            sure = st.checkbox("I have looked: let the Brain send orders again", key="remote_release_sure")
            if st.button("Allow Brain orders again", icon=":material/lock_open:", disabled=not sure,
                         key="remote_release") and guarded(
                lambda: api().post(f"{BASE}/kill-switch", {"active": False}), "Brain kill switch"
            ):  # fmt: skip
                st.rerun()
            return
        if not ex["owns_account"]:
            st.caption(ui.md(f"QP_BRAIN_MODE={ex['mode']}: the Brain proposes only; it sends no orders."))
            return
        text, button = st.columns([3, 1], vertical_alignment="center")
        text.markdown("**Emergency stop**")
        text.caption(
            "No new Brain orders, and its working orders are canceled. Positions stay as they are; the switch "
            "survives restarts."
        )
        if button.button("STOP BRAIN TRADING", icon=":material/block:", type="primary", width="stretch",
                         key="remote_stop"):  # fmt: skip
            body = {"active": True, "reason": "stopped from the phone", "cancel_open_orders": True}
            if guarded(lambda: api().post(f"{BASE}/kill-switch", body), "Brain kill switch"):
                st.rerun()


def _details(cs: dict[str, Any] | None, report: dict[str, Any] | None, ex: dict[str, Any] | None) -> None:
    with st.expander("System details", icon=":material/tune:"):
        if report:
            st.markdown(f"**Health: {report['status'].upper()}**")
            st.dataframe(
                pd.DataFrame(
                    [{"": MARK.get(p["status"], "–"), "Part": name.replace("_", " "), "Detail": p["detail"]}
                     for name, p in report["parts"].items()]
                ),
                hide_index=True,
                width="stretch",
            )  # fmt: skip
        if cs:
            sv, sup, sw = cs["service"], cs["supervisor"], cs["switches"]
            leader = sup.get("leader") or {}
            recon = cs.get("reconciliation") or {}
            ui.facts(
                [
                    ("Version", f"v{sv['version']} · {sv.get('short_commit') or 'local'}"),
                    ("Uptime", ui.ago(sv.get("uptime_seconds")).replace(" ago", "")),
                    ("Database", f"{cs['database']['status']} · schema {cs['database'].get('schema') or '—'}"),
                    ("Supervisor lease", "this server" if sup.get("this_process_is_leader") else (leader.get("holder") or "nobody")),
                    ("Last reconciliation", ui.ago(recon.get("last_success_age_seconds"))),
                    ("Paper endpoint", "verified" if cs["alpaca"]["paper_endpoint_verified"] else "NOT verified"),
                    ("Alpaca", cs["alpaca"]["connectivity"]["status"]),
                    ("Trading enabled", "yes" if sw["trading_enabled"] else "no"),
                    ("Dry run", "yes" if sw["dry_run"] else "no"),
                    ("Brain mode", sw["brain_mode"].replace("_", " ")),
                    ("Market data", (cs.get("data") or {}).get("detail", "—")),
                ]
            )  # fmt: skip
            ae = cs["autonomous_execution"]
            st.caption(
                ui.md(
                    "Autonomous PAPER execution: "
                    + ("permitted" if ae["permitted"] else "not permitted now — " + "; ".join(ae["reasons"]))
                )
            )
        if ex and ex.get("blockers_scheduled"):
            st.caption(ui.md("Scheduled Brain orders wait for: " + "; ".join(ex["blockers_scheduled"])))
    with st.expander("Recent alerts", icon=":material/notifications:"):
        alerts = guarded(lambda: api().get("/system/alerts"), "alerts") or {}
        channels = ", ".join(alerts.get("channels") or []) or "none configured (recorded only)"
        st.caption(ui.md(f"Delivered to: {channels}"))
        recent = alerts.get("recent") or []
        if recent:
            st.dataframe(
                pd.DataFrame(
                    [{"When": ui.when(a["at"]), "Level": a["severity"], "Alert": a["title"], "Message": a["message"]}
                     for a in recent[:15]]
                ),
                hide_index=True,
                width="stretch",
            )  # fmt: skip
        else:
            st.caption("No alerts.")


def _page() -> None:
    cs = guarded(lambda: api().get(f"{BASE}/cloud-status"), "status")
    ex = guarded(lambda: api().get(f"{BASE}/execution"), "Brain execution")
    report = guarded(lambda: api().get("/system/health"), "health")
    _status(cs, ex, report)
    # the account and its positions are read live from Alpaca together, so the cards and the table agree
    acct = guarded(lambda: api().get("/trading/account"), "account")
    live = guarded(lambda: api().get("/trading/positions"), "positions")
    _account(acct)
    _positions(
        live,
        acct,
        guarded(lambda: api().get(f"{BASE}/positions", closed=0, live="false"), "Brain positions"),
        guarded(lambda: api().get("/trading/orders"), "orders"),
    )
    _brain(cs, guarded(lambda: api().get(f"{BASE}/supervisor"), "supervisor"))
    _stop(ex)
    _details(cs, report, ex)


def render() -> None:
    ui.header("Home", "Your Alpaca paper account and what the Brain is doing.")
    every = st.session_state.get("refresh_seconds")
    st.fragment(_page, run_every=every)()
