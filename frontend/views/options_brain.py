"""Options Intelligence: what the Options Brain sees, considers, holds and learns — paper and shadow kept apart,
research evidence labelled model-priced. Alpaca PAPER only."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend import ui
from frontend.components import api, guarded, money, pct


def _table(
    rows: list[dict[str, Any]] | None, cols: list[str] | None = None, empty: str = "Nothing yet."
) -> None:
    if not rows:
        st.caption(empty)
        return
    df = pd.DataFrame(rows)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    st.dataframe(df, width="stretch", hide_index=True, column_config=ui.et_times(df))


CLEARED = ("PAPER_SHADOW", "PAPER_ACTIVE", "PROVEN")
# trade one exploration contract (and in shadow) while exploration is on
EXPLORING = ("VALIDATION", "WALK_FORWARD")
# the lab's stages, as the ladder shows them: (label, stages, what it means, does it trade)
LADDER = (
    ("Ideas", ("RESEARCH", "EXTRACTED"), "written down", False),
    ("Backtested", ("BACKTESTING",), "five fill models", False),
    ("Validated", ("VALIDATION",), "positive after costs", False),
    ("Walk-forward", ("WALK_FORWARD",), "held-out period", False),
    ("Shadow", ("PAPER_SHADOW",), "live quotes, +1 contract", True),
    ("Paper", ("PAPER_ACTIVE",), "full size", True),
    ("Proven", ("PROVEN",), "50+ paper trades", True),
)


def _ladder(stages: dict[str, Any], exploring: bool) -> None:
    trades = (
        "from validated on a strategy trades: one exploration contract on paper beside its shadow trades, then "
        "full size once every gate and its shadow record earn it"
        if exploring
        else "only the last three trade (shadow on live quotes, then paper)"
    )
    retired = f" {int(stages.get('RETIRED') or 0)} retired." if stages.get("RETIRED") else ""
    ui.section("The strategy ladder", f"Every option strategy climbs one gate at a time; {trades}.{retired}")
    ui.ladder(
        [
            ui.Step(
                label,
                sum(int(stages.get(k) or 0) for k in keys),
                f"{note}, +1 contract" if exploring and set(keys) <= set(EXPLORING) else note,
                live or (exploring and set(keys) <= set(EXPLORING)),
            )
            for label, keys, note, live in LADDER
        ]
    )


def _limits(limits: dict[str, Any]) -> None:
    def usd(x: Any) -> str:
        return money(x, 0) if x is not None else "—"

    dte = limits.get("dte") or [None, None]
    ui.facts(
        [
            ("Per position", f"{usd(limits.get('max_loss_per_trade'))} · {pct(limits.get('max_loss_pct_per_trade'), 0)}"),
            ("All options", pct(limits.get("max_total_risk_pct"), 0)),
            ("One underlying", pct(limits.get("max_underlying_risk_pct"), 0)),
            ("Open positions", limits.get("max_positions")),
            ("Contracts per leg", limits.get("max_contracts")),
            ("Days to expiry", f"{dte[0]}–{dte[1]} (closed at {limits.get('close_dte')})"),
            ("Net delta / vega", f"{pct(limits.get('max_delta_pct'), 0)} / {pct(limits.get('max_vega_pct'), 1)}"),
            ("Widest spread", pct(limits.get("max_spread_pct"), 0)),
            ("Quote age", f"{limits.get('max_quote_age_seconds', 0):.0f} s"),
            ("Exploration", f"{'on' if limits.get('exploration') else 'off'} · {usd(limits.get('exploration_max_loss'))}"),
        ]
    )  # fmt: skip


def _scan_line(st_: dict[str, Any]) -> str | None:
    """Which underlyings the Options Brain reads: the core list every cycle, plus the rotating scan."""
    core = st_.get("universe") or []
    scan = st_.get("scan") or {}
    size, per = int(scan.get("size") or 0), int(scan.get("per_cycle") or 0)
    if not core and not (size and per):
        return None
    line = f"Reads {len(core)} core underlying{'s' if len(core) != 1 else ''} every cycle"
    if size and per:
        line += f", plus {per} a cycle from a rotating scan of the {size} most liquid stocks and ETFs"
        read = ((scan.get("last") or {}).get("read")) or []
        if read:
            line += f" (last: {', '.join(read[:8])}{'…' if len(read) > 8 else ''})"
    return line + "."


def _overview(st_: dict[str, Any]) -> None:
    lab = st_.get("lab") or {}
    stages = lab.get("by_stage") or {}
    exploring = bool((st_.get("limits") or {}).get("exploration"))
    cleared = sum(int(stages.get(k) or 0) for k in (*CLEARED, *(EXPLORING if exploring else ())))
    if not st_.get("enabled"):
        ui.status("gray", "Options are off", "QP_OPTIONS_ENABLED=false")
    elif cleared:
        ui.status(
            "green",
            f"{cleared} option strateg{'y is' if cleared == 1 else 'ies are'} cleared to trade on paper",
        )
    else:
        ui.status(
            "yellow",
            "No option trades yet: no strategy has passed validation",
            "Research runs while the market is closed. "
            + (
                "A strategy may place its first (one-contract, exploration) paper order once its backtests are "
                "positive after realistic and pessimistic costs."
                if exploring
                else "A strategy needs a full backtest, walk-forward, stress tests and live shadow trades before it "
                "may place a paper order."
            ),
        )
    op = st_.get("open_positions") or {}
    ui.kpis(
        [
            ui.Kpi("Paper execution", "on" if st_.get("execution") else "off"),
            ui.Kpi("Strategies cleared", cleared, f"of {sum(int(v or 0) for v in stages.values())} researched",
                   delta_color="off", arrow="off"),
            ui.Kpi("Open positions", op.get("paper", 0), f"{op.get('shadow', 0)} shadow", delta_color="off",
                   arrow="off"),
            ui.Kpi("Priority weight", f"{st_.get('priority_weight', 0):.2f}", help="Favours options, never forces them"),
        ],
        key="opt_status",
    )  # fmt: skip
    if stages:
        _ladder(stages, exploring)
    scan = _scan_line(st_)
    if scan:
        st.caption(ui.md(scan))
    last = st_.get("last_pass") or {}
    if cleared:  # without a cleared strategy the status line above already says why
        for n in (st_.get("last_cycle") or {}).get("notes") or []:
            st.caption(ui.md(n))
    if last.get("no_trade"):
        st.caption(ui.md("Last pass, no trade: " + "; ".join(last["no_trade"].get("reasons") or [])))
    if last.get("candidates"):
        ui.section("Last options pass")
        _table(
            last.get("candidates"),
            ["underlying", "strategy", "family", "status", "gate", "score", "explanation"],
        )
    with st.expander("Rules and limits", icon=":material/shield:"):
        st.markdown(
            "Options are traded on the **Alpaca paper account only**, through the same trading service, risk "
            "engine and order manager as shares. Defined-risk structures only; never naked short options, never "
            "0DTE, never an exercise. Research evidence is **model-priced** (no historical option quotes) and "
            "labelled."
        )
        st.caption(ui.md(f"Feed: {st_.get('feed')} — {st_.get('feed_note')}"))
        st.markdown("**Limits** (protected: they can only be tightened)")
        _limits(st_.get("limits") or {})
    with st.expander("The agents and their questions", icon=":material/groups:"):
        _table(st_.get("agents"), ["agent", "question", "weight", "where"])


def _chains(configured: bool) -> None:
    if not configured:
        st.caption(
            "Options market data needs the Alpaca paper keys (QP_ALPACA_API_KEY_ID / QP_ALPACA_API_SECRET_KEY)."
        )
        return
    u = st.text_input("Underlying", "SPY", key="ob_chain_u").strip().upper()
    if not u:
        return
    data = guarded(lambda: api().get("/options/chains", underlying=u), "option chain")
    if not data:
        return
    c = st.columns(4)
    c[0].metric("Spot", money(data.get("spot")))
    c[1].metric("30-day ATM IV", pct(data.get("atm_iv_30d")))
    c[2].metric("Term structure", str((data.get("term_structure") or {}).get("shape")))
    c[3].metric("Execution-grade contracts", str((data.get("quality") or {}).get("usable_for_execution")))
    st.caption(f"feed {data.get('feed')}: " + "; ".join(data.get("notes") or []))
    _table(data.get("expirations"), ["expiration", "dte", "atm_iv", "skew", "implied_move_pct"])
    _table(
        data.get("contracts"),
        ["symbol", "kind", "strike", "bid", "ask", "iv", "delta", "open_interest", "age_seconds"],
    )


def _candidates() -> None:
    rows = guarded(lambda: api().get("/options/candidates", limit=100), "candidates") or []
    _table(rows, ["at", "underlying", "family", "mode", "status", "gate", "score", "dte", "iv_rank", "reject_reason"],
           "No candidate yet: strategies must pass validation first.")  # fmt: skip
    for r in rows[:10]:
        with st.expander(f"{r['underlying']} {r['family']} — {r['status']}"):
            if r.get("explanation"):
                st.write(r["explanation"])
            if r.get("debate"):
                st.json(r["debate"])
            if r.get("comparison"):
                st.caption("Options vs shares: " + str(r["comparison"].get("verdict")))
            _table(r.get("agents"), ["agent", "verdict", "score", "reasons"])


def _positions() -> None:
    mode = st.segmented_control("Book", ["paper", "shadow"], default="paper", key="ob_mode")
    rows = guarded(lambda: api().get("/options/positions", mode=mode or "paper"), "positions") or []
    _table(rows, ["id", "status", "underlying", "family", "quantity", "first_expiration", "expiry_state", "entry_value",
                  "max_loss", "unrealized_pnl", "realized_pnl", "exit_reason", "strategy", "exploration"],
           f"No {mode} option positions.")  # fmt: skip


def _greeks() -> None:
    g = guarded(lambda: api().get("/options/greeks"), "greeks") or {}
    for mode in ("paper", "shadow"):
        st.markdown(f"**{mode.capitalize()} book**")
        st.dataframe(
            pd.DataFrame([((g.get(mode) or {}).get("total")) or {}]), hide_index=True, width="stretch"
        )
        _table((g.get(mode) or {}).get("positions"), None, "No open positions.")


def _performance() -> None:
    p = guarded(lambda: api().get("/options/performance"), "performance") or {}
    st.caption(p.get("note", ""))
    for mode in ("paper", "shadow"):
        m = p.get(mode) or {}
        st.markdown(f"**{mode.capitalize()}** — {m.get('trades', 0)} closed trade(s)")
        cols = st.columns(4)
        cols[0].metric("Win rate", pct(m.get("win_rate")))
        cols[1].metric("Expectancy", money(m.get("expectancy")))
        cols[2].metric("Per $ at risk", pct(m.get("expectancy_on_risk")))
        cols[3].metric("Total P&L", money(m.get("total_pnl")))
        if m.get("attribution"):
            st.caption("P&L attribution: " + ", ".join(f"{k} {v:+,.0f}" for k, v in m["attribution"].items()))


def _strategies() -> None:
    rows = guarded(lambda: api().get("/options/strategies", limit=300), "strategies") or []
    _table(rows, ["id", "key", "version", "family", "stage", "origin", "generation", "name"],
           "The research library has not been seeded yet (the first research run does it).")  # fmt: skip
    ids = [r["id"] for r in rows]
    if ids:
        pick = st.selectbox("Strategy version", ids, format_func=lambda i: next(
            f"#{r['id']} {r['key']} v{r['version']} ({r['stage']})" for r in rows if r["id"] == i))  # fmt: skip
        d = guarded(lambda: api().get(f"/options/strategies/{pick}"), "strategy")
        if d:
            st.caption(d.get("label", ""))
            st.write(d.get("name"))
            if d.get("next_gate"):
                st.markdown("**What the next stage still needs:** " + "; ".join(d["next_gate"]))
            _table(d.get("backtests"), ["execution_model", "data_source", "trades", "period", "run_at"])
            _table(d.get("regimes"), ["regime", "source", "trades", "expectancy_on_risk", "win_rate"])
            _table(d.get("stress"), ["kind", "passed"])
            with st.expander("Stage history"):
                st.json(d.get("stage_history"))


def _research() -> None:
    r = guarded(lambda: api().get("/options/research"), "research") or {}
    st.caption("A source's claim is a hypothesis QuantPulse tests — never a fact it assumes.")
    _table(r.get("sources"), ["key", "title", "author", "quality", "status", "claim_status", "claim"])
    if st.button("Start a research run (background; never trades)", key="ob_research"):
        job = guarded(lambda: api().post("/options/research/run"), "research run")
        if job:
            st.success(f"Research job {job.get('job_id')} started.")
    with st.expander("Knowledge graph (latest edges)"):
        _table(r.get("knowledge"), ["src", "relation", "dst", "weight"])


def _experiments() -> None:
    rows = guarded(lambda: api().get("/options/experiments"), "experiments") or []
    st.caption("Queued by expected information, not expected profit.")
    _table(
        rows, ["id", "hypothesis", "status", "priority", "parent_version_id", "child_version_id", "decision"]
    )


def _learning() -> None:
    lr = guarded(lambda: api().get("/options/learning"), "learning") or {}
    st.caption(lr.get("note", ""))
    st.markdown(
        f"Graded trades — shadow {lr.get('graded', {}).get('shadow', 0)}, paper {lr.get('graded', {}).get('paper', 0)}"
    )
    _table(lr.get("weights"), ["strategy", "regime", "structure", "vol_state", "p_edge", "mean", "n"])
    _table(lr.get("lessons"), ["memory", "observation", "hypothesis", "confidence", "n", "status"])


def _counterfactuals() -> None:
    _table(guarded(lambda: api().get("/options/counterfactuals"), "counterfactuals"),
           ["position_id", "alternative", "pnl", "chosen_pnl", "better_than_chosen", "data", "at"])  # fmt: skip


def _missed() -> None:
    m = guarded(lambda: api().get("/options/missed-opportunities"), "missed opportunities") or {}
    if m.get("summary"):
        st.json(m["summary"])
    _table(
        m.get("items"),
        ["underlying", "family", "gate", "reason", "grade_after", "outcome_pnl", "classification"],
    )


TABS = ["Overview", "Candidates", "Positions", "Strategies", "Learning", "Chains"]


def render() -> None:
    ui.header(
        "Options", "What the Options Brain sees, considers, holds and learns: paper and shadow kept apart."
    )
    st_ = guarded(lambda: api().get("/options/status"), "options status")
    if st_ is None:
        return
    tabs = st.tabs(TABS, key="ob_tab", on_change="rerun")  # only the open tab is computed
    if tabs[0].open:
        with tabs[0]:
            _overview(st_)
    if tabs[1].open:
        with tabs[1]:
            _candidates()
            ui.section("Missed opportunities")
            _missed()
    if tabs[2].open:
        with tabs[2]:
            _positions()
            ui.section("Greeks")
            _greeks()
            ui.section("Performance")
            _performance()
    if tabs[3].open:
        with tabs[3]:
            _strategies()
            ui.section("Research sources")
            _research()
            ui.section("Experiments")
            _experiments()
    if tabs[4].open:
        with tabs[4]:
            _learning()
            ui.section("Counterfactuals")
            _counterfactuals()
    if tabs[5].open:
        with tabs[5]:
            _chains(bool(st_.get("data_configured")))
