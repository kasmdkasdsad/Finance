"""Track Record: every logged prediction, graded against what actually happened."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, guarded, job_progress, num, pct

SOURCE_LABEL = {"forecast": "Price forecast", "model": "Stock model"}


def _reliability(score: dict[str, Any], key: str) -> None:
    buckets = [b for b in score["calibration"] if b["n"] > 0]
    if not buckets:
        return
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
            x=[b["mean_predicted"] for b in buckets],
            y=[b["observed"] for b in buckets],
            mode="lines+markers",
            name="Observed",
            line={"color": charts.series(0), "width": 2},
            marker={"size": [max(8, min(22, 6 + b["n"] ** 0.5)) for b in buckets]},
            customdata=[b["n"] for b in buckets],
            hovertemplate="predicted %{x:.0%} → happened %{y:.0%} (n=%{customdata})<extra></extra>",
        )
    )
    what = "price higher" if score["source"] == "forecast" else "beat the benchmark"
    charts.base_layout(fig, f"Reliability: predicted vs actual ({what})", height=320)
    fig.update_xaxes(title="Predicted probability", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="How often it happened", range=[0, 1], tickformat=".0%")
    charts.show(fig, key=key)


def _count(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 10_000 else f"{n:,}"


def _score_panel(score: dict[str, Any]) -> None:
    title = f"{SOURCE_LABEL[score['source']]} · {score['horizon_days']} trading days"
    with st.container(border=True):
        st.markdown(f"**{title}**")
        k = st.columns(5)
        k[0].metric(
            "Graded",
            _count(score["resolved"]),
            f"{score['open']:,} open",
            delta_color="off",
            delta_arrow="off",
        )
        k[1].metric(
            "Brier score",
            num(score["brier"], 3),
            help="Mean squared error of the probabilities; 0.25 = coin flip",
        )
        k[2].metric(
            "Brier skill", num(score["brier_skill"], 3), help="> 0 beats always guessing the base rate"
        )
        k[3].metric("Hit rate", pct(score["hit_rate"], 0))
        if score["source"] == "forecast":
            k[4].metric("Inside 90% range", pct(score["coverage_90"], 0), help="Should be close to 90%")
        else:
            gap = None
            if score["top_ranked_excess"] is not None and score["others_excess"] is not None:
                gap = score["top_ranked_excess"] - score["others_excess"]
            k[4].metric(
                "Top-5 vs rest",
                pct(gap, 2, signed=True) if gap is not None else "—",
                help="Excess return gap",
            )
        if score["resolved"]:
            _reliability(score, f"rel_{score['source']}_{score['horizon_days']}")
        else:
            st.caption("Nothing graded yet: predictions are graded on their target date.")


ORIGINS = {"Live record": "live", "Historical replay": "backfill", "Both": "all"}


def _backfill_panel() -> None:
    """Start the point-in-time replay and show its progress."""
    status = guarded(lambda: api().get("/predictions/backfill"), "backfill status")
    job = (status or {}).get("job")
    with st.container(border=True):
        st.markdown("**Historical replay**")
        st.caption(
            "Replays the last few years point-in-time: each prediction uses only data available on its date and is "
            "graded against what happened next. The replay is kept apart from the live record."
        )
        c = st.columns([1, 1, 2], vertical_alignment="bottom")
        replace = c[1].toggle("Redo from scratch", value=False, key="track_backfill_replace")
        if c[0].button(
            "Run the replay", icon=":material/history:", disabled=bool(job and job["status"] == "running")
        ):
            guarded(lambda: api().post("/predictions/backfill", replace=replace), "backfill")
            st.rerun()
        if job and job["status"] == "running":
            job_progress(job)
        elif status and status.get("result"):
            r = status["result"]
            c[2].caption(
                f"Last replay: {r['forecast_rows']} forecasts and {r['model_rows']} model rankings "
                f"({r['first_date']} → {r['last_date']})."
                + (
                    " Skipped: " + "; ".join(f"{k} ({v})" for k, v in list(r["skipped"].items())[:4])
                    if r["skipped"]
                    else ""
                )
            )
        if status and status["counts"]:
            counts = pd.DataFrame(status["counts"])
            st.dataframe(counts, hide_index=True)


def render() -> None:
    st.title("Track Record")
    st.caption(
        "After each close the platform logs its forecasts and stock-model calls from live prices only, then grades "
        "each one on its target date. This page is how you find out whether any of it works."
    )
    c = st.columns([1, 1, 1, 2], vertical_alignment="bottom")
    symbol = c[3].text_input("Filter by ticker (optional)", "", key="track_symbol").strip().upper() or None
    choice = c[2].segmented_control("Record", list(ORIGINS), default="Live record", key="track_origin")
    origin = ORIGINS[choice or "Live record"]
    if c[0].button(
        "Log today's predictions", icon=":material/edit_note:", help="Runs automatically after the close"
    ):
        out = guarded(lambda: api().post("/predictions/log"), "logging")
        if out:
            st.success(
                f"Logged {out['logged']} predictions for {out['made_on']}.", icon=":material/check_circle:"
            )
            if out["skipped"]:
                st.caption("Skipped: " + "; ".join(f"{k} ({v})" for k, v in list(out["skipped"].items())[:8]))
    if c[1].button("Grade due predictions", icon=":material/grading:"):
        out = guarded(lambda: api().post("/predictions/resolve"), "grading")
        if out:
            st.success(
                f"Graded {out['resolved']}, voided {out['voided']}, {out['pending']} still waiting for prices."
            )
    _backfill_panel()
    card = guarded(lambda: api().get("/predictions/scorecard", symbol=symbol, origin=origin), "scorecard")
    if not card:
        return
    if not card["sources"]:
        hint = (
            "Run the historical replay above to score the platform on past data."
            if origin == "backfill"
            else "With live data enabled they are logged automatically after each close (QP_PREDICTIONS_LOG_TIME), "
            "or use the button above after 4 pm ET. The historical replay gives a scorecard right away."
        )
        st.info(f"No predictions in this record yet. {hint}", icon=":material/hourglass_empty:")
        return
    if origin != "live":
        st.caption(
            "The replay is point-in-time but not live: it uses today's risk-free rate and dividend yield and no "
            "options. Trust the live record more once it has a few months of results."
        )
    for score in card["sources"]:
        _score_panel(score)
    st.subheader("Recent predictions")
    recent = pd.DataFrame(card["recent"])
    if not recent.empty:
        st.dataframe(
            recent[
                [
                    "made_on",
                    "target_date",
                    "symbol",
                    "origin",
                    "source",
                    "horizon_days",
                    "reference_price",
                    "prob_up",
                    "prob_outperform",
                    "q05",
                    "q95",
                    "status",
                    "realized_price",
                    "realized_return",
                    "outcome_up",
                    "outcome_outperform",
                    "in_90",
                ]
            ],
            hide_index=True,
            column_config={
                "prob_up": st.column_config.NumberColumn("P(up)", format="percent"),
                "prob_outperform": st.column_config.NumberColumn("P(beat)", format="percent"),
                "realized_return": st.column_config.NumberColumn("return", format="percent"),
                "q05": st.column_config.NumberColumn("5%", format="$%.2f"),
                "q95": st.column_config.NumberColumn("95%", format="$%.2f"),
            },
        )
    with st.expander("Reliability tables"):
        for score in card["sources"]:
            st.markdown(f"**{SOURCE_LABEL[score['source']]} · {score['horizon_days']}d**")
            st.dataframe(pd.DataFrame(score["calibration"]), hide_index=True)
    st.caption(card["note"])
