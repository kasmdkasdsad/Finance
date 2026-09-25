"""Sports Hub: live NFL / college football scores, win probabilities and Elo power ratings."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend import charts
from frontend.components import api, composite_badges, guarded, num


def _game_card(g: dict) -> None:
    game, wp = g["game"], g["win_probability"]
    home, away = game["home"], game["away"]

    def label(side: dict) -> str:
        t = side["team"]
        rank = f"#{t['rank']} " if t.get("rank") else ""
        rec = f" ({t['record']})" if t.get("record") else ""
        return f"{rank}{t['name']}{rec}"

    with st.container(border=True):
        a, b = st.columns([3, 2])
        score = "" if game["state"] == "pre" else f"{away['score']} – {home['score']}"
        a.markdown(
            f"**{label(away)}**  at  **{label(home)}**  \n{game['status_detail']} {('· ' + score) if score else ''}"
        )
        line = game.get("market")
        if line and line.get("spread_home") is not None:
            spread = line["spread_home"]
            fav = home["team"]["abbreviation"] if spread < 0 else away["team"]["abbreviation"]
            a.caption(
                f"Line: {fav} -{abs(spread):g} · O/U {num(line.get('over_under'), 1)} · {line['source']}"
                if spread != 0
                else f"Line: pick'em · {line['source']}"
            )
        p_home = wp["home"]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=[""],
                x=[wp["away"] * 100],
                orientation="h",
                name=away["team"]["abbreviation"],
                marker={"color": charts.series(1)},
                text=[f"{away['team']['abbreviation']} {wp['away']:.0%}"],
                textposition="inside",
                insidetextanchor="start",
                hovertemplate="%{x:.1f}%<extra>" + away["team"]["name"] + "</extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                y=[""],
                x=[p_home * 100],
                orientation="h",
                name=home["team"]["abbreviation"],
                marker={"color": charts.series(0)},
                text=[f"{home['team']['abbreviation']} {p_home:.0%}"],
                textposition="inside",
                insidetextanchor="end",
                hovertemplate="%{x:.1f}%<extra>" + home["team"]["name"] + "</extra>",
            )
        )
        fig.update_traces(
            marker_line_color=charts.theme()["surface"], marker_line_width=2, textfont_color="#ffffff"
        )
        charts.base_layout(
            fig, None, height=70, barmode="stack", showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0}
        )
        fig.update_xaxes(visible=False, range=[0, 100])
        fig.update_yaxes(visible=False)
        with b:
            charts.show(fig, key=f"wp_{game['event_id']}")
            extra = (
                f" · ESPN {game['espn_home_win_prob']:.0%}"
                if game.get("espn_home_win_prob") is not None
                else ""
            )
            st.caption(
                f"Exp. margin {wp['expected_margin_home']:+.1f} (home) · pregame {wp['pregame_home']:.0%} · Elo {wp['elo_home']:.0%}{extra}"
            )


def render() -> None:
    st.title("Sports Hub")
    league = st.segmented_control(
        "League",
        ["nfl", "college-football"],
        default="nfl",
        format_func=lambda v: "NFL" if v == "nfl" else "College Football (FBS)",
    )
    league = league or "nfl"
    view = st.segmented_control(
        "View", ["Scoreboard", "Power ratings"], default="Scoreboard", key="sports_view"
    )
    if (view or "Scoreboard") == "Scoreboard":
        refresh = st.session_state.get("refresh_seconds")

        @st.fragment(run_every=refresh)
        def board() -> None:
            res = guarded(lambda: api().get(f"/sports/{league}/scoreboard"), "scoreboard")
            if not res:
                return
            d = res["data"]
            composite_badges(res["meta"])
            st.caption(
                f"Season {d['season']} · week {d['week']} · {len(d['games'])} games · model: {d['games'][0]['win_probability']['model'] if d['games'] else '—'}"
            )
            order = {"in": 0, "pre": 1, "post": 2}
            for g in sorted(d["games"], key=lambda g: (order[g["game"]["state"]], g["game"]["start_time"])):
                _game_card(g)

        board()
    else:
        res = guarded(lambda: api().get(f"/sports/{league}/ratings"), "ratings")
        if not res:
            return
        d = res["data"]
        composite_badges(res["meta"])
        rows = [
            {
                "rank": r["rank"],
                "team": r["team"]["name"],
                "rating": r["rating"],
                "W-L-T": f"{r['wins']}-{r['losses']}-{r['ties']}",
                "avg margin": r["avg_margin"],
                "last change": r["last_change"],
            }
            for r in d["ratings"]
        ]
        df = pd.DataFrame(rows)
        top = df.head(25)
        fig = go.Figure(
            go.Bar(
                x=top["rating"],
                y=top["team"],
                orientation="h",
                marker={"color": charts.series(0), "cornerradius": 4},
                hovertemplate="%{y}: %{x:.0f}<extra></extra>",
                name="Elo",
            )
        )
        charts.base_layout(
            fig,
            f"Elo power ratings (top 25 of {len(df)}) · {d['games_processed']} games processed",
            height=640,
            showlegend=False,
        )
        fig.update_xaxes(
            range=[min(top["rating"]) - 40, max(top["rating"]) + 20],
            showgrid=True,
            gridcolor=charts.theme()["grid"],
        )
        fig.update_yaxes(autorange="reversed")
        charts.show(fig)
        st.dataframe(df, hide_index=True)
        p = d["params"]
        st.caption(
            f"K={p['k']:g}, home field={p['home_field']:g} Elo, {p['points_per_elo']:g} Elo/pt, season regression {p['season_regression']:.0%}, "
            f"margin σ={p['margin_sd']} pts, market weight {p['market_weight']:.0%}."
        )
