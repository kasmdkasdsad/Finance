"""The dashboard's newer building blocks: the strategy ladder, the equity chart, the option limits, and text with
dollar amounts that Markdown must not read as a formula."""

from frontend import charts, ui
from frontend.views import options_brain


def _capture(monkeypatch) -> list[str]:
    out: list[str] = []
    monkeypatch.setattr(ui.st, "html", lambda body: out.append(str(body)))
    return out


def test_the_ladder_counts_each_rung_and_marks_the_ones_that_trade(monkeypatch):
    html = _capture(monkeypatch)
    monkeypatch.setattr(ui, "section", lambda *a, **k: None)
    stages = {
        "RESEARCH": 2,
        "EXTRACTED": 3,
        "VALIDATION": 4,
        "WALK_FORWARD": 2,
        "PAPER_SHADOW": 1,
        "RETIRED": 7,
    }
    options_brain._ladder(stages, exploring=False)
    [body] = html
    steps = body.split('<div class="qp-step')[1:]
    assert len(steps) == len(options_brain.LADDER) == 7
    assert "<b>5</b><span>Ideas</span>" in steps[0]  # RESEARCH and EXTRACTED together
    assert "qp-step-on" in steps[2] and "<b>4</b>" in steps[2]
    assert "qp-step-on" not in steps[1]  # nothing backtested right now
    assert "qp-step-live qp-step-on" in steps[4] and "<b>1</b><span>Shadow</span>" in steps[4]
    assert all("qp-step-live" in s for s in steps[4:]) and not any("qp-step-live" in s for s in steps[:4])
    # with exploration on, validated and walk-forward strategies trade too (one exploration contract)
    html.clear()
    options_brain._ladder(stages, exploring=True)
    steps = html[0].split('<div class="qp-step')[1:]
    for rung in steps[2:4]:
        assert "qp-step-live qp-step-on" in rung and "+1 contract" in rung
    assert not any("qp-step-live" in s for s in steps[:2])  # ideas and backtests never trade


def test_the_ladder_escapes_its_labels(monkeypatch):
    html = _capture(monkeypatch)
    ui.ladder([ui.Step("<b>x</b>", 1, "a & b")])
    assert "&lt;b&gt;x&lt;/b&gt;" in html[0] and "a &amp; b" in html[0]


def test_the_equity_chart_is_green_up_red_down_and_fitted_to_the_data():
    up = charts.equity([("2026-09-01", 100_000.0), ("2026-09-02", 101_000.0), ("2026-09-03", 102_500.0)])
    down = charts.equity([("2026-09-01", 100_000.0), ("2026-09-02", 98_000.0)])
    assert (
        up.data[1].line.color == charts.STATUS["good"]
        and down.data[1].line.color == charts.STATUS["critical"]
    )
    lo, hi = up.layout.yaxis.range
    assert 99_000 < lo < 100_000 < 102_500 < hi < 103_500  # not stretched down to zero
    assert up.data[1].fill == "tonexty" and up.data[0].hoverinfo == "skip"  # the floor trace is invisible


def test_the_option_limits_read_as_words_not_json(monkeypatch):
    html = _capture(monkeypatch)
    options_brain._limits({"max_loss_per_trade": 1500.0, "max_loss_pct_per_trade": 0.02, "max_total_risk_pct": 0.15,
                           "max_underlying_risk_pct": 0.05, "max_positions": 12, "max_contracts": 20, "dte": [7, 60],
                           "close_dte": 2, "max_delta_pct": 0.5, "max_vega_pct": 0.01, "max_spread_pct": 0.15,
                           "max_quote_age_seconds": 120.0, "exploration": True, "exploration_max_loss": 1000.0})  # fmt: skip
    [body] = html
    for text in ("$1,500 · 2%", "15%", "12", "7–60 (closed at 2)", "50% / 1.0%", "on · $1,000", "120 s"):
        assert text in body, text


def test_dollar_amounts_never_turn_into_a_formula():
    """Streamlit Markdown reads text between two dollar signs as LaTeX: every line with two amounts is escaped."""
    from frontend.components import today_split

    line = today_split(160.0, [{"intraday_pl": 120.0}, {"intraday_pl": -20.0}])
    assert line is not None and line.count("$") == 3
    assert ui.md(line).count("\\$") == 3
