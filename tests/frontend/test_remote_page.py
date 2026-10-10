"""Home (the phone page) against a real API wired to a fake Alpaca paper account: the status line, the account,
the Brain's latest decision, the stop button, and the details folded away."""

import httpx
import pytest

from frontend.components import today_split
from frontend.views.remote import _learning
from tests.frontend.test_brain_page import _html, _texts, brain_server  # noqa: F401  (the module's fixture)
from tests.frontend.test_pages import page


def test_home_shows_the_status_the_account_and_the_brain(brain_server):  # noqa: F811
    url, _fake = brain_server
    at = page("remote", url).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    html = _html(at)
    # the one line to read first: this test server's trading switches are off (a dry run), so it is red
    assert 'class="qp-status qp-red"' in html and "trading is switched off" in html
    labels = [m.label for m in at.metric]
    assert labels == ["Equity", "Today's P&L", "Total P&L", "Buying power"]  # four cards, nothing more
    positions = next(d.value for d in at.dataframe if "Symbol" in d.value.columns)
    assert list(positions.columns) == ["Symbol", "Shares", "Value", "Today", "Total", "Return", "Stop"]
    assert "UPA" in set(positions["Symbol"])
    # the table is Alpaca's positions read live with the account, and today's P&L is split between them
    acct = httpx.get(f"{url}/api/v1/trading/account").json()
    live = {p["symbol"]: p for p in httpx.get(f"{url}/api/v1/trading/positions").json()}
    row = positions[positions["Symbol"] == "UPA"].iloc[0]
    assert row["Value"] == live["UPA"]["market_value"] and row["Today"] == live["UPA"]["intraday_pl"]
    assert positions["Value"].sum() == pytest.approx(acct["long_market_value"])
    split = next(c.value for c in at.caption if c.value.startswith("Today's P&L"))
    assert "$" not in split.replace("\\$", "")  # every amount escaped: never rendered as a formula
    assert "Supervisor" in html and "Next cycle" in html and "Learning" in html  # the Brain at a glance
    assert any("**Latest decision**" in m.value for m in at.markdown)
    assert any(b.label == "STOP BRAIN TRADING" for b in at.button)
    expanders = [e.label for e in [*at.expander, *at.status]]
    assert "System details" in expanders and "Recent alerts" in expanders
    text = _texts(at)
    assert "Paper endpoint" in text and "Autonomous PAPER execution" in text  # folded away, still there
    for secret in ("ui-secret", "PKUITEST"):
        assert secret not in text


def test_one_tap_stops_brain_trading_and_releasing_takes_a_deliberate_second_step(brain_server):  # noqa: F811
    url, fake = brain_server
    at = page("remote", url).run()
    next(b for b in at.button if b.label == "STOP BRAIN TRADING").click().run()
    assert not at.exception
    kill = httpx.get(f"{url}/api/v1/brain/kill-switch").json()
    assert kill["active"] and kill["reason"] == "stopped from the phone"
    assert any("BRAIN TRADING STOPPED" in e.value for e in at.error)
    assert "Brain trading is stopped" in _html(at)  # and the status line says so first
    release = next(b for b in at.button if b.label == "Allow Brain orders again")
    assert release.disabled  # not until the box is ticked
    next(c for c in at.checkbox if c.key == "remote_release_sure").check().run()
    next(b for b in at.button if b.label == "Allow Brain orders again").click().run()
    assert not httpx.get(f"{url}/api/v1/brain/kill-switch").json()["active"]
    assert fake.orders == {}  # nothing was ever sent from this page


def test_the_learning_line_calls_a_small_sample_unproven():
    few = {"open": 40, "evaluated": 6, "hit_rate": 0.6667, "next_due": "2026-10-05"}
    line = _learning({"predictions": few, "min_observations": 20})
    assert line == "6 graded · 67% right (unproven) · 40 open · next graded 2026-10-05"
    enough = {**few, "evaluated": 25, "hit_rate": 0.56}
    assert "56% right ·" in _learning({"predictions": enough, "min_observations": 20})
    assert _learning({"predictions": {"open": 3, "evaluated": 0, "hit_rate": None}}) == "0 graded · 3 open"
    assert _learning(None) is None  # the API had nothing to say: the fact is left out


def test_todays_pnl_splits_into_the_open_positions_and_the_rest():
    positions = [{"intraday_pl": 120.0}, {"intraday_pl": -20.0}]
    assert today_split(100.0, positions) == "Today's P&L +$100.00: open positions +$100.00"
    # a position sold today: its gain is in the account's figure, not in any open position
    assert today_split(160.0, positions) == (
        "Today's P&L +$160.00: open positions +$100.00, closed trades and fees +$60.00"
    )
    assert (
        today_split(-35.0, []) == "Today's P&L -$35.00: open positions +$0.00, closed trades and fees -$35.00"
    )
    assert today_split(None, positions) is None and today_split(10.0, None) is None
