"""Home (the phone page) against a real API wired to a fake Alpaca paper account: the status line, the account,
the Brain's latest decision, the stop button, and the details folded away."""

import httpx

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
    assert list(positions.columns) == ["Symbol", "Shares", "Value", "P&L", "Return"]
    assert "UPA" in set(positions["Symbol"])
    assert "Supervisor" in html and "Next cycle" in html  # the Brain at a glance
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
