"""The phone page against a real API wired to a fake Alpaca paper account: health, the stop button, P&L."""

import httpx

from tests.frontend.test_brain_page import _texts, brain_server  # noqa: F401  (the module's fixture)
from tests.frontend.test_pages import assert_clean, page


def test_the_remote_page_shows_health_the_brain_and_the_account(brain_server):  # noqa: F811
    url, _fake = brain_server
    at = page("remote", url).run()
    assert_clean(at)
    text = _texts(at)
    assert "Brain supervisor" in text and "Equity" in text and "Day P&L" in text
    assert any(b.label == "STOP BRAIN TRADING" for b in at.button)
    assert "Cycle #" in text  # the last cycle, its agents and consensus
    # CLOUD and TRADING from /brain/cloud-status
    assert (
        "Cloud" in text and "Trading" in text and "Paper endpoint" in text and "Orders / fills today" in text
    )
    assert "Autonomous PAPER execution" in " ".join(
        [*(w.value for w in at.warning), *(s.value for s in at.success)]
    )
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
    release = next(b for b in at.button if b.label == "Allow Brain orders again")
    assert release.disabled  # not until the box is ticked
    next(c for c in at.checkbox if c.key == "remote_release_sure").check().run()
    next(b for b in at.button if b.label == "Allow Brain orders again").click().run()
    assert not httpx.get(f"{url}/api/v1/brain/kill-switch").json()["active"]
    assert fake.orders == {}  # nothing was ever sent from this page
