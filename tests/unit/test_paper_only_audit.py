"""A standing audit of the repository: there is no path to live trading and no way around the order path.

These read the source itself, so a future change that adds a live endpoint, a second way to send an order,
a Brain module that talks to the broker directly, or a proposal that could loosen a protected control fails
the build — whatever the rest of the tests happen to exercise.
"""

import re
from pathlib import Path

from quantpulse.brain import improvement
from quantpulse.services import preflight

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "quantpulse"
SCANNED = [
    *SRC.rglob("*.py"),
    *(ROOT / "frontend").rglob("*.py"),
    *(ROOT / "deploy").rglob("*"),
    *(ROOT / "launcher").rglob("*"),
    *(ROOT / "scripts").rglob("*"),
    ROOT / "render.yaml",
    ROOT / ".env.example",
    ROOT / "pyproject.toml",
    ROOT / "Dockerfile",
]
LIVE = re.compile(r"(?<![\w.-])(api|broker-api)\.alpaca\.markets", re.IGNORECASE)
BROKER_WRITE = re.compile(r"\b_?broker\.(submit|cancel|cancel_all|close_position|close_all_positions)\(")


def text_files() -> list[Path]:
    out = []
    for p in SCANNED:
        if p.is_file() and p.suffix not in (".png", ".ico", ".icns", ".pyc") and "__pycache__" not in p.parts:
            out.append(p)
    return out


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def test_no_live_alpaca_endpoint_anywhere():
    found = []
    for p in text_files():
        for n, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if LIVE.search(line) and "LIVE_ENDPOINT = re.compile" not in line:
                found.append(f"{rel(p)}:{n}: {line.strip()[:120]}")
    assert found == []


def test_the_sdk_client_is_only_ever_built_for_paper_without_a_url_override():
    mentions, builds = [], []
    for p in SRC.rglob("*.py"):
        text = p.read_text()
        assert "paper=False" not in text.replace(" ", ""), rel(p)
        assert "url_override" not in text, rel(p)
        mentions += [m.group(0) for m in re.finditer(r"TradingClient\([^)]*\)", text)]
        builds += [(rel(p), m.group(0)) for m in re.finditer(r"=\s*TradingClient\([^)]*\)", text)]
    assert mentions and all("paper=True" in m for m in mentions)
    assert builds == [
        (
            "src/quantpulse/providers/alpaca_trading.py",
            "= TradingClient(self._key_id, self._secret, paper=True)",
        )
    ]


def test_orders_leave_only_through_the_order_manager_and_the_trading_service():
    writers = {}
    for p in SRC.rglob("*.py"):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if BROKER_WRITE.search(line):
                writers.setdefault(rel(p), []).append(n)
    assert set(writers) == {"src/quantpulse/services/order_manager.py", "src/quantpulse/services/trading.py"}
    submits = [
        (rel(p), n)
        for p in SRC.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"\b_?broker\.submit\(", line)
    ]
    assert [f for f, _ in submits] == ["src/quantpulse/services/order_manager.py"]  # one submission call
    order_manager = [
        rel(p) for p in SRC.rglob("*.py") if re.search(r"\bOrderManager\(", p.read_text()) and p.name != "order_manager.py"
    ]  # fmt: skip
    assert order_manager == ["src/quantpulse/services/trading.py"]  # built once, by the trading service


def test_every_order_the_trading_service_sends_passes_the_last_gate_first():
    text = (SRC / "services" / "trading.py").read_text()
    calls = [m.start() for m in re.finditer(r"await self\.orders\.submit\(", text)]
    assert len(calls) == 2  # the Brain/strategy path and the confirmed diagnostic test order
    for at in calls:
        before = text[max(0, at - 6000) : at]
        assert "await self.pre_submit_blockers(" in before, "an order path without the last gate"


def test_the_brain_never_talks_to_the_broker_or_builds_an_order_path():
    for p in (SRC / "brain").rglob("*.py"):
        text = p.read_text()
        assert not BROKER_WRITE.search(text), rel(p)
        assert "OrderManager(" not in text and "TradingClient" not in text, rel(p)
        assert ".orders.submit(" not in text, rel(p)


def test_a_proposal_can_never_touch_a_control_the_preflight_protects():
    for setting, _rule in preflight.PROTECTED:
        env = "QP_" + setting.upper()
        assert any(env.startswith(prefix) for prefix in improvement.PROTECTED), env
        assert improvement.withheld({"change": f"set {env} to something looser"}) is not None
    for switch in ("QP_ALPACA_PAPER", "QP_BRAIN_KILL_SWITCH", "QP_TRADING_KILL_SWITCH", "QP_BRAIN_MODE"):
        assert improvement.withheld({"change": f"{switch}=false"}) == switch


def test_the_paper_setting_cannot_be_turned_off():
    import pytest

    from quantpulse.config import Settings

    with pytest.raises(ValueError):
        Settings(_env_file=None, alpaca_paper=False)
