import json
from pathlib import Path

import pytest

from quantpulse.core.http import HttpClient

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load_json(name: str):
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


@pytest.fixture
async def http():
    client = HttpClient(timeout=5, max_retries=1, backoff_base=0.0)
    yield client
    await client.aclose()
