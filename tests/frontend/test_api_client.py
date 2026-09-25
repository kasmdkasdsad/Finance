import httpx
import pytest
import respx

from frontend.api_client import ApiClient, ApiError
from frontend.components import money, num, pct


@respx.mock
def test_structured_errors_are_parsed():
    respx.get("http://api.test/api/v1/market/quote/BAD").mock(
        return_value=httpx.Response(
            422,
            json={"error": "validation_error", "detail": [{"loc": ["path", "symbol"], "msg": "bad symbol"}]},
        )
    )
    client = ApiClient("http://api.test", token="t0k")
    with pytest.raises(ApiError) as exc:
        client.get("/market/quote/BAD")
    assert exc.value.status == 422 and exc.value.message == "symbol: bad symbol"
    assert respx.calls[0].request.headers["X-API-Key"] == "t0k"


@respx.mock
def test_success_and_no_content():
    respx.get("http://api.test/api/v1/rates/at").mock(return_value=httpx.Response(200, json={"data": 1}))
    respx.delete("http://api.test/api/v1/portfolios/1").mock(return_value=httpx.Response(204))
    client = ApiClient("http://api.test")
    assert client.get("/rates/at", years=1, ignored=None) == {"data": 1}
    assert respx.calls[0].request.url.params == httpx.QueryParams({"years": "1"})
    assert client.delete("/portfolios/1") is None


def test_unreachable_api_is_reported():
    client = ApiClient("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(ApiError) as exc:
        client.get("/market/session")
    assert exc.value.status == 0 and "cannot reach" in exc.value.message


def test_formatters():
    assert (
        money(3.7e12) == "$3.70T"
        and money(-2.5e9, 1) == "-$2.5B"
        and money(12.5) == "$12.50"
        and money(None) == "—"
    )
    assert pct(0.1234) == "12.34%" and pct(-0.01, 1, signed=True) == "-1.0%"
    assert num(1234.5, 1) == "1,234.5" and num(None) == "—"
