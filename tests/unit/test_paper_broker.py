import pytest

from quantpulse.core.errors import DomainError
from quantpulse.domain.paper_broker import ExecutionModel, PaperBook

FREE = ExecutionModel(slippage_bps=0, commission_per_trade=0, commission_bps=0)


def test_buy_applies_slippage_commission_and_average_cost():
    model = ExecutionModel(slippage_bps=10, commission_per_trade=1.0)
    book = PaperBook(cash=10_000)
    fill = book.buy("AAPL", 10, 100.0, model)
    assert fill.price == pytest.approx(100.1) and fill.commission == 1.0
    assert book.cash == pytest.approx(10_000 - 10 * 100.1 - 1.0)
    book.buy("AAPL", 10, 120.0, FREE)
    assert book.positions["AAPL"].quantity == 20
    assert book.positions["AAPL"].avg_cost == pytest.approx((10 * 100.1 + 10 * 120.0) / 20)


def test_sell_realises_pnl_and_closes_position():
    model = ExecutionModel(slippage_bps=0, commission_per_trade=2.0)
    book = PaperBook(cash=1_000)
    book.buy("MSFT", 2, 300.0, FREE)
    fill = book.sell("MSFT", 2, 310.0, model)
    assert fill.realized_pnl == pytest.approx(2 * 10.0 - 2.0)
    assert "MSFT" not in book.positions
    assert book.cash == pytest.approx(1_000 - 600 + 620 - 2)


@pytest.mark.parametrize(
    ("action", "match"),
    [
        (lambda b: b.buy("AAPL", 1000, 100.0, FREE), "insufficient cash"),
        (lambda b: b.sell("AAPL", 1, 100.0, FREE), "short selling is disabled"),
        (lambda b: b.buy("AAPL", 0, 100.0, FREE), "positive"),
        (lambda b: b.buy("AAPL", 1, 0.0, FREE), "positive"),
    ],
)
def test_guard_rails(action, match):
    with pytest.raises(DomainError, match=match):
        action(PaperBook(cash=1_000))


def test_rebalance_from_cash_hits_targets_without_overspending():
    model = ExecutionModel(slippage_bps=5, commission_per_trade=1.0)
    book = PaperBook(cash=100_000)
    prices = {"A": 50.0, "B": 125.0, "C": 10.0}
    fills = book.rebalance(prices, {"A": 0.3, "B": 0.3, "C": 0.3}, model, cash_buffer=0.02)
    assert {f.side for f in fills} == {"buy"} and len(fills) == 3
    assert book.cash >= 0
    equity = book.equity(prices)
    for s in prices:
        assert book.quantity(s) * prices[s] / equity == pytest.approx(0.3 * 0.98, abs=0.002)
    # Only slippage and fees are lost when converting cash into stock.
    assert equity == pytest.approx(
        100_000 - sum(f.quantity * (f.price - f.reference_price) + f.commission for f in fills)
    )


def test_rebalance_sells_first_exits_dropped_names_and_scales_buys():
    book = PaperBook(cash=0)
    book.positions.clear()
    book.cash = 1_000
    book.buy("OLD", 90, 10.0, FREE)  # 900 invested, 100 cash
    prices = {"OLD": 10.0, "NEW": 20.0}
    fills = book.rebalance(prices, {"NEW": 1.0}, FREE)
    assert [f.side for f in fills] == ["sell", "buy"]
    assert "OLD" not in book.positions
    assert book.quantity("NEW") == pytest.approx(50.0)
    assert book.cash == pytest.approx(0.0, abs=1e-6)


def test_rebalance_respects_min_trade_value_and_validation():
    book = PaperBook(cash=1_000)
    book.buy("A", 49, 10.0, FREE)
    fills = book.rebalance({"A": 10.0}, {"A": 0.5}, FREE, min_trade_value=50)
    assert fills == []  # 490 held vs 500 target: drift below the threshold
    with pytest.raises(DomainError):
        book.rebalance({"A": 10.0}, {"A": 0.7, "B": 0.5}, FREE)
    with pytest.raises(DomainError, match="missing prices"):
        book.rebalance({}, {"A": 0.5}, FREE)
