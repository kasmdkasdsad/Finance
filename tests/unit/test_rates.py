import math

import pytest

from quantpulse.quant.rates import (
    bey_to_continuous,
    continuous_rate_at,
    continuous_to_bey,
    interpolate_rate,
    parse_tenor_label,
)


@pytest.mark.parametrize(
    ("label", "years"),
    [
        ("1 Mo", 1 / 12),
        ("1.5 Month", 0.125),
        ("6 Mo", 0.5),
        ("1 Yr", 1.0),
        ("30 Yr", 30.0),
        ("4 Wk", 28 / 365),
    ],
)
def test_parse_tenor_label(label, years):
    assert parse_tenor_label(label) == pytest.approx(years)


def test_parse_tenor_label_rejects_garbage():
    with pytest.raises(ValueError):
        parse_tenor_label("Date")


def test_bey_conversion_round_trip():
    assert bey_to_continuous(0.05) == pytest.approx(2 * math.log(1.025))
    assert continuous_to_bey(bey_to_continuous(0.0437)) == pytest.approx(0.0437)


def test_interpolation_and_flat_extrapolation():
    tenors, rates = [1.0, 0.25, 2.0], [0.045, 0.05, 0.04]  # unsorted on purpose
    assert interpolate_rate(tenors, rates, 0.625) == pytest.approx(0.0475)
    assert interpolate_rate(tenors, rates, 0.01) == pytest.approx(0.05)
    assert interpolate_rate(tenors, rates, 10.0) == pytest.approx(0.04)
    assert continuous_rate_at(tenors, rates, 1.0) == pytest.approx(bey_to_continuous(0.045))
