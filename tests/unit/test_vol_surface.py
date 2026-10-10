import math
from datetime import UTC, date, datetime

import pytest

from quantpulse.quant.black_scholes import bsm_price
from quantpulse.quant.vol_surface import build_vol_surface, years_to_expiry
from quantpulse.schemas.options import OptionContract

AS_OF = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)  # 10:00 New York
SPOT = 200.0
R = 0.04
Q = 0.005


def smile_vol(k_over_s: float, t: float) -> float:
    return 0.22 + 0.25 * (k_over_s - 1.0) ** 2 - 0.05 * (k_over_s - 1.0) + 0.02 * t


def make_chain(expiries):
    contracts = []
    for exp in expiries:
        t = years_to_expiry(exp, AS_OF)
        for strike in range(140, 262, 5):
            vol = smile_vol(strike / SPOT, t)
            for kind in ("call", "put"):
                px = bsm_price(SPOT, strike, t, R, vol, Q, kind)
                contracts.append(
                    OptionContract(
                        contract_symbol=f"X{exp:%y%m%d}{kind[0].upper()}{strike:08d}",
                        kind=kind,
                        strike=strike,
                        expiration=exp,
                        bid=max(px - 0.01, 0.0),
                        ask=px + 0.01,
                        last=px,
                    )
                )
    return contracts


def test_years_to_expiry_uses_4pm_new_york_cutoff():
    # 2026-09-25 16:00 EDT == 20:00 UTC -> 6 hours after AS_OF
    assert years_to_expiry(date(2026, 9, 25), AS_OF) == pytest.approx(6 / (365 * 24))


def test_surface_recovers_input_smile():
    expiries = [date(2026, 10, 16), date(2026, 11, 20), date(2027, 1, 15)]
    surface = build_vol_surface("TEST", make_chain(expiries), SPOT, AS_OF, lambda t: R, Q)
    assert surface.expirations == expiries
    assert surface.points_used > 60
    for smile in surface.smiles:
        for p in smile.points:
            # mids are the exact model prices, so the solved IV must equal the generating vol
            assert p.iv == pytest.approx(smile_vol(p.strike / SPOT, p.years), abs=2e-3)
            assert p.iv_source == "model"
            # OTM only: puts below the forward, calls at/above it
            if p.kind == "put":
                assert p.strike < smile.forward
            else:
                assert p.strike >= smile.forward
        assert smile.fit is not None
        assert smile.skew_90_110 == pytest.approx(
            smile_vol(0.9, smile.years) - smile_vol(1.1, smile.years), abs=5e-3
        )
    # Grid rows align with expiries and are None outside quoted strikes.
    assert len(surface.iv_grid) == 3 and len(surface.iv_grid[0]) == len(surface.moneyness_grid)
    atm_col = min(range(len(surface.moneyness_grid)), key=lambda i: abs(surface.moneyness_grid[i] - 1))
    assert surface.iv_grid[0][atm_col] == pytest.approx(
        smile_vol(surface.moneyness_grid[atm_col], surface.years[0]), abs=3e-3
    )


def test_expired_wide_and_empty_quotes_rejected():
    exp = date(2026, 10, 16)
    bad = [
        OptionContract(
            contract_symbol="A", kind="call", strike=210, expiration=date(2026, 9, 25), bid=1, ask=1.1
        ),
        OptionContract(contract_symbol="B", kind="call", strike=210, expiration=exp, bid=0.1, ask=5.0),
        OptionContract(contract_symbol="C", kind="call", strike=210, expiration=exp),
    ]
    surface = build_vol_surface("TEST", bad, SPOT, AS_OF, lambda t: R, Q)
    assert surface.points_used == 0
    assert surface.points_rejected == 3
    assert surface.smiles == []


def test_vendor_iv_used_when_price_unsolvable():
    exp = date(2026, 12, 18)
    c = OptionContract(
        # A bad print above the spot price cannot be reproduced by any volatility.
        contract_symbol="D",
        kind="call",
        strike=260,
        expiration=exp,
        bid=0.0,
        ask=0.0,
        last=250.0,
        implied_volatility=0.41,
    )
    surface = build_vol_surface("TEST", [c], SPOT, AS_OF, lambda t: R, Q)
    assert surface.points_used == 1
    assert surface.smiles[0].points[0].iv_source == "vendor"
    assert surface.smiles[0].points[0].iv == pytest.approx(0.41)
    assert math.isfinite(surface.smiles[0].points[0].log_moneyness)
