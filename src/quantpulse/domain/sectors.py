"""Industry classification: SEC SIC codes mapped to the Fama-French 12 industries.

The SEC publishes every registrant's four-digit Standard Industrial Classification code, so this works
for any US-listed stock without a paid data feed. The ranges below are Kenneth French's published
``Siccodes12`` definitions; anything unmatched (including missing codes) is "Other".
"""

from __future__ import annotations

FF12_NAMES: dict[str, str] = {
    "NoDur": "Consumer non-durables",
    "Durbl": "Consumer durables",
    "Manuf": "Manufacturing",
    "Enrgy": "Energy",
    "Chems": "Chemicals",
    "BusEq": "Business equipment & tech",
    "Telcm": "Telecom",
    "Utils": "Utilities",
    "Shops": "Retail & wholesale",
    "Hlth": "Healthcare",
    "Money": "Finance",
    "Other": "Other",
}

_RANGES: list[tuple[str, int, int]] = [
    ("NoDur", 100, 999), ("NoDur", 2000, 2399), ("NoDur", 2700, 2749), ("NoDur", 2770, 2799),
    ("NoDur", 3100, 3199), ("NoDur", 3940, 3989),
    ("Durbl", 2500, 2519), ("Durbl", 2590, 2599), ("Durbl", 3630, 3659), ("Durbl", 3710, 3711),
    ("Durbl", 3714, 3714), ("Durbl", 3716, 3716), ("Durbl", 3750, 3751), ("Durbl", 3792, 3792),
    ("Durbl", 3900, 3939), ("Durbl", 3990, 3999),
    ("Manuf", 2520, 2589), ("Manuf", 2600, 2699), ("Manuf", 2750, 2769), ("Manuf", 3000, 3099),
    ("Manuf", 3200, 3569), ("Manuf", 3580, 3629), ("Manuf", 3700, 3709), ("Manuf", 3712, 3713),
    ("Manuf", 3715, 3715), ("Manuf", 3717, 3749), ("Manuf", 3752, 3791), ("Manuf", 3793, 3799),
    ("Manuf", 3830, 3839), ("Manuf", 3860, 3899),
    ("Enrgy", 1200, 1399), ("Enrgy", 2900, 2999),
    ("Chems", 2800, 2829), ("Chems", 2840, 2899),
    ("BusEq", 3570, 3579), ("BusEq", 3660, 3692), ("BusEq", 3694, 3699), ("BusEq", 3810, 3829),
    ("BusEq", 7370, 7379),
    ("Telcm", 4800, 4899),
    ("Utils", 4900, 4949),
    ("Shops", 5000, 5999), ("Shops", 7200, 7299), ("Shops", 7600, 7699),
    ("Hlth", 2830, 2839), ("Hlth", 3693, 3693), ("Hlth", 3840, 3859), ("Hlth", 8000, 8099),
    ("Money", 6000, 6999),
]  # fmt: skip


def ff12(sic: str | int | None) -> str:
    """Fama-French 12-industry code for a SIC code ("Other" when unknown)."""
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return "Other"
    for name, lo, hi in _RANGES:
        if lo <= code <= hi:
            return name
    return "Other"


def ff12_label(sic: str | int | None) -> str:
    return FF12_NAMES[ff12(sic)]
