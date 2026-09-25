from __future__ import annotations

from model_forensics.exp009_pilot_states import (
    EXPECTED_BASELINE_SHA256,
    EXPECTED_COMPOSITE_SHA256,
    STATES,
)


def test_frozen_pilot_constants() -> None:
    assert EXPECTED_BASELINE_SHA256 == (
        "cb83232c055c4c55ca50f2fbd86627d59dc806ab33a861abc7402990ddb7e20c"
    )
    assert EXPECTED_COMPOSITE_SHA256 == (
        "16aaea426d1ae9a2383e9124220769957b43e1761deaa6625b930b45e664fe0f"
    )
    assert STATES == (
        "baseline",
        "composite",
        "restore_root",
        "restore_n1",
        "restore_n2",
        "restore_n3",
        "restore_n4",
    )
