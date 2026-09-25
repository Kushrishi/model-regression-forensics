from __future__ import annotations

from dataclasses import dataclass

from model_forensics.exp009_nuisance import (
    build_balanced_cross_intent_refresh,
    combine_disjoint_release_changes,
)
from model_forensics.exp009_release import (
    Exp009ReleaseSlot,
    build_symmetric_label_swap_candidate,
    changed_slot_ids,
    release_sha256,
    restore_release_slots,
)

TARGET_A = "Refund_not_showing_up"
TARGET_B = "request_refund"
PER_DIRECTION = 33

EXPECTED_BASELINE_SHA256 = "cb83232c055c4c55ca50f2fbd86627d59dc806ab33a861abc7402990ddb7e20c"
EXPECTED_COMPOSITE_SHA256 = "16aaea426d1ae9a2383e9124220769957b43e1761deaa6625b930b45e664fe0f"

FROZEN_NUISANCE_PAIRS = (
    ("activate_my_card", "card_not_working"),
    ("card_about_to_expire", "getting_spare_card"),
    ("card_payment_wrong_exchange_rate", "exchange_charge"),
    ("cash_withdrawal_charge", "cash_withdrawal_not_recognised"),
)

STATES = (
    "baseline",
    "composite",
    "restore_root",
    "restore_n1",
    "restore_n2",
    "restore_n3",
    "restore_n4",
)


@dataclass(frozen=True)
class Exp009PilotStateBundle:
    baseline: tuple[Exp009ReleaseSlot, ...]
    root_change: tuple[Exp009ReleaseSlot, ...]
    nuisance_changes: tuple[tuple[Exp009ReleaseSlot, ...], ...]
    composite: tuple[Exp009ReleaseSlot, ...]
    restorations: tuple[tuple[Exp009ReleaseSlot, ...], ...]

    def state_map(self) -> dict[str, tuple[Exp009ReleaseSlot, ...]]:
        states = {
            "baseline": self.baseline,
            "composite": self.composite,
            "restore_root": self.restorations[0],
        }
        for index, release in enumerate(self.restorations[1:], start=1):
            states[f"restore_n{index}"] = release
        return states

    def candidate_change_map(self) -> dict[str, tuple[Exp009ReleaseSlot, ...]]:
        candidates = {"root": self.root_change}
        for index, release in enumerate(self.nuisance_changes, start=1):
            candidates[f"nuisance_{index}"] = release
        return candidates


def build_pilot_state_bundle(
    baseline: tuple[Exp009ReleaseSlot, ...],
) -> Exp009PilotStateBundle:
    """Construct the exact frozen nuisance-v2 development state family."""

    if release_sha256(baseline) != EXPECTED_BASELINE_SHA256:
        raise AssertionError("clean baseline release hash drift")

    root = build_symmetric_label_swap_candidate(
        baseline,
        label_a=TARGET_A,
        label_b=TARGET_B,
        per_direction=PER_DIRECTION,
    )
    root_changed = changed_slot_ids(baseline, root)

    nuisances: list[tuple[Exp009ReleaseSlot, ...]] = []
    nuisance_changed: list[tuple[str, ...]] = []
    for nuisance_index, (label_a, label_b) in enumerate(FROZEN_NUISANCE_PAIRS, start=1):
        nuisance, _selection = build_balanced_cross_intent_refresh(
            baseline,
            nuisance_index=nuisance_index,
            label_a=label_a,
            label_b=label_b,
            per_direction=PER_DIRECTION,
        )
        nuisances.append(nuisance)
        nuisance_changed.append(changed_slot_ids(baseline, nuisance))

    composite = combine_disjoint_release_changes(baseline, root, *nuisances)
    if release_sha256(composite) != EXPECTED_COMPOSITE_SHA256:
        raise AssertionError("frozen five-change composite release hash drift")

    restorations = [
        restore_release_slots(
            composite,
            baseline,
            restore_slot_ids=root_changed,
        )
    ]
    restorations.extend(
        restore_release_slots(
            composite,
            baseline,
            restore_slot_ids=restore_ids,
        )
        for restore_ids in nuisance_changed
    )

    bundle = Exp009PilotStateBundle(
        baseline=baseline,
        root_change=root,
        nuisance_changes=tuple(nuisances),
        composite=composite,
        restorations=tuple(restorations),
    )

    expected_changed_counts = {
        "baseline": 0,
        "composite": 330,
        "restore_root": 264,
        "restore_n1": 264,
        "restore_n2": 264,
        "restore_n3": 264,
        "restore_n4": 264,
    }
    for state, release in bundle.state_map().items():
        observed = len(changed_slot_ids(baseline, release))
        expected = expected_changed_counts[state]
        if observed != expected:
            raise AssertionError(
                f"{state} changed-slot count drift: expected={expected} observed={observed}"
            )

    return bundle
