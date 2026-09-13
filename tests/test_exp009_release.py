from __future__ import annotations

import json

import pytest

from model_forensics.exp009_data import Banking77Record, content_id_for_record
from model_forensics.exp009_release import (
    build_clean_release_slots,
    build_symmetric_label_swap_candidate,
    changed_slot_ids,
    release_diff_manifest,
    release_sha256,
    restore_release_slots,
    select_symmetric_swap_slot_ids,
)


def _record(label: str, text: str) -> Banking77Record:
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def _records() -> tuple[Banking77Record, ...]:
    rows = [
        *(_record("intent_a", f"PRIVATE-A-{index}") for index in range(8)),
        *(_record("intent_b", f"PRIVATE-B-{index}") for index in range(9)),
        *(_record("other", f"PRIVATE-O-{index}") for index in range(6)),
    ]
    return tuple(rows)


def test_clean_release_preserves_stable_slot_and_content_identity() -> None:
    release = build_clean_release_slots(_records())

    assert len(release) == len(_records())
    assert len({slot.slot_id for slot in release}) == len(release)
    assert all(slot.source_content_id == slot.model_content_id for slot in release)
    assert len(release_sha256(release)) == 64


def test_symmetric_swap_is_deterministic_balanced_and_slot_preserving() -> None:
    baseline = build_clean_release_slots(_records())
    candidate = build_symmetric_label_swap_candidate(
        baseline,
        label_a="intent_a",
        label_b="intent_b",
        per_direction=3,
    )
    repeated = build_symmetric_label_swap_candidate(
        baseline,
        label_a="intent_a",
        label_b="intent_b",
        per_direction=3,
    )

    assert candidate == repeated
    assert tuple(slot.slot_id for slot in candidate) == tuple(slot.slot_id for slot in baseline)
    assert tuple(slot.text for slot in candidate) == tuple(slot.text for slot in baseline)
    assert tuple(slot.source_content_id for slot in candidate) == tuple(
        slot.source_content_id for slot in baseline
    )

    manifest = release_diff_manifest(baseline, candidate)
    assert manifest["changed_slot_count"] == 6
    assert manifest["label_transitions"] == {
        "intent_a->intent_b": 3,
        "intent_b->intent_a": 3,
    }
    assert release_sha256(candidate) != release_sha256(baseline)


def test_symmetric_swap_doses_are_nested() -> None:
    baseline = build_clean_release_slots(_records())

    a_small, b_small = select_symmetric_swap_slot_ids(
        baseline,
        label_a="intent_a",
        label_b="intent_b",
        per_direction=2,
    )
    a_large, b_large = select_symmetric_swap_slot_ids(
        baseline,
        label_a="intent_a",
        label_b="intent_b",
        per_direction=5,
    )

    assert set(a_small) < set(a_large)
    assert set(b_small) < set(b_large)


def test_full_root_restoration_recovers_exact_baseline_release() -> None:
    baseline = build_clean_release_slots(_records())
    candidate = build_symmetric_label_swap_candidate(
        baseline,
        label_a="intent_a",
        label_b="intent_b",
        per_direction=4,
    )
    changed = changed_slot_ids(baseline, candidate)

    restored = restore_release_slots(
        candidate,
        baseline,
        restore_slot_ids=changed,
    )

    assert restored == baseline
    assert release_sha256(restored) == release_sha256(baseline)
    assert changed_slot_ids(baseline, restored) == ()


def test_release_manifest_does_not_expose_training_text() -> None:
    baseline = build_clean_release_slots(_records())
    candidate = build_symmetric_label_swap_candidate(
        baseline,
        label_a="intent_a",
        label_b="intent_b",
        per_direction=2,
    )

    serialized = json.dumps(release_diff_manifest(baseline, candidate), sort_keys=True)

    assert "PRIVATE-A" not in serialized
    assert "PRIVATE-B" not in serialized
    assert "PRIVATE-O" not in serialized


def test_symmetric_swap_rejects_invalid_labels_or_dose() -> None:
    baseline = build_clean_release_slots(_records())

    with pytest.raises(ValueError, match="labels must differ"):
        build_symmetric_label_swap_candidate(
            baseline,
            label_a="intent_a",
            label_b="intent_a",
            per_direction=1,
        )

    with pytest.raises(ValueError, match="per_direction must be positive"):
        build_symmetric_label_swap_candidate(
            baseline,
            label_a="intent_a",
            label_b="intent_b",
            per_direction=0,
        )

    with pytest.raises(ValueError, match="exceeds available"):
        build_symmetric_label_swap_candidate(
            baseline,
            label_a="intent_a",
            label_b="intent_b",
            per_direction=99,
        )
