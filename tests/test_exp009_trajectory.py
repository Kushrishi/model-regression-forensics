from __future__ import annotations

import json

import pytest

from model_forensics.exp009_data import Banking77Record, content_id_for_record
from model_forensics.exp009_trajectory import (
    TRAJECTORY_NAMESPACE,
    build_stable_training_slots,
    derive_trajectory_seeds,
    epoch_slot_order,
    slot_schedule_sha256,
    trajectory_manifest,
)


def _record(label: str, text: str) -> Banking77Record:
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def test_stable_slots_are_input_order_independent() -> None:
    records = tuple(_record("intent", f"example-{index}") for index in range(8))

    forward = build_stable_training_slots(records)
    reverse = build_stable_training_slots(tuple(reversed(records)))

    assert forward == reverse
    assert [slot.slot_id for slot in forward] == [f"slot_{index:06d}" for index in range(1, 9)]
    assert [slot.content_id for slot in forward] == sorted(record.content_id for record in records)


def test_stable_slots_refuse_duplicate_content_ids() -> None:
    duplicate = _record("intent", "same")

    with pytest.raises(ValueError, match="unique content IDs"):
        build_stable_training_slots((duplicate, duplicate))


def test_trajectory_seed_family_is_reproducible_and_trajectory_specific() -> None:
    first = derive_trajectory_seeds(0)
    repeated = derive_trajectory_seeds(0)
    second = derive_trajectory_seeds(1)

    assert first == repeated
    assert first != second
    assert first.trajectory_id == 0
    assert (
        len(
            {
                first.python_seed,
                first.torch_seed,
                first.classifier_head_seed,
                first.dropout_seed,
                first.data_order_seed,
            }
        )
        == 5
    )
    assert all(
        0 <= seed < 2**63
        for seed in (
            first.python_seed,
            first.torch_seed,
            first.classifier_head_seed,
            first.dropout_seed,
            first.data_order_seed,
        )
    )


def test_trajectory_seed_rejects_negative_id() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_trajectory_seeds(-1)


def test_epoch_slot_order_is_a_deterministic_input_order_independent_permutation() -> None:
    slot_ids = tuple(f"slot_{index:06d}" for index in range(1, 33))

    order = epoch_slot_order(slot_ids, trajectory_id=3, epoch_index=0)
    repeated = epoch_slot_order(tuple(reversed(slot_ids)), trajectory_id=3, epoch_index=0)
    next_epoch = epoch_slot_order(slot_ids, trajectory_id=3, epoch_index=1)
    other_trajectory = epoch_slot_order(slot_ids, trajectory_id=4, epoch_index=0)

    assert order == repeated
    assert set(order) == set(slot_ids)
    assert len(order) == len(slot_ids)
    assert order != next_epoch
    assert order != other_trajectory


def test_schedule_hash_changes_with_trajectory_or_epoch_count() -> None:
    slot_ids = tuple(f"slot_{index:06d}" for index in range(1, 17))

    trajectory_zero = slot_schedule_sha256(slot_ids, trajectory_id=0, epochs=3)
    repeated = slot_schedule_sha256(tuple(reversed(slot_ids)), trajectory_id=0, epochs=3)
    trajectory_one = slot_schedule_sha256(slot_ids, trajectory_id=1, epochs=3)
    extra_epoch = slot_schedule_sha256(slot_ids, trajectory_id=0, epochs=4)

    assert trajectory_zero == repeated
    assert trajectory_zero != trajectory_one
    assert trajectory_zero != extra_epoch
    assert len(trajectory_zero) == 64


def test_trajectory_manifest_is_text_free_and_auditable() -> None:
    records = tuple(_record("intent", f"PRIVATE-TEXT-{index}") for index in range(12))
    slots = build_stable_training_slots(records)

    manifest = trajectory_manifest(slots, trajectory_id=7, epochs=3)
    serialized = json.dumps(manifest, sort_keys=True)

    assert manifest["trajectory_namespace"] == TRAJECTORY_NAMESPACE
    assert manifest["trajectory_id"] == 7
    assert manifest["slot_count"] == 12
    assert manifest["epochs"] == 3
    assert len(str(manifest["slot_schedule_sha256"])) == 64
    assert "PRIVATE-TEXT" not in serialized
