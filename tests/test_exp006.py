from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

import model_forensics.task as task
from model_forensics.task import (
    EXP006_FROZEN_SEED,
    EXP006_LABEL_CHANGES_PER_SHARD,
    EXP006_MAX_WORLD_ATTEMPTS,
    EXP006_RECORDS_PER_SHARD,
    EXP006_SHARD_IDS,
    EXP006_SLOT_IDS,
    TARGET_SLICE_ID,
    build_exp006_data,
    build_exp006_plan,
    build_exp006_restoration_train,
    derive_exp006_world_seed,
)

EXPECTED_WORLD_SEEDS = (
    4225581908838437585,
    17420296024992258847,
    9978131756333736937,
    9112758517252332254,
    14221577644723289641,
)
EXPECTED_ROOTS = (
    "shard_semantic_01",
    "shard_semantic_03",
    "shard_semantic_05",
    "shard_semantic_05",
    "shard_semantic_05",
)
EXPECTED_MANIFEST_SHA256 = "275743ec6bd5ce130fd149da0b621b6a9d59c578d56518c5aaca3ed897011c27"


def _changed_examples(attempt_index: int):
    data = build_exp006_data(seed=EXP006_FROZEN_SEED, attempt_index=attempt_index)
    baseline_by_id = {example.example_id: example for example in data.baseline_train}
    candidate_by_id = {example.example_id: example for example in data.candidate_train}
    changed = [
        baseline_by_id[example_id]
        for example_id in baseline_by_id
        if baseline_by_id[example_id].response != candidate_by_id[example_id].response
    ]
    return data, baseline_by_id, candidate_by_id, changed


def test_exp006_frozen_manifest_hash_is_exact() -> None:
    manifest = Path(task.__file__).with_name("data") / "exp006_frozen_worlds.json"

    assert manifest.is_file()
    assert task.EXP006_FROZEN_MANIFEST_SHA256 == EXPECTED_MANIFEST_SHA256
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256


def test_exp006_world_seed_derivation_is_frozen_and_distinct() -> None:
    seeds = tuple(
        derive_exp006_world_seed(EXP006_FROZEN_SEED, index)
        for index in range(EXP006_MAX_WORLD_ATTEMPTS)
    )

    assert seeds == EXPECTED_WORLD_SEEDS
    assert len(set(seeds)) == EXP006_MAX_WORLD_ATTEMPTS


def test_exp006_plan_is_deterministic_and_has_frozen_roots() -> None:
    plans = tuple(
        build_exp006_plan(seed=EXP006_FROZEN_SEED, attempt_index=index)
        for index in range(EXP006_MAX_WORLD_ATTEMPTS)
    )

    assert tuple(plan.world_seed for plan in plans) == EXPECTED_WORLD_SEEDS
    assert tuple(plan.planted_candidate_id for plan in plans) == EXPECTED_ROOTS
    assert plans == tuple(
        build_exp006_plan(seed=EXP006_FROZEN_SEED, attempt_index=index)
        for index in range(EXP006_MAX_WORLD_ATTEMPTS)
    )


@pytest.mark.parametrize("attempt_index", range(EXP006_MAX_WORLD_ATTEMPTS))
def test_exp006_candidate_world_invariants(attempt_index: int) -> None:
    data, baseline_by_id, candidate_by_id, changed = _changed_examples(attempt_index)
    plan = build_exp006_plan(seed=EXP006_FROZEN_SEED, attempt_index=attempt_index)

    assert len(data.baseline_train) == 288
    assert len(data.candidate_train) == 288
    assert len(changed) == len(EXP006_SHARD_IDS) * EXP006_LABEL_CHANGES_PER_SHARD
    assert len({example.example_id for example in changed}) == len(changed)

    baseline_labels = Counter(example.response for example in data.baseline_train)
    candidate_labels = Counter(example.response for example in data.candidate_train)
    assert baseline_labels == {"ACCEPT": 192, "REJECT": 96}
    assert candidate_labels == {"ACCEPT": 176, "REJECT": 112}

    a_to_r = sum(
        example.response == "ACCEPT" and candidate_by_id[example.example_id].response == "REJECT"
        for example in changed
    )
    r_to_a = sum(
        example.response == "REJECT" and candidate_by_id[example.example_id].response == "ACCEPT"
        for example in changed
    )
    assert (a_to_r, r_to_a) == (38, 22)

    shard_sizes = Counter(example.shard_id for example in data.baseline_train)
    assert {shard_id: shard_sizes[shard_id] for shard_id in EXP006_SHARD_IDS} == {
        shard_id: EXP006_RECORDS_PER_SHARD for shard_id in EXP006_SHARD_IDS
    }

    changed_by_shard = {
        shard_id: [example for example in changed if example.shard_id == shard_id]
        for shard_id in EXP006_SHARD_IDS
    }
    assert all(
        len(examples) == EXP006_LABEL_CHANGES_PER_SHARD for examples in changed_by_shard.values()
    )

    for shard_id, examples in changed_by_shard.items():
        slot_counts = Counter(example.selected_slot for example in examples)
        assert slot_counts == Counter({slot: 2 for slot in EXP006_SLOT_IDS})

        color_counts = Counter(example.color for example in examples)
        color_vector = tuple(color_counts[color] for color in ("amber", "blue", "green", "violet"))
        assert max(color_vector) - min(color_vector) <= 2
        assert max(color_vector) <= 4

        material_counts = Counter(example.material for example in examples)
        assert max(material_counts.values()) <= 2

        target_count = sum(example.selected_slice_id == TARGET_SLICE_ID for example in examples)

        if shard_id == plan.planted_candidate_id:
            assert target_count == EXP006_LABEL_CHANGES_PER_SHARD
            assert all(example.response == "ACCEPT" for example in examples)
            assert color_vector == (3, 3, 3, 3)
            assert max(material_counts.values()) == 1
            assert len(material_counts) == 12
        else:
            assert target_count == 0

    protected = Counter(
        example.selected_slice_id
        for example in changed
        if example.selected_slice_id != TARGET_SLICE_ID
    )
    assert protected["square_small"] == 11
    assert protected["square_large"] == 11
    assert sorted(
        protected[slice_id] for slice_id in ("circle_small", "circle_large", "triangle_small")
    ) == [8, 9, 9]

    target_changes = [
        example for example in changed if example.selected_slice_id == TARGET_SLICE_ID
    ]
    assert len(target_changes) == 12
    assert all(example.shard_id == plan.planted_candidate_id for example in target_changes)

    assert set(baseline_by_id) == set(candidate_by_id)


@pytest.mark.parametrize("attempt_index", range(EXP006_MAX_WORLD_ATTEMPTS))
def test_exp006_restoration_restores_exactly_one_candidate(attempt_index: int) -> None:
    data = build_exp006_data(seed=EXP006_FROZEN_SEED, attempt_index=attempt_index)

    for shard_id in EXP006_SHARD_IDS:
        restored = build_exp006_restoration_train(
            shard_id,
            seed=EXP006_FROZEN_SEED,
            attempt_index=attempt_index,
        )

        restored_ids = [
            baseline.example_id
            for baseline, candidate, after in zip(
                data.baseline_train,
                data.candidate_train,
                restored,
                strict=True,
            )
            if candidate.response != after.response
        ]
        remaining_changes = [
            baseline.example_id
            for baseline, after in zip(
                data.baseline_train,
                restored,
                strict=True,
            )
            if baseline.response != after.response
        ]

        assert len(restored_ids) == EXP006_LABEL_CHANGES_PER_SHARD
        assert all(
            next(
                example for example in data.baseline_train if example.example_id == example_id
            ).shard_id
            == shard_id
            for example_id in restored_ids
        )
        assert len(remaining_changes) == (
            (len(EXP006_SHARD_IDS) - 1) * EXP006_LABEL_CHANGES_PER_SHARD
        )
        assert all(
            next(
                example for example in data.baseline_train if example.example_id == example_id
            ).shard_id
            != shard_id
            for example_id in remaining_changes
        )


def test_exp006_public_records_remain_opaque_and_policy_explicit() -> None:
    data = build_exp006_data(seed=EXP006_FROZEN_SEED, attempt_index=0)
    examples = data.baseline_train + data.candidate_train + data.all_eval

    assert all(example.prompt.startswith(task.EXP003D_POLICY_TEXT) for example in examples)

    records = [example.to_sft_record() for example in examples]
    assert all(set(record) == {"example_id", "prompt", "response"} for record in records)
    assert all(record["example_id"].startswith("rec_") for record in records)
    assert all(":" not in record["example_id"] for record in records)
    assert all("triangle_large" not in record["example_id"] for record in records)
    assert all("shard_semantic" not in record["example_id"] for record in records)


def test_exp006_rejects_unfrozen_seed_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="frozen only"):
        build_exp006_plan(seed=EXP006_FROZEN_SEED + 1, attempt_index=0)

    with pytest.raises(ValueError, match="attempt index"):
        build_exp006_plan(seed=EXP006_FROZEN_SEED, attempt_index=-1)

    with pytest.raises(ValueError, match="attempt index"):
        build_exp006_plan(
            seed=EXP006_FROZEN_SEED,
            attempt_index=EXP006_MAX_WORLD_ATTEMPTS,
        )

    with pytest.raises(ValueError, match="Unknown Experiment 006 candidate"):
        build_exp006_restoration_train(
            "not_a_candidate",
            seed=EXP006_FROZEN_SEED,
            attempt_index=0,
        )
