from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from pathlib import Path

import pytest

import model_forensics.exp007 as exp007
from model_forensics.exp007 import (
    EXP007_CALIBRATION_MATERIALS,
    EXP007_CALIBRATION_WORLD_COUNT,
    EXP007_CERTIFICATION_MATERIALS,
    EXP007_CERTIFICATION_WORLD_COUNT,
    EXP007_CHANGES_PER_SHARD,
    EXP007_FROZEN_SEED,
    EXP007_MAX_PAIRWISE_CHANGED_OVERLAP,
    EXP007_RECORDS_PER_SHARD,
    EXP007_SHARD_IDS,
    EXP007_TARGET_DOSES,
    build_exp007_data,
    build_exp007_plan,
    build_exp007_restoration_train,
    derive_exp007_world_seed,
)
from model_forensics.task import EXP003_SLOT_IDS, TARGET_SLICE_ID

EXPECTED_MANIFEST_SHA256 = "a6c5be745c5f4f4a97db7bf882651591886d25a0f125d967a147b861b19bdc28"
COLORS = ("amber", "blue", "green", "violet")
TRAIN_MATERIALS = (
    "cedar",
    "copper",
    "granite",
    "linen",
    "rubber",
    "glass",
    "paper",
    "steel",
    "clay",
    "leather",
    "plaster",
    "silk",
)
ACCEPT_PROTECTED = ("circle_small", "circle_large", "triangle_small")


def _world_cases():
    cases = []

    for dose in EXP007_TARGET_DOSES:
        cases.extend(
            ("calibration", dose, index) for index in range(EXP007_CALIBRATION_WORLD_COUNT)
        )
        cases.extend(
            ("certification", dose, index) for index in range(EXP007_CERTIFICATION_WORLD_COUNT)
        )

    return tuple(cases)


def _changed(phase: str, dose: int, world_index: int):
    data = build_exp007_data(
        phase=phase,
        target_dose=dose,
        world_index=world_index,
    )
    baseline = {x.example_id: x for x in data.baseline_train}
    candidate = {x.example_id: x for x in data.candidate_train}
    changed = [
        baseline[example_id]
        for example_id in baseline
        if baseline[example_id].response != candidate[example_id].response
    ]
    return data, baseline, candidate, changed


def test_exp007_frozen_manifest_hash_is_exact() -> None:
    manifest = Path(exp007.__file__).with_name("data") / "exp007_frozen_worlds.json"

    assert manifest.is_file()
    assert exp007.EXP007_FROZEN_MANIFEST_SHA256 == EXPECTED_MANIFEST_SHA256
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256


def test_exp007_world_seed_namespaces_are_frozen_and_distinct() -> None:
    seeds = [
        derive_exp007_world_seed(
            EXP007_FROZEN_SEED,
            phase,
            dose,
            world_index,
        )
        for phase, dose, world_index in _world_cases()
    ]

    assert len(seeds) == 14
    assert len(set(seeds)) == 14


def test_exp007_changed_training_worlds_are_distinct_within_each_dose() -> None:
    for dose in EXP007_TARGET_DOSES:
        signatures = []

        for phase, case_dose, world_index in _world_cases():
            if case_dose != dose:
                continue

            _, baseline, candidate, _ = _changed(
                phase,
                dose,
                world_index,
            )
            changed_ids = sorted(
                example_id
                for example_id in baseline
                if baseline[example_id].response != candidate[example_id].response
            )
            signatures.append(hashlib.sha256(",".join(changed_ids).encode()).hexdigest())

        assert len(signatures) == 7
        assert len(set(signatures)) == 7


@pytest.mark.parametrize(
    ("phase", "dose", "world_index"),
    _world_cases(),
)
def test_exp007_frozen_world_invariants(
    phase: str,
    dose: int,
    world_index: int,
) -> None:
    data, baseline, candidate, changed = _changed(
        phase,
        dose,
        world_index,
    )
    plan = build_exp007_plan(
        phase=phase,
        target_dose=dose,
        world_index=world_index,
    )

    assert len(data.baseline_train) == 288
    assert len(data.candidate_train) == 288
    assert len(changed) == 180
    assert len({x.example_id for x in changed}) == 180

    candidate_labels = Counter(x.response for x in data.candidate_train)
    assert candidate_labels == {"REJECT": 156, "ACCEPT": 132}

    shard_sizes = Counter(x.shard_id for x in data.baseline_train)
    assert all(shard_sizes[shard_id] == EXP007_RECORDS_PER_SHARD for shard_id in EXP007_SHARD_IDS)

    material_histograms = []

    for shard_id in EXP007_SHARD_IDS:
        shard_changed = [x for x in changed if x.shard_id == shard_id]
        assert len(shard_changed) == EXP007_CHANGES_PER_SHARD

        assert Counter(x.response for x in shard_changed) == {
            "ACCEPT": 24,
            "REJECT": 12,
        }

        semantics = Counter(x.selected_slice_id for x in shard_changed)

        if shard_id == plan.planted_candidate_id:
            protected_each = (24 - dose) // 3
            assert semantics[TARGET_SLICE_ID] == dose
            assert all(semantics[slice_id] == protected_each for slice_id in ACCEPT_PROTECTED)
        else:
            assert semantics[TARGET_SLICE_ID] == 0
            assert all(semantics[slice_id] == 8 for slice_id in ACCEPT_PROTECTED)

        assert semantics["square_small"] == 6
        assert semantics["square_large"] == 6

        for slot in EXP003_SLOT_IDS:
            assert (
                sum(x.response == "ACCEPT" and x.selected_slot == slot for x in shard_changed) == 4
            )
            assert (
                sum(x.response == "REJECT" and x.selected_slot == slot for x in shard_changed) == 2
            )

        for color in COLORS:
            assert sum(x.response == "ACCEPT" and x.color == color for x in shard_changed) == 6
            assert sum(x.response == "REJECT" and x.color == color for x in shard_changed) == 3

        histogram = tuple(
            Counter(x.material for x in shard_changed)[material] for material in TRAIN_MATERIALS
        )
        assert min(histogram) >= 2
        assert max(histogram) <= 4
        material_histograms.append(histogram)

    assert len(set(material_histograms)) == 1
    assert set(baseline) == set(candidate)

    assert len(data.all_eval) == 96
    assert all(len(data.eval_by_slice[slice_id]) == 16 for slice_id in data.eval_by_slice)

    eval_materials = {x.material for x in data.all_eval}
    if phase == "calibration":
        assert eval_materials == set(EXP007_CALIBRATION_MATERIALS)
    else:
        assert eval_materials == set(EXP007_CERTIFICATION_MATERIALS)


def test_exp007_calibration_and_certification_ids_are_disjoint() -> None:
    calibration = build_exp007_data(
        phase="calibration",
        target_dose=9,
        world_index=0,
    )
    certification = build_exp007_data(
        phase="certification",
        target_dose=9,
        world_index=0,
    )

    train_ids = {x.example_id for x in calibration.baseline_train}
    calibration_ids = {x.example_id for x in calibration.all_eval}
    certification_ids = {x.example_id for x in certification.all_eval}

    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(certification_ids)
    assert calibration_ids.isdisjoint(certification_ids)


def test_exp007_frozen_worlds_pass_pairwise_diversity_gate() -> None:
    for dose in EXP007_TARGET_DOSES:
        changed_sets = []

        for phase, case_dose, world_index in _world_cases():
            if case_dose != dose:
                continue

            _, baseline, candidate, _ = _changed(
                phase,
                dose,
                world_index,
            )
            changed_sets.append(
                {
                    example_id
                    for example_id in baseline
                    if baseline[example_id].response != candidate[example_id].response
                }
            )

        overlaps = [len(a & b) for a, b in itertools.combinations(changed_sets, 2)]

        assert len(changed_sets) == 7
        assert max(overlaps) <= EXP007_MAX_PAIRWISE_CHANGED_OVERLAP


@pytest.mark.parametrize("dose", EXP007_TARGET_DOSES)
def test_exp007_restoration_restores_exactly_one_candidate(dose: int) -> None:
    data = build_exp007_data(
        phase="certification",
        target_dose=dose,
        world_index=0,
    )

    for shard_id in EXP007_SHARD_IDS:
        restored = build_exp007_restoration_train(
            shard_id,
            phase="certification",
            target_dose=dose,
            world_index=0,
        )

        restored_count = sum(
            candidate.response != after.response
            for candidate, after in zip(
                data.candidate_train,
                restored,
                strict=True,
            )
        )
        remaining_count = sum(
            baseline.response != after.response
            for baseline, after in zip(
                data.baseline_train,
                restored,
                strict=True,
            )
        )

        assert restored_count == EXP007_CHANGES_PER_SHARD
        assert remaining_count == 4 * EXP007_CHANGES_PER_SHARD


def test_exp007_public_records_remain_opaque() -> None:
    data = build_exp007_data(
        phase="calibration",
        target_dose=9,
        world_index=0,
    )

    records = [
        x.to_sft_record() for x in data.baseline_train + data.candidate_train + data.all_eval
    ]

    assert all(set(record) == {"example_id", "prompt", "response"} for record in records)
    assert all(record["example_id"].startswith("rec_") for record in records)
    assert all("shard_sensitivity" not in record["example_id"] for record in records)


def test_exp007_rejects_unfrozen_inputs() -> None:
    with pytest.raises(ValueError, match="frozen only"):
        build_exp007_plan(
            seed=43,
            phase="calibration",
            target_dose=9,
            world_index=0,
        )

    with pytest.raises(ValueError, match="target dose"):
        build_exp007_plan(
            phase="calibration",
            target_dose=12,
            world_index=0,
        )

    with pytest.raises(ValueError, match="world index"):
        build_exp007_plan(
            phase="certification",
            target_dose=9,
            world_index=5,
        )

    with pytest.raises(ValueError, match="Unknown Experiment 007 candidate"):
        build_exp007_restoration_train(
            "not_a_candidate",
            phase="certification",
            target_dose=9,
            world_index=0,
        )
