from __future__ import annotations

from collections import Counter

from model_forensics.config import load_experiment_config
from model_forensics.exp008 import (
    EXP008_CHANGES_PER_SHARD,
    EXP008_NUISANCE_SOURCE_QUOTAS,
    EXP008_RECORDS_PER_SHARD,
    EXP008_SHARD_IDS,
    EXP008_WORLD_COUNT,
    build_exp008_data,
    build_exp008_plan,
    build_exp008_restoration_train,
    derive_exp008_world_seed,
    summarize_exp008_interventions,
)
from model_forensics.task import EXP003D_SLICE_IDS, TARGET_SLICE_ID


def test_exp008_config_freezes_model_and_behavioral_thresholds() -> None:
    config = load_experiment_config("configs/exp008.yaml")

    assert config.experiment_id == "exp008"
    assert config.seed == 42
    assert config.model.name == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert config.model.revision == "a10cc1512eabd3dde888204e902eca88bddb4951"

    assert config.training.epochs == 10
    assert config.training.batch_size == 8
    assert config.training.learning_rate == 0.0005
    assert config.training.lora_r == 16
    assert config.training.lora_alpha == 32
    assert config.training.lora_dropout == 0.0

    assert config.evaluation.minimum_baseline_score == 0.95
    assert config.evaluation.minimum_regression_delta == 0.15
    assert config.evaluation.minimum_recovery_delta == 0.15
    assert config.evaluation.maximum_unrelated_delta == 0.05
    assert config.lineage.artifact_kinds == ["dataset_shard"]


def test_exp008_worlds_are_deterministic_and_have_distinct_roots() -> None:
    plans = [
        build_exp008_plan(world_index=world_index) for world_index in range(EXP008_WORLD_COUNT)
    ]

    assert len({plan.planted_candidate_id for plan in plans}) == EXP008_WORLD_COUNT

    for world_index, plan in enumerate(plans):
        assert plan.world_index == world_index
        assert plan.world_seed == derive_exp008_world_seed(42, world_index)
        assert plan == build_exp008_plan(world_index=world_index)


def test_exp008_static_causal_contract_holds_in_both_worlds() -> None:
    expected_semantics = {slice_id: 48 for slice_id in EXP003D_SLICE_IDS}
    expected_nuisance = dict(sorted(EXP008_NUISANCE_SOURCE_QUOTAS.items()))

    for world_index in range(EXP008_WORLD_COUNT):
        plan = build_exp008_plan(world_index=world_index)
        data = build_exp008_data(world_index=world_index)
        summary = summarize_exp008_interventions(data)

        assert len(data.baseline_train) == 288
        assert len(data.candidate_train) == 288

        assert summary["baseline_semantics"] == expected_semantics
        assert summary["candidate_semantics"] == expected_semantics

        assert summary["baseline_responses"] == {
            "ACCEPT": 192,
            "REJECT": 96,
        }
        assert summary["candidate_responses"] == {
            "ACCEPT": 156,
            "REJECT": 132,
        }

        assert summary["global_prompt_multiset_preserved"]
        assert summary["baseline_unique_prompt_count"] == 288
        assert summary["candidate_unique_prompt_count"] == 288

        assert summary["aggregate_policy_inconsistent_by_slice"] == {
            TARGET_SLICE_ID: EXP008_CHANGES_PER_SHARD
        }

        baseline_prompt_counts = Counter(example.prompt for example in data.baseline_train)
        candidate_prompt_counts = Counter(example.prompt for example in data.candidate_train)
        assert baseline_prompt_counts == candidate_prompt_counts

        baseline_ids = [example.example_id for example in data.baseline_train]
        candidate_ids = [example.example_id for example in data.candidate_train]
        assert baseline_ids == candidate_ids
        assert len(set(baseline_ids)) == 288

        all_changed_ids: set[str] = set()

        for candidate_id in EXP008_SHARD_IDS:
            shard = [example for example in data.baseline_train if example.shard_id == candidate_id]
            assert len(shard) == EXP008_RECORDS_PER_SHARD

            candidate_summary = summary["candidates"][candidate_id]

            assert candidate_summary["changed_records"] == 36
            assert candidate_summary["prompt_changes"] == 36
            assert candidate_summary["response_changes"] == 36
            assert candidate_summary["prompt_multiset_preserved"]

            pairs = [
                (baseline, candidate)
                for baseline, candidate in zip(
                    data.baseline_train,
                    data.candidate_train,
                    strict=True,
                )
                if baseline.shard_id == candidate_id
                and baseline.to_sft_record() != candidate.to_sft_record()
            ]

            pair_ids = {baseline.example_id for baseline, _ in pairs}
            assert not all_changed_ids.intersection(pair_ids)
            all_changed_ids.update(pair_ids)

            assert Counter(baseline.prompt for baseline, _ in pairs) == Counter(
                candidate.prompt for _, candidate in pairs
            )

            if candidate_id == plan.planted_candidate_id:
                assert candidate_summary["source_semantics"] == {TARGET_SLICE_ID: 36}
                assert candidate_summary["result_semantics"] == {TARGET_SLICE_ID: 36}
                assert candidate_summary["source_responses"] == {"ACCEPT": 36}
                assert candidate_summary["result_responses"] == {"REJECT": 36}
                assert candidate_summary["policy_inconsistent_by_slice"] == {TARGET_SLICE_ID: 36}
                assert not candidate_summary["prompt_response_multiset_preserved"]

                root_materials = Counter(baseline.material for baseline, _ in pairs)
                root_colors = Counter(baseline.color for baseline, _ in pairs)
                root_slots = Counter(baseline.selected_slot for baseline, _ in pairs)

                assert set(root_materials.values()) == {3}
                assert set(root_colors.values()) == {9}
                assert set(root_slots.values()) == {6}
            else:
                assert candidate_summary["source_semantics"] == expected_nuisance
                assert candidate_summary["result_semantics"] == expected_nuisance
                assert candidate_summary["source_responses"] == {
                    "ACCEPT": 18,
                    "REJECT": 18,
                }
                assert candidate_summary["result_responses"] == {
                    "ACCEPT": 18,
                    "REJECT": 18,
                }
                assert candidate_summary["policy_inconsistent_by_slice"] == {}
                assert candidate_summary["prompt_response_multiset_preserved"]

                nuisance_slots = Counter(baseline.selected_slot for baseline, _ in pairs)
                assert set(nuisance_slots) == {
                    "slot_a",
                    "slot_b",
                    "slot_c",
                    "slot_d",
                    "slot_e",
                    "slot_f",
                }
                assert min(nuisance_slots.values()) >= 4
                assert max(nuisance_slots.values()) <= 8

                assert Counter(
                    (baseline.prompt, baseline.response) for baseline, _ in pairs
                ) == Counter((candidate.prompt, candidate.response) for _, candidate in pairs)

                assert all(
                    baseline.selected_slice_id != TARGET_SLICE_ID
                    and candidate.selected_slice_id != TARGET_SLICE_ID
                    for baseline, candidate in pairs
                )

        assert len(all_changed_ids) == 5 * 36


def test_exp008_nuisance_changes_are_true_content_permutations() -> None:
    for world_index in range(EXP008_WORLD_COUNT):
        plan = build_exp008_plan(world_index=world_index)
        data = build_exp008_data(world_index=world_index)

        for candidate_id in EXP008_SHARD_IDS:
            if candidate_id == plan.planted_candidate_id:
                continue

            before = [
                (example.prompt, example.response)
                for example in data.baseline_train
                if example.shard_id == candidate_id
            ]
            after = [
                (example.prompt, example.response)
                for baseline, example in zip(
                    data.baseline_train,
                    data.candidate_train,
                    strict=True,
                )
                if baseline.shard_id == candidate_id
            ]

            assert Counter(before) == Counter(after)
            assert before != after


def test_exp008_restoration_reverses_complete_selected_change() -> None:
    for world_index in range(EXP008_WORLD_COUNT):
        data = build_exp008_data(world_index=world_index)

        for restoration_candidate_id in EXP008_SHARD_IDS:
            restoration = build_exp008_restoration_train(
                restoration_candidate_id,
                world_index=world_index,
            )

            assert len(restoration) == len(data.candidate_train)

            restored_count = 0

            for baseline, candidate, restored in zip(
                data.baseline_train,
                data.candidate_train,
                restoration,
                strict=True,
            ):
                belongs = baseline.shard_id == restoration_candidate_id
                changed = baseline.to_sft_record() != candidate.to_sft_record()

                if belongs and changed:
                    restored_count += 1
                    assert restored == baseline
                else:
                    assert restored == candidate

            assert restored_count == EXP008_CHANGES_PER_SHARD
