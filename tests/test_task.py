from dataclasses import replace

from model_forensics.task import (
    EXP001_CHANGED_SHARD_IDS,
    EXP001_CONTROL_SLICE_ID,
    EXP001_SHARD_BY_SLICE,
    EXP002_CONTROL_SLICE_ID,
    EXP002_LABEL_CHANGES_PER_SHARD,
    EXP002_RECORDS_PER_SHARD,
    EXP002_SHARD_IDS,
    EXP003_CONTROL_SLICE_ID,
    EXP003_LABEL_CHANGES_PER_SHARD,
    EXP003_RECORDS_PER_SHARD,
    EXP003_SHARD_IDS,
    EXP003_SLOT_IDS,
    EXP003C_EVAL_CONTEXTS,
    EXP003C_SLOT_IDS,
    EXP003C_TRAIN_CONTEXTS,
    EXP003D_POLICY,
    EXP003D_POLICY_TEXT,
    EXP003D_SLICE_IDS,
    EXP004_CANDIDATE_SLICES,
    EXP004_CONTROL_SLICE_ID,
    EXP004_LABEL_CHANGES_PER_SHARD,
    EXP004_RECORDS_PER_SHARD,
    EXP004_SHARD_IDS,
    EXP004_SLOT_IDS,
    EXP005_LABEL_CHANGES_PER_SHARD,
    EXP005_MAX_WORLD_ATTEMPTS,
    EXP005_RECORDS_PER_SHARD,
    EXP005_SHARD_IDS,
    EXP005_SLOT_IDS,
    REGRESSION_SHARD_ID,
    TARGET_SLICE_ID,
    build_exp000_data,
    build_exp001_data,
    build_exp002_data,
    build_exp002_plan,
    build_exp003_data,
    build_exp003_plan,
    build_exp003c_lookup_data,
    build_exp003d_explicit_policy_data,
    build_exp004_data,
    build_exp004_intervention_train,
    build_exp004_plan,
    build_exp005_data,
    build_exp005_plan,
    build_exp005_restoration_train,
    derive_exp005_world_seed,
)


def test_candidate_changes_only_regression_shard_labels() -> None:
    data = build_exp000_data(seed=42)

    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    assert baseline.keys() == candidate.keys()
    changed = [
        (baseline[example_id], candidate[example_id])
        for example_id in baseline
        if baseline[example_id].response != candidate[example_id].response
    ]

    assert len(data.baseline_train) == 288
    assert len(changed) == 48
    assert all(before.prompt == after.prompt for before, after in changed)
    assert all(before.shard_id == REGRESSION_SHARD_ID for before, _ in changed)
    assert all(before.slice_id == TARGET_SLICE_ID for before, _ in changed)
    assert data.recovery_train == data.baseline_train


def test_eval_materials_are_held_out_and_target_slice_isolated() -> None:
    data = build_exp000_data(seed=42)

    train_materials = {example.material for example in data.baseline_train}
    eval_examples = data.target_eval + data.unrelated_eval
    eval_materials = {example.material for example in eval_examples}

    assert train_materials.isdisjoint(eval_materials)
    assert len(data.target_eval) == 16
    assert len(data.unrelated_eval) == 80
    assert all(example.slice_id == TARGET_SLICE_ID for example in data.target_eval)
    assert all(example.slice_id != TARGET_SLICE_ID for example in data.unrelated_eval)


def test_data_generation_is_deterministic_for_seed() -> None:
    first = build_exp000_data(seed=42)
    second = build_exp000_data(seed=42)

    assert first == second


def test_exp001_candidate_has_five_equal_opaque_changed_shards() -> None:
    data = build_exp001_data(seed=42)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    changed = [
        (baseline[example_id], candidate[example_id])
        for example_id in baseline
        if baseline[example_id].response != candidate[example_id].response
    ]

    assert len(changed) == 240
    assert {before.shard_id for before, _ in changed} == EXP001_CHANGED_SHARD_IDS
    assert all(before.prompt == after.prompt for before, after in changed)
    assert all(
        sum(before.shard_id == shard_id for before, _ in changed) == 48
        for shard_id in EXP001_CHANGED_SHARD_IDS
    )
    assert all("triangle" not in shard_id for shard_id in EXP001_CHANGED_SHARD_IDS)
    assert all("large" not in shard_id for shard_id in EXP001_CHANGED_SHARD_IDS)


def test_exp001_intervention_restores_only_target_causal_shard() -> None:
    data = build_exp001_data(seed=42)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}
    intervention = {example.example_id: example for example in data.intervention_train}
    target_shard = EXP001_SHARD_BY_SLICE[TARGET_SLICE_ID]

    restored = [
        example_id
        for example_id in baseline
        if candidate[example_id].response != intervention[example_id].response
    ]
    remaining_changes = [
        example_id
        for example_id in baseline
        if baseline[example_id].response != intervention[example_id].response
    ]

    assert len(restored) == 48
    assert all(baseline[example_id].shard_id == target_shard for example_id in restored)
    assert len(remaining_changes) == 192
    assert all(baseline[example_id].shard_id != target_shard for example_id in remaining_changes)


def test_exp001_eval_uses_held_out_target_and_control_slices() -> None:
    data = build_exp001_data(seed=42)
    train_materials = {example.material for example in data.baseline_train}
    eval_materials = {example.material for example in data.all_eval}

    assert train_materials.isdisjoint(eval_materials)
    assert len(data.target_eval) == 16
    assert len(data.control_eval) == 16
    assert len(data.all_eval) == 96
    assert all(example.slice_id == TARGET_SLICE_ID for example in data.target_eval)
    assert all(example.slice_id == EXP001_CONTROL_SLICE_ID for example in data.control_eval)


def test_exp002_entangles_target_content_across_equal_candidate_shards() -> None:
    data = build_exp002_data(seed=42)
    plan = build_exp002_plan(seed=42)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    shard_records = {
        shard_id: [example for example in data.baseline_train if example.shard_id == shard_id]
        for shard_id in EXP002_SHARD_IDS
    }
    target_counts = sorted(
        sum(example.slice_id == TARGET_SLICE_ID for example in examples)
        for examples in shard_records.values()
    )
    changed_by_shard = {
        shard_id: sum(
            baseline[example_id].shard_id == shard_id
            and baseline[example_id].response != candidate[example_id].response
            for example_id in baseline
        )
        for shard_id in EXP002_SHARD_IDS
    }

    assert len(data.baseline_train) == 288
    assert set(plan.changed_slice_by_shard) == set(EXP002_SHARD_IDS)
    assert all(len(examples) == EXP002_RECORDS_PER_SHARD for examples in shard_records.values())
    assert target_counts == [4, 4, 4, 4, 32]
    assert all(
        {example.color for example in examples if example.slice_id == TARGET_SLICE_ID}
        == {"amber", "blue", "green", "violet"}
        for examples in shard_records.values()
    )
    assert set(changed_by_shard.values()) == {EXP002_LABEL_CHANGES_PER_SHARD}


def test_exp002_intervention_restores_only_generated_target_cause() -> None:
    data = build_exp002_data(seed=42)
    plan = build_exp002_plan(seed=42)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}
    intervention = {example.example_id: example for example in data.intervention_train}

    restored = [
        example_id
        for example_id in baseline
        if candidate[example_id].response != intervention[example_id].response
    ]
    remaining_changes = [
        example_id
        for example_id in baseline
        if baseline[example_id].response != intervention[example_id].response
    ]

    assert len(restored) == 32
    assert all(baseline[example_id].shard_id == plan.root_cause_id for example_id in restored)
    assert all(baseline[example_id].slice_id == TARGET_SLICE_ID for example_id in restored)
    assert len(remaining_changes) == 128
    assert all(
        baseline[example_id].shard_id != plan.root_cause_id for example_id in remaining_changes
    )


def test_exp002_eval_uses_held_out_target_and_control_slices() -> None:
    data = build_exp002_data(seed=42)
    train_materials = {example.material for example in data.baseline_train}
    eval_materials = {example.material for example in data.all_eval}

    assert train_materials.isdisjoint(eval_materials)
    assert len(data.target_eval) == 16
    assert len(data.control_eval) == 16
    assert len(data.all_eval) == 96
    assert all(example.slice_id == TARGET_SLICE_ID for example in data.target_eval)
    assert all(example.slice_id == EXP002_CONTROL_SLICE_ID for example in data.control_eval)


def test_exp003_role_binding_candidates_are_equal_and_slot_balanced() -> None:
    data = build_exp003_data(seed=42)
    plan = build_exp003_plan(seed=42)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    assert len(data.baseline_train) == 288
    assert set(plan.selected_slice_by_shard) == set(EXP003_SHARD_IDS)

    for shard_id in EXP003_SHARD_IDS:
        shard = [example for example in data.baseline_train if example.shard_id == shard_id]
        changed = [
            example
            for example in shard
            if baseline[example.example_id].response != candidate[example.example_id].response
        ]
        assert len(shard) == EXP003_RECORDS_PER_SHARD
        assert len(changed) == EXP003_LABEL_CHANGES_PER_SHARD
        assert {example.selected_slot for example in changed} == set(EXP003_SLOT_IDS)
        assert all(
            sum(example.selected_slot == slot for example in changed) == 6
            for slot in EXP003_SLOT_IDS
        )
        assert all("shape=triangle,size=large" in example.prompt for example in shard)
        assert all("shape=triangle,size=large" in example.prompt for example in changed)


def test_exp003_intervention_restores_only_generated_target_cause() -> None:
    data = build_exp003_data(seed=42)
    plan = build_exp003_plan(seed=42)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}
    intervention = {example.example_id: example for example in data.intervention_train}

    restored = [
        example_id
        for example_id in baseline
        if candidate[example_id].response != intervention[example_id].response
    ]
    remaining_changes = [
        example_id
        for example_id in baseline
        if baseline[example_id].response != intervention[example_id].response
    ]

    assert len(restored) == 36
    assert all(baseline[example_id].shard_id == plan.root_cause_id for example_id in restored)
    assert all(baseline[example_id].selected_slice_id == TARGET_SLICE_ID for example_id in restored)
    assert len(remaining_changes) == 144
    assert all(
        baseline[example_id].shard_id != plan.root_cause_id for example_id in remaining_changes
    )


def test_exp003_eval_uses_held_out_target_and_control_roles() -> None:
    data = build_exp003_data(seed=42)
    train_materials = {example.material for example in data.baseline_train}
    eval_materials = {example.material for example in data.all_eval}

    assert train_materials.isdisjoint(eval_materials)
    assert len(data.target_eval) == 16
    assert len(data.control_eval) == 16
    assert len(data.all_eval) == 96
    assert all(example.selected_slice_id == TARGET_SLICE_ID for example in data.target_eval)
    assert all(
        example.selected_slice_id == EXP003_CONTROL_SLICE_ID for example in data.control_eval
    )


def test_exp003_public_records_hide_benchmark_annotations() -> None:
    data = build_exp003_data(seed=42)
    records = [example.to_sft_record() for example in data.baseline_train]

    assert all(set(record) == {"example_id", "prompt", "response"} for record in records)
    assert all(record["example_id"].startswith("rec_") for record in records)
    assert all(":" not in record["example_id"] for record in records)
    assert all("triangle_large" not in record["example_id"] for record in records)


def test_exp003c_lookup_construction_is_balanced_and_held_out() -> None:
    data = build_exp003c_lookup_data(seed=42)

    assert len(data.baseline_train) == 288
    assert len(data.all_eval) == 96
    assert len(data.train_patterns) == 16
    assert len(data.eval_patterns) == 4
    assert set(data.train_patterns).isdisjoint(data.eval_patterns)
    assert len(set(data.train_patterns) | set(data.eval_patterns)) == 20

    train_labels = {
        label: sum(example.response == label for example in data.baseline_train)
        for label in ("ACCEPT", "REJECT")
    }
    eval_labels = {
        label: sum(example.response == label for example in data.all_eval)
        for label in ("ACCEPT", "REJECT")
    }
    assert train_labels == {"ACCEPT": 144, "REJECT": 144}
    assert eval_labels == {"ACCEPT": 48, "REJECT": 48}

    for example in data.baseline_train + data.all_eval:
        assert example.prompt.count("decision=ACCEPT") == 3
        assert example.prompt.count("decision=REJECT") == 3


def test_exp003c_lookup_is_balanced_within_every_selected_slot() -> None:
    data = build_exp003c_lookup_data(seed=42)

    for slot in EXP003C_SLOT_IDS:
        train = [example for example in data.baseline_train if example.selected_slot == slot]
        evaluation = [example for example in data.all_eval if example.selected_slot == slot]

        assert len(train) == 48
        assert len(evaluation) == 16
        assert sum(example.response == "ACCEPT" for example in train) == 24
        assert sum(example.response == "REJECT" for example in train) == 24
        assert sum(example.response == "ACCEPT" for example in evaluation) == 8
        assert sum(example.response == "REJECT" for example in evaluation) == 8
        assert len(data.eval_by_slot[slot]) == 16


def test_exp003c_contexts_and_public_records_do_not_leak_shortcuts() -> None:
    data = build_exp003c_lookup_data(seed=42)

    assert set(EXP003C_TRAIN_CONTEXTS).isdisjoint(EXP003C_EVAL_CONTEXTS)
    assert {example.context_id for example in data.baseline_train} == set(EXP003C_TRAIN_CONTEXTS)
    assert {example.context_id for example in data.all_eval} == set(EXP003C_EVAL_CONTEXTS)

    records = [example.to_sft_record() for example in data.baseline_train + data.all_eval]
    assert all(set(record) == {"example_id", "prompt", "response"} for record in records)
    assert all(record["example_id"].startswith("rec_") for record in records)
    assert all(":" not in record["example_id"] for record in records)


def test_exp003c_generation_is_deterministic_for_seed() -> None:
    assert build_exp003c_lookup_data(seed=42) == build_exp003c_lookup_data(seed=42)


def test_exp003d_changes_only_the_model_visible_policy_prefix() -> None:
    source = build_exp003_data(seed=42)
    data = build_exp003d_explicit_policy_data(seed=42)

    assert len(data.baseline_train) == len(source.baseline_train) == 288
    assert len(data.all_eval) == len(source.all_eval) == 96

    for original, explicit in zip(
        source.baseline_train + source.all_eval,
        data.baseline_train + data.all_eval,
        strict=True,
    ):
        assert explicit.prompt == f"{EXP003D_POLICY_TEXT} {original.prompt}"
        assert replace(explicit, prompt=original.prompt) == original


def test_exp003d_preserves_exp003_counts_labels_and_slice_evals() -> None:
    data = build_exp003d_explicit_policy_data(seed=42)

    assert sum(example.response == "ACCEPT" for example in data.baseline_train) == 192
    assert sum(example.response == "REJECT" for example in data.baseline_train) == 96
    assert sum(example.response == "ACCEPT" for example in data.all_eval) == 64
    assert sum(example.response == "REJECT" for example in data.all_eval) == 32

    for slice_id in EXP003D_SLICE_IDS:
        train = [
            example for example in data.baseline_train if example.selected_slice_id == slice_id
        ]
        evaluation = data.eval_by_slice[slice_id]
        assert len(train) == 48
        assert len(evaluation) == 16
        shape = slice_id.split("_", maxsplit=1)[0]
        assert {example.response for example in (*train, *evaluation)} == {EXP003D_POLICY[shape]}


def test_exp003d_public_records_remain_opaque_and_policy_is_explicit() -> None:
    data = build_exp003d_explicit_policy_data(seed=42)
    examples = data.baseline_train + data.all_eval

    assert all(example.prompt.startswith(EXP003D_POLICY_TEXT) for example in examples)
    assert all(
        set(example.to_sft_record()) == {"example_id", "prompt", "response"} for example in examples
    )
    assert all(example.example_id.startswith("rec_") for example in examples)
    assert all(":" not in example.example_id for example in examples)


def test_exp003d_generation_is_deterministic_for_seed() -> None:
    assert build_exp003d_explicit_policy_data(seed=42) == build_exp003d_explicit_policy_data(
        seed=42
    )


def test_exp004_plan_is_fresh_deterministic_and_targets_declared_slice() -> None:
    plan = build_exp004_plan(seed=42)

    assert plan == build_exp004_plan(seed=42)
    assert set(plan.selected_slice_by_shard) == set(EXP004_SHARD_IDS)
    assert set(plan.selected_slice_by_shard.values()) == set(EXP004_CANDIDATE_SLICES)
    assert set(EXP004_SHARD_IDS).isdisjoint(EXP003_SHARD_IDS)
    assert plan.root_cause_id in EXP004_SHARD_IDS
    assert plan.selected_slice_by_shard[plan.root_cause_id] == TARGET_SLICE_ID


def test_exp004_clean_model_facing_data_matches_exp003d_exactly() -> None:
    source = build_exp003d_explicit_policy_data(seed=42)
    data = build_exp004_data(seed=42)

    assert len(data.baseline_train) == len(source.baseline_train) == 288
    assert len(data.all_eval) == len(source.all_eval) == 96

    assert [example.to_sft_record() for example in data.baseline_train] == [
        example.to_sft_record() for example in source.baseline_train
    ]

    for original, exp004 in zip(
        source.baseline_train,
        data.baseline_train,
        strict=True,
    ):
        assert replace(exp004, shard_id=original.shard_id) == original

    assert data.all_eval == source.all_eval
    assert data.eval_by_slice == source.eval_by_slice

    assert len(data.target_eval) == 16
    assert len(data.control_eval) == 16
    assert all(example.selected_slice_id == TARGET_SLICE_ID for example in data.target_eval)
    assert all(
        example.selected_slice_id == EXP004_CONTROL_SLICE_ID for example in data.control_eval
    )


def test_exp004_candidates_are_equal_sized_changed_and_slot_balanced() -> None:
    data = build_exp004_data(seed=42)

    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    descriptors = tuple(
        f"shape={shape},size={size}"
        for shape in ("circle", "square", "triangle")
        for size in ("small", "large")
    )

    total_changed = 0

    for shard_id in EXP004_SHARD_IDS:
        shard = [example for example in data.baseline_train if example.shard_id == shard_id]

        changed = [
            example
            for example in shard
            if baseline[example.example_id].response != candidate[example.example_id].response
        ]

        assert len(shard) == EXP004_RECORDS_PER_SHARD
        assert len(changed) == EXP004_LABEL_CHANGES_PER_SHARD
        assert {example.selected_slot for example in changed} == set(EXP004_SLOT_IDS)

        assert all(
            sum(example.selected_slot == slot for example in changed) == 6
            for slot in EXP004_SLOT_IDS
        )

        # Every prompt contains every shape-size descriptor, so marginal
        # target-surface exposure is exactly equal across candidates.
        for descriptor in descriptors:
            assert sum(descriptor in example.prompt for example in shard) == len(shard)
            assert sum(descriptor in example.prompt for example in changed) == len(changed)

        total_changed += len(changed)

    stable = [example for example in data.baseline_train if example.shard_id == "shard_stable_00"]

    assert len(stable) == 48
    assert all(
        baseline[example.example_id].response == candidate[example.example_id].response
        for example in stable
    )

    assert total_changed == 5 * EXP004_LABEL_CHANGES_PER_SHARD == 180


def test_exp004_intervention_restores_only_selected_candidate() -> None:
    data = build_exp004_data(seed=42)
    plan = build_exp004_plan(seed=42)

    # Deliberately choose a non-root candidate. This verifies that the
    # intervention builder follows the supplied diagnosis rather than truth.
    selected_candidate = next(
        shard_id for shard_id in EXP004_SHARD_IDS if shard_id != plan.root_cause_id
    )

    intervention_train = build_exp004_intervention_train(
        selected_candidate,
        seed=42,
    )

    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}
    intervention = {example.example_id: example for example in intervention_train}

    restored = [
        example_id
        for example_id in baseline
        if candidate[example_id].response != intervention[example_id].response
    ]

    remaining_changes = [
        example_id
        for example_id in baseline
        if baseline[example_id].response != intervention[example_id].response
    ]

    assert len(restored) == EXP004_LABEL_CHANGES_PER_SHARD
    assert all(baseline[example_id].shard_id == selected_candidate for example_id in restored)

    assert len(remaining_changes) == (4 * EXP004_LABEL_CHANGES_PER_SHARD)
    assert all(
        baseline[example_id].shard_id != selected_candidate for example_id in remaining_changes
    )


def test_exp004_public_records_remain_opaque_and_policy_explicit() -> None:
    data = build_exp004_data(seed=42)

    examples = data.baseline_train + data.candidate_train + data.all_eval

    assert all(example.prompt.startswith(EXP003D_POLICY_TEXT) for example in examples)

    records = [example.to_sft_record() for example in examples]

    assert all(set(record) == {"example_id", "prompt", "response"} for record in records)
    assert all(record["example_id"].startswith("rec_") for record in records)
    assert all(":" not in record["example_id"] for record in records)
    assert all("triangle_large" not in record["example_id"] for record in records)


def test_exp005_world_seed_derivation_is_frozen_and_distinct() -> None:
    seeds = [derive_exp005_world_seed(42, index) for index in range(EXP005_MAX_WORLD_ATTEMPTS)]

    assert seeds == [derive_exp005_world_seed(42, index) for index in range(5)]
    assert len(set(seeds)) == EXP005_MAX_WORLD_ATTEMPTS


def test_exp005_plan_is_deterministic_private_and_targets_one_candidate() -> None:
    plan = build_exp005_plan(seed=42, attempt_index=0)

    assert plan == build_exp005_plan(seed=42, attempt_index=0)
    assert plan.planted_candidate_id in EXP005_SHARD_IDS
    assert build_exp005_plan(seed=42, attempt_index=1).world_seed != plan.world_seed


def test_exp005_candidate_shards_are_balanced_bidirectionally_and_by_slot() -> None:
    data = build_exp005_data(seed=42, attempt_index=0)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    assert len(data.baseline_train) == 288

    for shard_id in EXP005_SHARD_IDS:
        shard = [example for example in data.baseline_train if example.shard_id == shard_id]
        changed = [
            example
            for example in shard
            if baseline[example.example_id].response != candidate[example.example_id].response
        ]

        assert len(shard) == EXP005_RECORDS_PER_SHARD
        assert len(changed) == EXP005_LABEL_CHANGES_PER_SHARD
        assert sum(example.response == "ACCEPT" for example in changed) == 12
        assert sum(example.response == "REJECT" for example in changed) == 12
        assert {
            slot: sum(example.selected_slot == slot for example in changed)
            for slot in EXP005_SLOT_IDS
        } == {slot: 4 for slot in EXP005_SLOT_IDS}


def test_exp005_corruption_preserves_global_label_counts_exactly() -> None:
    data = build_exp005_data(seed=42, attempt_index=0)

    baseline_counts = {
        label: sum(example.response == label for example in data.baseline_train)
        for label in ("ACCEPT", "REJECT")
    }
    candidate_counts = {
        label: sum(example.response == label for example in data.candidate_train)
        for label in ("ACCEPT", "REJECT")
    }

    assert baseline_counts == {"ACCEPT": 192, "REJECT": 96}
    assert candidate_counts == baseline_counts


def test_exp005_only_planted_changed_shard_has_selected_target_records() -> None:
    data = build_exp005_data(seed=42, attempt_index=0)
    plan = build_exp005_plan(seed=42, attempt_index=0)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    target_changed_counts = {}
    for shard_id in EXP005_SHARD_IDS:
        target_changed_counts[shard_id] = sum(
            example.shard_id == shard_id
            and example.selected_slice_id == TARGET_SLICE_ID
            and baseline[example.example_id].response != candidate[example.example_id].response
            for example in data.baseline_train
        )

    assert target_changed_counts[plan.planted_candidate_id] == 12
    assert all(
        count == 0
        for shard_id, count in target_changed_counts.items()
        if shard_id != plan.planted_candidate_id
    )


def test_exp005_changed_records_match_target_color_slot_coverage_for_every_candidate() -> None:
    data = build_exp005_data(seed=42, attempt_index=0)
    baseline = {example.example_id: example for example in data.baseline_train}
    candidate = {example.example_id: example for example in data.candidate_train}

    target_pairs = {(example.color, example.selected_slot) for example in data.target_eval}
    assert len(target_pairs) == 12

    for shard_id in EXP005_SHARD_IDS:
        changed_pairs = {
            (example.color, example.selected_slot)
            for example in data.baseline_train
            if example.shard_id == shard_id
            and baseline[example.example_id].response != candidate[example.example_id].response
        }
        assert target_pairs <= changed_pairs


def test_exp005_restoration_restores_exactly_one_candidate() -> None:
    data = build_exp005_data(seed=42, attempt_index=0)

    for shard_id in EXP005_SHARD_IDS:
        restoration = build_exp005_restoration_train(
            shard_id,
            seed=42,
            attempt_index=0,
        )

        restored = [
            before.example_id
            for before, candidate, after in zip(
                data.baseline_train,
                data.candidate_train,
                restoration,
                strict=True,
            )
            if candidate.response != after.response
        ]
        remaining = [
            before.example_id
            for before, after in zip(
                data.baseline_train,
                restoration,
                strict=True,
            )
            if before.response != after.response
        ]

        assert len(restored) == EXP005_LABEL_CHANGES_PER_SHARD
        assert all(
            next(
                example for example in data.baseline_train if example.example_id == example_id
            ).shard_id
            == shard_id
            for example_id in restored
        )
        assert len(remaining) == ((len(EXP005_SHARD_IDS) - 1) * EXP005_LABEL_CHANGES_PER_SHARD)


def test_exp005_clean_model_facing_data_preserves_exp003d_parity_and_opacity() -> None:
    data = build_exp005_data(seed=42, attempt_index=0)
    source = build_exp003d_explicit_policy_data(seed=42)

    assert [example.to_sft_record() for example in data.baseline_train] == [
        example.to_sft_record() for example in source.baseline_train
    ]
    assert [example.to_sft_record() for example in data.all_eval] == [
        example.to_sft_record() for example in source.all_eval
    ]
    assert all(
        set(example.to_sft_record()) == {"example_id", "prompt", "response"}
        for example in data.baseline_train + data.all_eval
    )
    assert all(
        example.example_id.startswith("rec_") and ":" not in example.example_id
        for example in data.baseline_train + data.all_eval
    )


def test_exp005_generation_is_deterministic_and_world_attempts_change_corruption() -> None:
    first = build_exp005_data(seed=42, attempt_index=0)
    second = build_exp005_data(seed=42, attempt_index=0)
    alternate = build_exp005_data(seed=42, attempt_index=1)

    assert first == second

    first_changed = {
        before.example_id
        for before, after in zip(first.baseline_train, first.candidate_train, strict=True)
        if before.response != after.response
    }
    alternate_changed = {
        before.example_id
        for before, after in zip(
            alternate.baseline_train,
            alternate.candidate_train,
            strict=True,
        )
        if before.response != after.response
    }
    assert first_changed != alternate_changed
