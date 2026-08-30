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
