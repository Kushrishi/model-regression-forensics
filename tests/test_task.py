from model_forensics.task import (
    EXP001_CHANGED_SHARD_IDS,
    EXP001_CONTROL_SLICE_ID,
    EXP001_SHARD_BY_SLICE,
    REGRESSION_SHARD_ID,
    TARGET_SLICE_ID,
    build_exp000_data,
    build_exp001_data,
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
