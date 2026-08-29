from model_forensics.task import (
    REGRESSION_SHARD_ID,
    TARGET_SLICE_ID,
    build_exp000_data,
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
