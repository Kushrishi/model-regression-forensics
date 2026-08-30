from model_forensics.config import load_experiment_config


def test_exp000_config_loads() -> None:
    config = load_experiment_config("configs/exp000.yaml")

    assert config.experiment_id == "exp000"
    assert config.seed == 42
    assert config.model.revision == "a10cc1512eabd3dde888204e902eca88bddb4951"
    assert config.generation.max_new_tokens == 8
    assert config.generation.do_sample is False
    assert config.training.method == "lora_sft"
    assert config.training.epochs == 10
    assert config.training.lora_r == 16
    assert config.evaluation.primary_metric == "label_accuracy"
    assert config.regression.hidden_root_cause_id == "shard_corrupt_03"
    assert config.evaluation.minimum_regression_delta == 0.15


def test_exp001_config_loads() -> None:
    config = load_experiment_config("configs/exp001.yaml")

    assert config.experiment_id == "exp001"
    assert config.seed == 42
    assert config.regression.hidden_root_cause_id == "shard_delta_04"
    assert config.lineage.artifact_kinds == ["dataset_shard"]


def test_exp002_config_loads_without_declared_root_cause() -> None:
    config = load_experiment_config("configs/exp002.yaml")

    assert config.experiment_id == "exp002"
    assert config.seed == 42
    assert config.regression.hidden_root_cause_id is None
    assert config.evaluation.minimum_baseline_score == 0.95
    assert config.benchmark_difficulty is not None
    assert config.benchmark_difficulty.candidate_count == 5
    assert config.benchmark_difficulty.records_per_candidate == 48
    assert config.benchmark_difficulty.label_changes_per_candidate == 32
    assert config.benchmark_difficulty.lexical_overlap_max_range == 1e-12


def test_exp003_config_freezes_role_binding_difficulty_gates() -> None:
    config = load_experiment_config("configs/exp003.yaml")

    assert config.experiment_id == "exp003"
    assert config.seed == 42
    assert config.regression.hidden_root_cause_id is None
    assert config.benchmark_difficulty is not None
    assert config.benchmark_difficulty.candidate_count == 5
    assert config.benchmark_difficulty.records_per_candidate == 48
    assert config.benchmark_difficulty.label_changes_per_candidate == 36
    assert config.benchmark_difficulty.lexical_overlap_max_range == 1e-12
    assert config.benchmark_difficulty.changed_lexical_overlap_max_range == 1e-12
    assert config.benchmark_difficulty.selected_slot_count_per_changed_candidate == 6


def test_exp003b_config_changes_only_balanced_loss_and_baseline_gate() -> None:
    exp003 = load_experiment_config("configs/exp003.yaml")
    exp003b = load_experiment_config("configs/exp003b.yaml")

    assert exp003b.experiment_id == "exp003b"
    assert exp003.training.response_loss_weights is None
    assert exp003b.training.response_loss_weights == {"ACCEPT": 1.0, "REJECT": 2.0}
    assert exp003.evaluation.baseline_required_splits == ["target"]
    assert exp003b.evaluation.baseline_required_splits == ["target", "control", "all"]

    exp003_payload = exp003.model_dump()
    exp003b_payload = exp003b.model_dump()
    exp003_payload["experiment_id"] = "exp003b"
    exp003_payload["training"]["response_loss_weights"] = {"ACCEPT": 1.0, "REJECT": 2.0}
    exp003_payload["evaluation"]["baseline_required_splits"] = ["target", "control", "all"]
    assert exp003b_payload == exp003_payload


def test_exp003c_config_freezes_selected_slot_lookup_diagnostic() -> None:
    config = load_experiment_config("configs/exp003c.yaml")

    assert config.experiment_id == "exp003c"
    assert config.regression.kind == "none"
    assert config.training.response_loss_weights is None
    assert config.lineage.artifact_kinds == []
    assert config.evaluation.baseline_required_splits == [
        "slot_a",
        "slot_b",
        "slot_c",
        "slot_d",
        "slot_e",
        "slot_f",
        "all",
    ]
    assert config.capability_diagnostic is not None
    assert config.capability_diagnostic.kind == "selected_slot_lookup"
    assert config.capability_diagnostic.slot_count == 6
    assert config.capability_diagnostic.accept_per_prompt == 3
    assert config.capability_diagnostic.train_pattern_count == 16
    assert config.capability_diagnostic.eval_pattern_count == 4
    assert config.capability_diagnostic.train_contexts_per_pattern == 3
    assert config.capability_diagnostic.eval_contexts_per_pattern == 4


def test_exp003d_config_freezes_one_factor_explicit_policy_diagnostic() -> None:
    config = load_experiment_config("configs/exp003d.yaml")

    assert config.experiment_id == "exp003d"
    assert config.regression.kind == "none"
    assert config.training.response_loss_weights is None
    assert config.lineage.artifact_kinds == []
    assert config.evaluation.baseline_required_splits == [
        "circle_small",
        "circle_large",
        "square_small",
        "square_large",
        "triangle_small",
        "triangle_large",
        "all",
    ]
    assert config.capability_diagnostic is not None
    assert config.capability_diagnostic.kind == "explicit_policy_role_binding"
    assert config.capability_diagnostic.source_experiment_id == "exp003"
    assert config.capability_diagnostic.slot_count == 6
    assert config.capability_diagnostic.train_example_count == 288
    assert config.capability_diagnostic.eval_example_count == 96
    assert config.capability_diagnostic.policy == {
        "circle": "ACCEPT",
        "triangle": "ACCEPT",
        "square": "REJECT",
    }
