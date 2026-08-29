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
