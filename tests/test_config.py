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
