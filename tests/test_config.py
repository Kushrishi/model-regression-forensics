from model_forensics.config import load_experiment_config


def test_exp000_config_loads() -> None:
    config = load_experiment_config("configs/exp000.yaml")

    assert config.experiment_id == "exp000"
    assert config.seed == 42
    assert config.regression.hidden_root_cause_id == "shard_corrupt_03"
    assert config.evaluation.minimum_regression_delta == 0.15
