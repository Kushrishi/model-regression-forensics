from pathlib import Path

from model_forensics.config import load_experiment_config


def test_exp006_config_freezes_model_training_and_evaluation_protocol() -> None:
    config = load_experiment_config(Path("configs/exp006.yaml"))

    assert config.experiment_id == "exp006"
    assert config.seed == 42

    assert config.model.name == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert config.model.revision == "a10cc1512eabd3dde888204e902eca88bddb4951"

    assert config.generation.max_new_tokens == 8
    assert config.generation.do_sample is False

    assert config.training.method == "lora_sft"
    assert config.training.epochs == 10
    assert config.training.batch_size == 8
    assert config.training.learning_rate == 0.0005
    assert config.training.weight_decay == 0.0
    assert config.training.warmup_ratio == 0.05
    assert config.training.max_length == 192
    assert config.training.max_grad_norm == 1.0
    assert config.training.lora_r == 16
    assert config.training.lora_alpha == 32
    assert config.training.lora_dropout == 0.0
    assert config.training.response_loss_weights is None

    assert config.regression.kind == "corrupted_sft_shard"
    assert config.regression.hidden_root_cause_id is None

    assert config.evaluation.primary_metric == "label_accuracy"
    assert config.evaluation.minimum_baseline_score == 0.95
    assert config.evaluation.minimum_regression_delta == 0.15
    assert config.evaluation.minimum_recovery_delta == 0.15
    assert config.evaluation.maximum_unrelated_delta == 0.05
    assert config.evaluation.baseline_required_splits == [
        "circle_small",
        "circle_large",
        "square_small",
        "square_large",
        "triangle_small",
        "triangle_large",
        "all",
    ]

    assert config.lineage.artifact_kinds == ["dataset_shard"]
    assert config.benchmark_difficulty is None
    assert config.capability_diagnostic is None


def test_exp006_protocol_pins_canonical_manifest_hash() -> None:
    readme = Path("experiments/006_semantic_balanced_causal_rca/README.md").read_text()

    assert "275743ec6bd5ce130fd149da0b621b6a9d59c578d56518c5aaca3ed897011c27" in readme
    assert 'sha256("exp006-world|42|i")' in readme
    assert 'sha256("exp006-order-control-a|42|example_id")' in readme
    assert "12 changed records" in readme
    assert "38 `ACCEPT -> REJECT`" in readme
    assert "22 `REJECT -> ACCEPT`" in readme
