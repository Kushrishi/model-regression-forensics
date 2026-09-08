from pathlib import Path

import pytest
from pydantic import ValidationError

from model_forensics.config import SensitivityCalibrationConfig, load_experiment_config


def test_exp007_config_freezes_model_training_and_calibration_scaffold() -> None:
    config = load_experiment_config(Path("configs/exp007.yaml"))

    assert config.experiment_id == "exp007"
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

    assert config.calibration is not None
    assert config.calibration.kind == "sensitivity_grid"
    assert config.calibration.target_doses == [9, 18]
    assert config.calibration.changes_per_candidate == 36
    assert config.calibration.accept_to_reject_per_candidate == 24
    assert config.calibration.reject_to_accept_per_candidate == 12
    assert config.calibration.material_count_min == 2
    assert config.calibration.material_count_max == 4
    assert config.calibration.require_identical_material_histogram is True
    assert config.calibration.calibration_world_count == 2
    assert config.calibration.minimum_passing_worlds == 2
    assert config.calibration.selection_rule == "minimum_target_dose_meeting_gate"
    assert config.calibration.calibration_materials == [
        "bronze",
        "cotton",
        "quartz",
        "velvet",
    ]
    assert config.calibration.certification_world_count == 5


def test_sensitivity_calibration_rejects_invalid_selection_protocol() -> None:
    with pytest.raises(ValidationError):
        SensitivityCalibrationConfig(
            kind="sensitivity_grid",
            target_doses=[18, 9],
            changes_per_candidate=36,
            accept_to_reject_per_candidate=24,
            reject_to_accept_per_candidate=12,
            material_count_min=2,
            material_count_max=4,
            require_identical_material_histogram=True,
            calibration_world_count=2,
            minimum_passing_worlds=2,
            selection_rule="minimum_target_dose_meeting_gate",
            calibration_materials=["bronze", "cotton", "quartz", "velvet"],
            certification_world_count=5,
        )

    with pytest.raises(ValidationError):
        SensitivityCalibrationConfig(
            kind="sensitivity_grid",
            target_doses=[9, 18],
            changes_per_candidate=36,
            accept_to_reject_per_candidate=24,
            reject_to_accept_per_candidate=12,
            material_count_min=2,
            material_count_max=4,
            require_identical_material_histogram=True,
            calibration_world_count=2,
            minimum_passing_worlds=3,
            selection_rule="minimum_target_dose_meeting_gate",
            calibration_materials=["bronze", "cotton", "quartz", "velvet"],
            certification_world_count=5,
        )


def test_exp007_readme_preserves_pre_model_boundary() -> None:
    readme = Path("experiments/007_sensitivity_calibrated_causal_rca/README.md").read_text()

    assert "9, 18" in readme
    assert 'sha256("exp007-calibration|42|target_dose|world_index")' in readme
    assert 'sha256("exp007-certification|42|target_dose|world_index")' in readme
    assert "No Experiment 007 behavioral result was observed before this execution-layer" in readme
    assert "a6c5be745c5f4f4a97db7bf882651591886d25a0f125d967a147b861b19bdc28" in readme
