from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from model_forensics.lineage import ArtifactKind


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown experiment configuration fields."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfigModel):
    """Model selected for an experiment."""

    name: str
    revision: str


class GenerationConfig(StrictConfigModel):
    """Deterministic generation settings for model evaluation."""

    max_new_tokens: int = Field(gt=0)
    do_sample: Literal[False] = False


class TrainingConfig(StrictConfigModel):
    """LoRA SFT settings shared by baseline, candidate, and recovery runs."""

    method: Literal["lora_sft"]
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    max_length: int = Field(gt=0)
    max_grad_norm: float = Field(gt=0.0)
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_dropout: float = Field(ge=0.0, lt=1.0)
    lora_target_modules: list[str]
    response_loss_weights: dict[str, PositiveFloat] | None = None


class RegressionConfig(StrictConfigModel):
    """Regression specification owned by the experiment harness."""

    kind: Literal["corrupted_sft_shard", "none"]
    hidden_root_cause_id: str | None = None


class BenchmarkDifficultyConfig(StrictConfigModel):
    """Construction-level difficulty constraints checked before model training."""

    candidate_count: int = Field(gt=1)
    records_per_candidate: int = Field(gt=0)
    label_changes_per_candidate: int = Field(gt=0)
    lexical_overlap_max_range: float = Field(ge=0.0)
    changed_lexical_overlap_max_range: float | None = Field(default=None, ge=0.0)
    selected_slot_count_per_changed_candidate: int | None = Field(default=None, gt=0)


class CapabilityDiagnosticConfig(StrictConfigModel):
    """Prospectively frozen construction settings for a capability diagnostic."""

    kind: Literal["selected_slot_lookup"]
    slot_count: int = Field(gt=1)
    accept_per_prompt: int = Field(gt=0)
    train_pattern_count: int = Field(gt=0)
    eval_pattern_count: int = Field(gt=0)
    train_contexts_per_pattern: int = Field(gt=0)
    eval_contexts_per_pattern: int = Field(gt=0)


class EvaluationConfig(StrictConfigModel):
    """Primary metric and thresholds required for a successful experiment."""

    primary_metric: Literal["label_accuracy"]
    minimum_baseline_score: float = Field(ge=0.0, le=1.0)
    minimum_regression_delta: float = Field(ge=0.0, le=1.0)
    minimum_recovery_delta: float = Field(ge=0.0, le=1.0)
    maximum_unrelated_delta: float = Field(ge=0.0, le=1.0)
    baseline_required_splits: list[str] = Field(default_factory=lambda: ["target"], min_length=1)


class LineageConfig(StrictConfigModel):
    """Artifact kinds the benchmark is designed to represent."""

    artifact_kinds: list[ArtifactKind]


class ExperimentConfig(StrictConfigModel):
    """Validated configuration for one regression-forensics experiment."""

    experiment_id: str
    seed: int
    model: ModelConfig
    generation: GenerationConfig
    training: TrainingConfig
    regression: RegressionConfig
    evaluation: EvaluationConfig
    lineage: LineageConfig
    benchmark_difficulty: BenchmarkDifficultyConfig | None = None
    capability_diagnostic: CapabilityDiagnosticConfig | None = None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(payload)
