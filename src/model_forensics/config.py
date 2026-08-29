from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from model_forensics.lineage import ArtifactKind


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown experiment configuration fields."""

    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfigModel):
    """Model selected for an experiment."""

    name: str


class RegressionConfig(StrictConfigModel):
    """Planted regression specification owned by the experiment harness."""

    kind: Literal["corrupted_sft_shard"]
    hidden_root_cause_id: str


class EvaluationConfig(StrictConfigModel):
    """Quantitative thresholds required for a successful experiment."""

    minimum_baseline_score: float = Field(ge=0.0, le=1.0)
    minimum_regression_delta: float = Field(ge=0.0, le=1.0)
    minimum_recovery_delta: float = Field(ge=0.0, le=1.0)
    maximum_unrelated_delta: float = Field(ge=0.0, le=1.0)


class LineageConfig(StrictConfigModel):
    """Artifact kinds the benchmark is designed to represent."""

    artifact_kinds: list[ArtifactKind]


class ExperimentConfig(StrictConfigModel):
    """Validated configuration for one regression-forensics experiment."""

    experiment_id: str
    seed: int
    model: ModelConfig
    regression: RegressionConfig
    evaluation: EvaluationConfig
    lineage: LineageConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(payload)
