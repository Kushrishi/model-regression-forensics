from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArtifactKind = Literal[
    "dataset_shard",
    "prompt_template",
    "optimizer",
    "schedule",
    "training_config",
    "checkpoint",
    "other",
]


class ArtifactChange(BaseModel):
    """One observable change between two training runs."""

    change_id: str
    kind: ArtifactKind
    description: str
    before: str | None = None
    after: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class DiagnosticManifest(BaseModel):
    """Lineage visible to a diagnostic method, with no benchmark ground truth."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    baseline_run_id: str
    candidate_run_id: str
    changes: list[ArtifactChange]

    def dump(self, path: str | Path) -> None:
        """Serialize the diagnostic manifest as stable, readable JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DiagnosticManifest:
        """Load a diagnostic lineage manifest from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)


class LineageManifest(BaseModel):
    """Benchmark-owned provenance between a baseline and candidate run."""

    experiment_id: str
    baseline_run_id: str
    candidate_run_id: str
    changes: list[ArtifactChange]
    hidden_root_cause_id: str | None = Field(
        default=None,
        description=(
            "Benchmark ground truth only; diagnostic code must receive a DiagnosticManifest."
        ),
    )

    def redacted(self) -> DiagnosticManifest:
        """Return a structurally ground-truth-free manifest for diagnosis."""

        return DiagnosticManifest(
            experiment_id=self.experiment_id,
            baseline_run_id=self.baseline_run_id,
            candidate_run_id=self.candidate_run_id,
            changes=self.changes,
        )

    def dump(self, path: str | Path) -> None:
        """Serialize the benchmark manifest as stable, readable JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LineageManifest:
        """Load a benchmark lineage manifest from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
