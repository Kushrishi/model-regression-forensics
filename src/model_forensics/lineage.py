from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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


class LineageManifest(BaseModel):
    """Structured provenance between a baseline run and a candidate run."""

    experiment_id: str
    baseline_run_id: str
    candidate_run_id: str
    changes: list[ArtifactChange]
    hidden_root_cause_id: str | None = Field(
        default=None,
        description=(
            "Benchmark ground truth only; diagnostic code must receive a redacted manifest."
        ),
    )

    def redacted(self) -> LineageManifest:
        """Return a manifest safe to pass to a diagnostic method."""
        return self.model_copy(update={"hidden_root_cause_id": None})

    def dump(self, path: str | Path) -> None:
        """Serialize the manifest as stable, readable JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LineageManifest:
        """Load a lineage manifest from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)
