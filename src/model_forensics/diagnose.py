from __future__ import annotations

from dataclasses import dataclass

from model_forensics.lineage import LineageManifest


@dataclass(frozen=True)
class CandidateCause:
    """A ranked candidate cause emitted by a diagnostic method."""

    change_id: str
    score: float
    rationale: str


def rank_candidates(manifest: LineageManifest) -> list[CandidateCause]:
    """Return a neutral interface baseline without reading benchmark ground truth."""
    safe_manifest = manifest.redacted()
    return [
        CandidateCause(change_id=change.change_id, score=0.0, rationale="unscored baseline")
        for change in safe_manifest.changes
    ]
