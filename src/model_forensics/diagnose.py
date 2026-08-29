from __future__ import annotations

from dataclasses import dataclass

from model_forensics.lineage import DiagnosticManifest


@dataclass(frozen=True)
class CandidateCause:
    """A ranked candidate cause emitted by a diagnostic method."""

    change_id: str
    score: float
    rationale: str


def rank_candidates(manifest: DiagnosticManifest) -> list[CandidateCause]:
    """Return a neutral interface baseline over ground-truth-free lineage."""

    if not isinstance(manifest, DiagnosticManifest):
        raise TypeError("diagnostic methods require a DiagnosticManifest")

    return [
        CandidateCause(change_id=change.change_id, score=0.0, rationale="unscored baseline")
        for change in manifest.changes
    ]
