import pytest

from model_forensics.lineage import ArtifactChange, DiagnosticManifest, LineageManifest


def _manifest() -> LineageManifest:
    return LineageManifest(
        experiment_id="exp000",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        hidden_root_cause_id="bad_shard",
        changes=[
            ArtifactChange(
                change_id="bad_shard",
                kind="dataset_shard",
                description="candidate includes a changed shard",
            )
        ],
    )


def test_redacted_manifest_structurally_hides_benchmark_ground_truth() -> None:
    manifest = _manifest()
    redacted = manifest.redacted()

    assert isinstance(redacted, DiagnosticManifest)
    assert "hidden_root_cause_id" not in DiagnosticManifest.model_fields
    assert "hidden_root_cause_id" not in redacted.model_dump()
    assert manifest.hidden_root_cause_id == "bad_shard"
    assert redacted.changes == manifest.changes


def test_diagnostic_manifest_rejects_hidden_ground_truth_field() -> None:
    payload = _manifest().model_dump()

    with pytest.raises(ValueError):
        DiagnosticManifest.model_validate(payload)
