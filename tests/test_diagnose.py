import pytest

from model_forensics.diagnose import rank_candidates
from model_forensics.lineage import ArtifactChange, DiagnosticManifest, LineageManifest


def test_rank_candidates_accepts_only_diagnostic_manifest() -> None:
    change = ArtifactChange(
        change_id="shard_a",
        kind="dataset_shard",
        description="changed shard",
    )
    benchmark = LineageManifest(
        experiment_id="exp000",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=[change],
        hidden_root_cause_id="shard_a",
    )

    ranked = rank_candidates(benchmark.redacted())

    assert [candidate.change_id for candidate in ranked] == ["shard_a"]

    with pytest.raises(TypeError, match="DiagnosticManifest"):
        rank_candidates(benchmark)  # type: ignore[arg-type]


def test_diagnostic_manifest_has_no_ground_truth_field() -> None:
    manifest = DiagnosticManifest(
        experiment_id="exp000",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=[],
    )

    assert "hidden_root_cause_id" not in manifest.model_dump()
