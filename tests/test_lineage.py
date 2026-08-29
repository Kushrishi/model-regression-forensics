from model_forensics.lineage import ArtifactChange, LineageManifest


def test_redacted_manifest_hides_benchmark_ground_truth() -> None:
    manifest = LineageManifest(
        experiment_id="exp000",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        hidden_root_cause_id="bad_shard",
        changes=[
            ArtifactChange(
                change_id="bad_shard",
                kind="dataset_shard",
                description="candidate includes a corrupted shard",
            )
        ],
    )

    redacted = manifest.redacted()

    assert redacted.hidden_root_cause_id is None
    assert manifest.hidden_root_cause_id == "bad_shard"
    assert redacted.changes == manifest.changes
