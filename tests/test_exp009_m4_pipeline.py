from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SOURCE = "a" * 40


def script(name, monkeypatch):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "verify_authorization", lambda root: {})
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: SOURCE)
    return module


def run(module, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["script", *(str(value) for value in args)])
    module.main()


def trajectories(root, methods):
    root.mkdir()
    for world in ("world_00", "world_01"):
        for trajectory in range(3):
            scores = {method: {"a": 1.0, "b": 2.0} for method in methods}
            # These neighboring uint64 scores collapse to the same float.
            scores[methods[0]] = {"a": 2**63, "b": 2**63 + 1}
            row = {
                "mode": "m4_blind_localization_trajectory",
                "source_git_sha": SOURCE,
                "truth_manifest_loaded": False,
                "official_test_split_loaded": False,
                "world_id": world,
                "world_index": int(world[-1]),
                "trajectory_id": trajectory,
                "development_partition_sha256": "partition",
                "blind_bundle_sha256": world,
                "diagnostic_manifest_sha256": world,
                "baseline_release_sha256": "baseline",
                "composite_release_sha256": world,
                "target_labels": ["x", "y"],
                "target_example_count": 40,
                "clean_target_mean_margin": 0.8,
                "composite_target_mean_margin": 0.3,
                "target_margin_regression": 0.5,
                "paired_initial_model_state_sha256": "initial",
                "paired_slot_schedule_sha256": "schedule",
                "checkpoint_records": [
                    {
                        "epoch": epoch,
                        "model_state_sha256": f"{epoch:064x}",
                        "optimizer_step_count": epoch * 250,
                        "producing_learning_rate": 1e-6,
                    }
                    for epoch in range(1, 8)
                ],
                "method_scores": scores,
                "slot_score_sha256": {},
            }
            (root / f"{world}_t{trajectory}.json").write_text(json.dumps(row))


def test_complete_aggregate_preserves_uint64_ranking_and_finalizes_digest(tmp_path, monkeypatch):
    module = script("analyze_exp009_m4_localization", monkeypatch)
    root, output = tmp_path / "inputs", tmp_path / "aggregate.json"
    trajectories(root, module.METHODS)
    run(module, monkeypatch, "--input-root", root, "--output", output)
    aggregate = json.loads(output.read_text())
    assert aggregate["world_count"] == 2
    assert aggregate["truth_manifest_loaded"] is False
    for world in aggregate["worlds"]:
        assert world["primary_candidate_rankings"][module.METHODS[0]] == ["b", "a"]
        assert world["primary_candidate_scores"][module.METHODS[0]]["b"] == 2**63 + 1
    assert (
        output.with_suffix(".json.sha256").read_text().strip()
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        run(module, monkeypatch, "--input-root", root, "--output", output)


@pytest.mark.parametrize("failure", ["missing", "static-drift", "truth", "source", "margin"])
def test_aggregate_rejects_incomplete_or_drifted_evidence(tmp_path, monkeypatch, failure):
    module = script("analyze_exp009_m4_localization", monkeypatch)
    root, output = tmp_path / "inputs", tmp_path / "aggregate.json"
    trajectories(root, module.METHODS)
    path = root / "world_00_t1.json"
    row = json.loads(path.read_text())
    if failure == "missing":
        path.unlink()
    else:
        if failure == "static-drift":
            row["method_scores"][module.METHODS[0]]["a"] += 1
        elif failure == "truth":
            row["truth_manifest_loaded"] = True
        elif failure == "source":
            row["source_git_sha"] = "b" * 40
        else:
            row["target_margin_regression"] = 0.1
        path.write_text(json.dumps(row))
    with pytest.raises((AssertionError, ValueError)):
        run(module, monkeypatch, "--input-root", root, "--output", output)
    assert not output.exists()


def test_truth_scoring_checks_persisted_digest_before_truth_access(tmp_path, monkeypatch):
    module = script("score_exp009_m4_truth", monkeypatch)
    blind, digest = tmp_path / "blind.json", tmp_path / "blind.json.sha256"
    blind.write_text("{}")
    digest.write_text("0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        run(
            module,
            monkeypatch,
            "--blind-aggregate",
            blind,
            "--blind-sha256-file",
            digest,
            "--truth-root",
            tmp_path / "does-not-exist",
            "--output",
            tmp_path / "output.json",
        )


def test_synthetic_pipeline_scores_only_finalized_blind_rankings(tmp_path, monkeypatch):
    aggregate = script("analyze_exp009_m4_localization", monkeypatch)
    root, blind = tmp_path / "inputs", tmp_path / "aggregate.json"
    trajectories(root, aggregate.METHODS)
    run(aggregate, monkeypatch, "--input-root", root, "--output", blind)
    truth_root = tmp_path / "truth"
    truth_root.mkdir()
    for world in ("world_00", "world_01"):
        truth = {
            "mode": "m4_truth_only_world",
            "world_id": world,
            "development_partition_sha256": "partition",
            "baseline_release_sha256": "baseline",
            "composite_release_sha256": world,
            "truth_manifest": {"truth": [{"candidate_id": "b", "internal_role": "root"}]},
        }
        (truth_root / f"{world}.json").write_text(json.dumps(truth))
    scorer = script("score_exp009_m4_truth", monkeypatch)
    output = tmp_path / "evaluation.json"
    run(
        scorer,
        monkeypatch,
        "--blind-aggregate",
        blind,
        "--blind-sha256-file",
        blind.with_suffix(".json.sha256"),
        "--truth-root",
        truth_root,
        "--output",
        output,
    )
    result = json.loads(output.read_text())
    assert result["blind_artifacts_finalized_before_truth"] is True
    assert result["official_test_split_loaded"] is False
    assert all(row["top1_worlds"] == 2 for row in result["method_summary"].values())
