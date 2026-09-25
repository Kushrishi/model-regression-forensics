from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_forensics.exp009_stage_b import (
    ALL_STATES,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_COMPOSITE_SHA256,
    EXPECTED_ORDERS,
    analyze_stage_b,
    run_id,
)


def _summary(trajectory: int, state: str, target: float) -> dict[str, object]:
    target_labels = ["target_a", "target_b"]
    protected = {"protected_a": 0.90, "protected_b": 0.80}
    per_label = {"target_a": target, "target_b": target, **protected}
    changed_count = 330 if state == "composite" else 264
    release_hash = EXPECTED_COMPOSITE_SHA256 if state == "composite" else f"{state}-release"
    protected_macro = sum(protected.values()) / len(protected)

    return {
        "confirmatory_result": False,
        "official_test_split_loaded": False,
        "run_id": run_id(trajectory, state),
        "source_git_sha": "source-sha",
        "trajectory_id": trajectory,
        "trajectory_seeds": {"trajectory_id": trajectory, "seed": trajectory + 1},
        "development_partition_sha256": "partition-sha",
        "pilot_config": {"epochs": 7, "batch_size": 32},
        "slot_schedule_sha256": f"schedule-{trajectory}",
        "release": {
            "baseline_release_sha256": EXPECTED_BASELINE_SHA256,
            "release_sha256": release_hash,
            "diff": {
                "changed_slot_count": changed_count,
                "changed_slot_ids_sha256": f"{state}-changed-slots",
            },
        },
        "model": {
            "resolved_revision": "model-revision",
            "safetensors_sha256": "weights-sha",
            "initial_model_state_sha256": f"initial-{trajectory}",
        },
        "attribution_target": {"target_labels": target_labels},
        "development_eval_metrics": {"per_label_recall": per_label},
        "behavior_slice_metrics": {
            "target_labels": target_labels,
            "target_macro_recall": target,
            "protected_macro_recall": protected_macro,
            "protected_worst_intent_recall": min(protected.values()),
        },
        "optimization": {"elapsed_seconds": 1.0},
        "runtime": {"device": "mps"},
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "stage_b_sessions"
    stage_a_rows = []
    target_by_state = {
        "composite": 0.50,
        "restore_root": 0.80,
        "restore_n1": 0.55,
        "restore_n2": 0.54,
        "restore_n3": 0.53,
        "restore_n4": 0.52,
    }

    for trajectory in (0, 1, 2):
        session = root / f"t{trajectory}"
        session.mkdir(parents=True)
        (session / "runner_provenance.txt").write_text(f"runner=t{trajectory}\n", encoding="utf-8")
        (session / "session_manifest.json").write_text(
            json.dumps(
                {
                    "trajectory_id": trajectory,
                    "state_order": list(EXPECTED_ORDERS[trajectory]),
                    "fresh_composite_anchor": True,
                }
            ),
            encoding="utf-8",
        )

        for state in ALL_STATES:
            run_dir = session / "runs" / run_id(trajectory, state)
            run_dir.mkdir(parents=True)
            (run_dir / "train_summary.json").write_text(
                json.dumps(_summary(trajectory, state, target_by_state[state])),
                encoding="utf-8",
            )
            (run_dir / "runner_provenance.txt").write_text(
                f"runner=t{trajectory}\n", encoding="utf-8"
            )

        stage_a_rows.append(
            {
                "trajectory_id": trajectory,
                "composite_target": 0.49,
                "composite_protected": 0.85,
            }
        )

    stage_a = tmp_path / "stage_a.json"
    stage_a.write_text(
        json.dumps(
            {
                "official_test_split_loaded": False,
                "aggregate": {"stage_a_gate_pass": True},
                "trajectories": stage_a_rows,
            }
        ),
        encoding="utf-8",
    )
    return root, stage_a


def test_stage_b_analysis_preserves_paired_effect_structure(tmp_path: Path) -> None:
    root, stage_a = _write_fixture(tmp_path)

    analysis = analyze_stage_b(
        root,
        stage_a_result_path=stage_a,
        expected_source_git_sha="source-sha",
    )

    assert analysis["trajectory_count"] == 3
    assert analysis["inference"]["hypothesis_tests_performed"] is False
    assert analysis["claim_boundary"]["causal_specificity_certified_by_this_analysis"] is False
    assert analysis["aggregate"]["root_target_recovery"]["sample_stdev"] == pytest.approx(0.0)

    for row in analysis["trajectories"]:
        assert row["restoration_effects"]["restore_root"]["delta_target"] == pytest.approx(0.30)
        assert row["max_nuisance_target_recovery"] == pytest.approx(0.05)
        assert row["root_vs_max_nuisance_margin"] == pytest.approx(0.25)
        assert row["root_strictly_largest_target_recovery"] is True

    assert analysis["aggregate"]["root_target_recovery"]["mean"] == pytest.approx(0.30)
    assert analysis["aggregate"]["root_vs_max_nuisance_margin"]["positive_count"] == 3


def test_stage_b_analysis_rejects_cross_state_initialization_drift(tmp_path: Path) -> None:
    root, stage_a = _write_fixture(tmp_path)
    path = root / "t1" / "runs" / run_id(1, "restore_n2") / "train_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["model"]["initial_model_state_sha256"] = "different"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AssertionError, match="initial-model-state mismatch"):
        analyze_stage_b(root, stage_a_result_path=stage_a)


def test_stage_b_analysis_rejects_cross_trajectory_release_drift(tmp_path: Path) -> None:
    root, stage_a = _write_fixture(tmp_path)
    path = root / "t2" / "runs" / run_id(2, "restore_n4") / "train_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["release"]["release_sha256"] = "drifted-release"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AssertionError, match="cross-trajectory restore_n4 release hash mismatch"):
        analyze_stage_b(root, stage_a_result_path=stage_a)
