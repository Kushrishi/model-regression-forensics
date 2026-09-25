from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

TRAJECTORIES = (0, 1, 2)
RESTORATION_STATES = (
    "restore_root",
    "restore_n1",
    "restore_n2",
    "restore_n3",
    "restore_n4",
)
ALL_STATES = ("composite", *RESTORATION_STATES)

EXPECTED_ORDERS = {
    0: (
        "composite",
        "restore_root",
        "restore_n1",
        "restore_n2",
        "restore_n3",
        "restore_n4",
    ),
    1: (
        "composite",
        "restore_n1",
        "restore_n2",
        "restore_n3",
        "restore_n4",
        "restore_root",
    ),
    2: (
        "composite",
        "restore_n3",
        "restore_n4",
        "restore_root",
        "restore_n1",
        "restore_n2",
    ),
}

EXPECTED_BASELINE_SHA256 = "cb83232c055c4c55ca50f2fbd86627d59dc806ab33a861abc7402990ddb7e20c"
EXPECTED_COMPOSITE_SHA256 = "16aaea426d1ae9a2383e9124220769957b43e1761deaa6625b930b45e664fe0f"


def run_id(trajectory_id: int, state: str) -> str:
    return f"mps_v2_t{trajectory_id:04d}_{state}_e7_bs32_lr2e5"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(root: Path, trajectory_id: int, state: str) -> dict[str, Any]:
    path = root / f"t{trajectory_id}" / "runs" / run_id(trajectory_id, state) / "train_summary.json"
    return _load_json(path)


def _describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _require_equal(values: list[Any], message: str) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise AssertionError(message)
    return first


def analyze_stage_b(
    root: Path,
    *,
    stage_a_result_path: Path,
    expected_source_git_sha: str | None = None,
) -> dict[str, Any]:
    stage_a = _load_json(stage_a_result_path)
    if stage_a.get("official_test_split_loaded") is not False:
        raise AssertionError("Stage-A official-test embargo metadata failed")
    if stage_a.get("aggregate", {}).get("stage_a_gate_pass") is not True:
        raise AssertionError("Stage B requires a recorded Stage-A gate pass")

    stage_a_by_trajectory = {int(row["trajectory_id"]): row for row in stage_a["trajectories"]}
    if set(stage_a_by_trajectory) != set(TRAJECTORIES):
        raise AssertionError("Stage-A trajectory set mismatch")

    trajectory_rows: list[dict[str, Any]] = []
    all_source_shas: list[str] = []
    all_partition_shas: list[str] = []
    all_configs: list[dict[str, Any]] = []
    all_model_artifacts: list[tuple[Any, Any]] = []
    release_hashes_by_state: dict[str, list[str]] = {state: [] for state in ALL_STATES}
    changed_slot_hashes_by_state: dict[str, list[str]] = {state: [] for state in ALL_STATES}

    for trajectory_id in TRAJECTORIES:
        session_root = root / f"t{trajectory_id}"
        manifest = _load_json(session_root / "session_manifest.json")
        if int(manifest.get("trajectory_id", -1)) != trajectory_id:
            raise AssertionError(f"t{trajectory_id}: session manifest trajectory mismatch")
        if tuple(manifest.get("state_order", ())) != EXPECTED_ORDERS[trajectory_id]:
            raise AssertionError(f"t{trajectory_id}: frozen state order mismatch")
        if manifest.get("fresh_composite_anchor") is not True:
            raise AssertionError(f"t{trajectory_id}: missing fresh-composite declaration")

        provenance_path = session_root / "runner_provenance.txt"
        if not provenance_path.exists():
            raise FileNotFoundError(provenance_path)
        provenance_sha = _sha256(provenance_path)

        summaries = {state: _summary(root, trajectory_id, state) for state in ALL_STATES}

        for state, summary in summaries.items():
            if summary.get("official_test_split_loaded") is not False:
                raise AssertionError(f"t{trajectory_id} {state}: official-test embargo failed")
            if summary.get("confirmatory_result") is not False:
                raise AssertionError(f"t{trajectory_id} {state}: confirmatory metadata drift")
            if int(summary["trajectory_id"]) != trajectory_id:
                raise AssertionError(f"t{trajectory_id} {state}: trajectory mismatch")
            if summary["runtime"]["device"] != "mps":
                raise AssertionError(f"t{trajectory_id} {state}: non-MPS runtime")
            if summary["run_id"] != run_id(trajectory_id, state):
                raise AssertionError(f"t{trajectory_id} {state}: run-ID mismatch")

            state_provenance = (
                session_root / "runs" / run_id(trajectory_id, state) / "runner_provenance.txt"
            )
            if _sha256(state_provenance) != provenance_sha:
                raise AssertionError(f"t{trajectory_id} {state}: runner provenance mismatch")

            release_hashes_by_state[state].append(str(summary["release"]["release_sha256"]))
            changed_slot_hashes_by_state[state].append(
                str(summary["release"]["diff"]["changed_slot_ids_sha256"])
            )

        summary_values = list(summaries.values())
        _require_equal(
            [value["source_git_sha"] for value in summary_values],
            f"t{trajectory_id}: source Git SHA mismatch",
        )
        _require_equal(
            [value["development_partition_sha256"] for value in summary_values],
            f"t{trajectory_id}: development partition mismatch",
        )
        _require_equal(
            [value["pilot_config"] for value in summary_values],
            f"t{trajectory_id}: training configuration mismatch",
        )
        _require_equal(
            [value["trajectory_seeds"] for value in summary_values],
            f"t{trajectory_id}: trajectory-seed mismatch",
        )
        _require_equal(
            [value["slot_schedule_sha256"] for value in summary_values],
            f"t{trajectory_id}: slot-schedule mismatch",
        )
        _require_equal(
            [value["model"]["initial_model_state_sha256"] for value in summary_values],
            f"t{trajectory_id}: initial-model-state mismatch",
        )
        _require_equal(
            [
                (
                    value["model"]["resolved_revision"],
                    value["model"]["safetensors_sha256"],
                )
                for value in summary_values
            ],
            f"t{trajectory_id}: model artifact mismatch",
        )
        _require_equal(
            [value["attribution_target"]["target_labels"] for value in summary_values],
            f"t{trajectory_id}: target-label mismatch",
        )

        for summary in summary_values:
            if summary["release"]["baseline_release_sha256"] != EXPECTED_BASELINE_SHA256:
                raise AssertionError(f"t{trajectory_id}: baseline release hash drift")

        composite = summaries["composite"]
        if composite["release"]["release_sha256"] != EXPECTED_COMPOSITE_SHA256:
            raise AssertionError(f"t{trajectory_id}: composite release hash drift")
        if int(composite["release"]["diff"]["changed_slot_count"]) != 330:
            raise AssertionError(f"t{trajectory_id}: composite changed-slot count drift")

        restoration_hashes: list[str] = []
        for state in RESTORATION_STATES:
            restoration = summaries[state]
            if int(restoration["release"]["diff"]["changed_slot_count"]) != 264:
                raise AssertionError(f"t{trajectory_id} {state}: changed-slot count drift")
            release_hash = str(restoration["release"]["release_sha256"])
            if release_hash == EXPECTED_COMPOSITE_SHA256:
                raise AssertionError(f"t{trajectory_id} {state}: restoration equals composite")
            restoration_hashes.append(release_hash)
        if len(set(restoration_hashes)) != len(restoration_hashes):
            raise AssertionError(f"t{trajectory_id}: restoration release hashes are not unique")

        target_labels = tuple(composite["behavior_slice_metrics"]["target_labels"])
        if len(target_labels) != 2:
            raise AssertionError(f"t{trajectory_id}: target-label cardinality drift")
        composite_recall = composite["development_eval_metrics"]["per_label_recall"]
        protected_labels = sorted(set(composite_recall) - set(target_labels))
        if not protected_labels:
            raise AssertionError(f"t{trajectory_id}: empty protected set")

        composite_target = float(composite["behavior_slice_metrics"]["target_macro_recall"])
        composite_protected = float(composite["behavior_slice_metrics"]["protected_macro_recall"])

        effects: dict[str, dict[str, Any]] = {}
        for state in RESTORATION_STATES:
            restoration = summaries[state]
            slices = restoration["behavior_slice_metrics"]
            per_label = restoration["development_eval_metrics"]["per_label_recall"]

            if set(per_label) != set(composite_recall):
                raise AssertionError(f"t{trajectory_id} {state}: label vocabulary drift")

            target_delta_by_label = {
                label: float(per_label[label]) - float(composite_recall[label])
                for label in target_labels
            }
            protected_deltas = {
                label: float(per_label[label]) - float(composite_recall[label])
                for label in protected_labels
            }
            max_abs_protected_delta = max(abs(value) for value in protected_deltas.values())
            max_abs_protected_labels = sorted(
                label
                for label, value in protected_deltas.items()
                if abs(value) == max_abs_protected_delta
            )

            effects[state] = {
                "target_recall": float(slices["target_macro_recall"]),
                "delta_target": float(slices["target_macro_recall"]) - composite_target,
                "target_delta_by_label": target_delta_by_label,
                "protected_macro_recall": float(slices["protected_macro_recall"]),
                "delta_protected_macro": (
                    float(slices["protected_macro_recall"]) - composite_protected
                ),
                "max_abs_protected_intent_delta": max_abs_protected_delta,
                "max_abs_protected_intent_delta_labels": max_abs_protected_labels,
                "worst_protected_intent_recall": float(slices["protected_worst_intent_recall"]),
                "release_sha256": restoration["release"]["release_sha256"],
                "elapsed_seconds": float(restoration["optimization"]["elapsed_seconds"]),
            }

        root_delta = float(effects["restore_root"]["delta_target"])
        nuisance_states = RESTORATION_STATES[1:]
        contrasts = {
            state: root_delta - float(effects[state]["delta_target"]) for state in nuisance_states
        }
        max_nuisance_delta = max(float(effects[state]["delta_target"]) for state in nuisance_states)
        root_vs_max = root_delta - max_nuisance_delta

        stage_a_row = stage_a_by_trajectory[trajectory_id]
        repeatability = {
            "stage_a_composite_target": float(stage_a_row["composite_target"]),
            "stage_b_fresh_composite_target": composite_target,
            "target_difference_stage_b_minus_stage_a": (
                composite_target - float(stage_a_row["composite_target"])
            ),
            "stage_a_composite_protected": float(stage_a_row["composite_protected"]),
            "stage_b_fresh_composite_protected": composite_protected,
            "protected_difference_stage_b_minus_stage_a": (
                composite_protected - float(stage_a_row["composite_protected"])
            ),
        }

        trajectory_rows.append(
            {
                "trajectory_id": trajectory_id,
                "session_state_order": list(EXPECTED_ORDERS[trajectory_id]),
                "runner_provenance_sha256": provenance_sha,
                "source_git_sha": composite["source_git_sha"],
                "development_partition_sha256": composite["development_partition_sha256"],
                "slot_schedule_sha256": composite["slot_schedule_sha256"],
                "initial_model_state_sha256": composite["model"]["initial_model_state_sha256"],
                "fresh_composite": {
                    "target_macro_recall": composite_target,
                    "protected_macro_recall": composite_protected,
                    "release_sha256": composite["release"]["release_sha256"],
                    "elapsed_seconds": float(composite["optimization"]["elapsed_seconds"]),
                },
                "restoration_effects": effects,
                "root_vs_nuisance_target_margins": contrasts,
                "max_nuisance_target_recovery": max_nuisance_delta,
                "root_vs_max_nuisance_margin": root_vs_max,
                "root_strictly_largest_target_recovery": root_vs_max > 0.0,
                "stage_a_composite_repeatability": repeatability,
            }
        )

        all_source_shas.extend(str(value["source_git_sha"]) for value in summary_values)
        all_partition_shas.extend(
            str(value["development_partition_sha256"]) for value in summary_values
        )
        all_configs.extend(value["pilot_config"] for value in summary_values)
        all_model_artifacts.extend(
            (
                value["model"]["resolved_revision"],
                value["model"]["safetensors_sha256"],
            )
            for value in summary_values
        )

    source_git_sha = _require_equal(all_source_shas, "cross-trajectory source Git SHA mismatch")
    if expected_source_git_sha is not None and source_git_sha != expected_source_git_sha:
        raise AssertionError(
            f"Stage-B source Git SHA {source_git_sha} != checkout {expected_source_git_sha}"
        )

    partition_sha = _require_equal(
        all_partition_shas, "cross-trajectory development partition mismatch"
    )
    _require_equal(all_configs, "cross-trajectory training configuration mismatch")
    model_artifact = _require_equal(all_model_artifacts, "cross-trajectory model artifact mismatch")

    stable_release_hashes = {
        state: _require_equal(
            hashes,
            f"cross-trajectory {state} release hash mismatch",
        )
        for state, hashes in release_hashes_by_state.items()
    }
    stable_changed_slot_hashes = {
        state: _require_equal(
            hashes,
            f"cross-trajectory {state} changed-slot hash mismatch",
        )
        for state, hashes in changed_slot_hashes_by_state.items()
    }

    root_recoveries = [
        float(row["restoration_effects"]["restore_root"]["delta_target"]) for row in trajectory_rows
    ]
    nuisance_summaries: dict[str, dict[str, Any]] = {}
    contrast_summaries: dict[str, dict[str, Any]] = {}
    for state in RESTORATION_STATES[1:]:
        nuisance_values = [
            float(row["restoration_effects"][state]["delta_target"]) for row in trajectory_rows
        ]
        contrast_values = [
            float(row["root_vs_nuisance_target_margins"][state]) for row in trajectory_rows
        ]
        nuisance_summaries[state] = _describe(nuisance_values)
        contrast_summaries[state] = {
            **_describe(contrast_values),
            "positive_count": sum(value > 0.0 for value in contrast_values),
            "zero_count": sum(value == 0.0 for value in contrast_values),
            "negative_count": sum(value < 0.0 for value in contrast_values),
        }

    root_vs_max_values = [float(row["root_vs_max_nuisance_margin"]) for row in trajectory_rows]

    return {
        "schema_version": 1,
        "experiment": "exp009_mps_certification_pilot_v2_stage_b",
        "evidence_class": "development",
        "confirmatory_result": False,
        "official_test_split_loaded": False,
        "trajectory_count": len(TRAJECTORIES),
        "fresh_composite_anchor_per_trajectory": True,
        "source_git_sha": source_git_sha,
        "development_partition_sha256": partition_sha,
        "model_artifact": {
            "resolved_revision": model_artifact[0],
            "safetensors_sha256": model_artifact[1],
        },
        "state_release_sha256": stable_release_hashes,
        "state_changed_slot_ids_sha256": stable_changed_slot_hashes,
        "trajectories": trajectory_rows,
        "aggregate": {
            "root_target_recovery": _describe(root_recoveries),
            "nuisance_target_recovery": nuisance_summaries,
            "root_vs_nuisance_target_margins": contrast_summaries,
            "root_vs_max_nuisance_margin": {
                **_describe(root_vs_max_values),
                "positive_count": sum(value > 0.0 for value in root_vs_max_values),
                "zero_count": sum(value == 0.0 for value in root_vs_max_values),
                "negative_count": sum(value < 0.0 for value in root_vs_max_values),
            },
        },
        "inference": {
            "hypothesis_tests_performed": False,
            "p_values_reported": False,
            "multiplicity_correction_applied": False,
            "causal_certification_threshold_applied": False,
            "interpretation": "descriptive effect-structure pilot",
        },
        "claim_boundary": {
            "causal_specificity_certified_by_this_analysis": False,
            "confirmatory_evidence": False,
            "structurally_matched_blind_localization": False,
        },
    }
