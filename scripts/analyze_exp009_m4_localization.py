from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from model_forensics.exp009_attribution import rank_candidate_scores
from model_forensics.exp009_m4 import (
    M4_TRAJECTORIES,
    M4CheckpointRecord,
    mean_trajectory_candidate_scores,
    validate_m4_checkpoint_records,
)
from model_forensics.exp009_m4_authorization import verify_authorization

METHODS = (
    "B0_deterministic_random",
    "B1_target_label_overlap",
    "B2_lexical_jaccard",
    "B3_final_checkpoint_grad_dot",
    "B4_seven_checkpoint_tracin",
)
STATIC_METHODS = METHODS[:3]
MODEL_METHODS = METHODS[3:]
WORLD_IDS = ("world_00", "world_01")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _checkpoint_rows(raw: object) -> tuple[M4CheckpointRecord, ...]:
    if not isinstance(raw, list):
        raise TypeError("checkpoint_records must be a list")
    records: list[M4CheckpointRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("checkpoint record must be an object")
        records.append(
            M4CheckpointRecord(
                epoch=int(item["epoch"]),
                model_state_sha256=str(item["model_state_sha256"]),
                optimizer_step_count=int(item["optimizer_step_count"]),
                producing_learning_rate=float(item["producing_learning_rate"]),
            )
        )
    return validate_m4_checkpoint_records(records)


def _numeric_scores(raw: object, *, integer: bool = False) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise TypeError("candidate score mapping must be a non-empty object")
    if integer:
        if any(type(value) is not int for value in raw.values()):
            raise ValueError("B0 scores must preserve frozen integer precision")
        return {str(key): value for key, value in raw.items()}
    scores = {str(candidate_id): float(value) for candidate_id, value in raw.items()}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("non-finite candidate score")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate already-written truth-free M4 localization trajectory artifacts."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_authorization(Path.cwd())
    if args.output.exists():
        raise FileExistsError("refusing to overwrite M4 evidence")

    records: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(args.input_root.glob("*.json")):
        row = _load_json(path)
        if row.get("mode") != "m4_blind_localization_trajectory":
            continue
        if row.get("truth_manifest_loaded") is not False:
            raise AssertionError(f"{path}: truth isolation violated")
        if row.get("official_test_split_loaded") is not False:
            raise AssertionError(f"{path}: official-test embargo violated")
        world_id = str(row.get("world_id"))
        trajectory_id = int(row.get("trajectory_id", -1))
        key = (world_id, trajectory_id)
        if key in records:
            raise AssertionError(f"duplicate M4 trajectory artifact: {key}")
        records[key] = row

    expected = {(world_id, trajectory) for world_id in WORLD_IDS for trajectory in M4_TRAJECTORIES}
    if set(records) != expected:
        raise AssertionError(
            f"M4 trajectory artifact set mismatch: expected={sorted(expected)} "
            f"observed={sorted(records)}"
        )

    source_shas = {str(row["source_git_sha"]) for row in records.values()}
    if len(source_shas) != 1:
        raise AssertionError("M4 trajectory artifacts were not produced from one source SHA")
    source_git_sha = next(iter(source_shas))
    if source_git_sha != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip():
        raise ValueError("trajectory source differs from the authorized execution commit")

    worlds: list[dict[str, object]] = []
    for world_id in WORLD_IDS:
        trajectory_rows = {
            trajectory: records[(world_id, trajectory)] for trajectory in M4_TRAJECTORIES
        }
        identities = {
            (
                str(row["development_partition_sha256"]),
                str(row["blind_bundle_sha256"]),
                str(row["diagnostic_manifest_sha256"]),
                str(row["baseline_release_sha256"]),
                str(row["composite_release_sha256"]),
                tuple(row["target_labels"]),
            )
            for row in trajectory_rows.values()
        }
        if len(identities) != 1:
            raise AssertionError(f"{world_id}: frozen world identity differs across trajectories")
        identity = next(iter(identities))

        candidate_sets: set[tuple[str, ...]] = set()
        scores_by_method: dict[str, dict[int, dict[str, float]]] = {
            method: {} for method in METHODS
        }
        trajectory_diagnostics: list[dict[str, object]] = []
        for trajectory, row in trajectory_rows.items():
            _checkpoint_rows(row.get("checkpoint_records"))
            method_scores = row.get("method_scores")
            if not isinstance(method_scores, dict) or set(method_scores) != set(METHODS):
                raise AssertionError(f"{world_id}/t{trajectory}: frozen M4 method set drift")
            for method in METHODS:
                scores = _numeric_scores(method_scores[method], integer=method == METHODS[0])
                scores_by_method[method][trajectory] = scores
                candidate_sets.add(tuple(sorted(scores)))

            clean_scalar = float(row["clean_target_mean_margin"])
            composite_scalar = float(row["composite_target_mean_margin"])
            regression = float(row["target_margin_regression"])
            if not all(
                math.isfinite(value) for value in (clean_scalar, composite_scalar, regression)
            ):
                raise ValueError("non-finite target margin")
            if abs((clean_scalar - composite_scalar) - regression) > 1e-12:
                raise AssertionError(
                    f"{world_id}/t{trajectory}: target regression arithmetic drift"
                )

            trajectory_diagnostics.append(
                {
                    "trajectory_id": trajectory,
                    "clean_target_mean_margin": clean_scalar,
                    "composite_target_mean_margin": composite_scalar,
                    "target_margin_regression": regression,
                    "paired_initial_model_state_sha256": row["paired_initial_model_state_sha256"],
                    "paired_slot_schedule_sha256": row["paired_slot_schedule_sha256"],
                    "checkpoint_records": row["checkpoint_records"],
                    "method_scores": method_scores,
                    "slot_score_sha256": row["slot_score_sha256"],
                }
            )

        if len(candidate_sets) != 1:
            raise AssertionError(
                f"{world_id}: candidate set differs across M4 methods/trajectories"
            )

        primary_scores: dict[str, dict[str, float]] = {}
        primary_rankings: dict[str, list[str]] = {}
        for method in METHODS:
            if method in STATIC_METHODS:
                first = scores_by_method[method][M4_TRAJECTORIES[0]]
                for trajectory in M4_TRAJECTORIES[1:]:
                    if scores_by_method[method][trajectory] != first:
                        raise AssertionError(
                            f"{world_id}: static method {method} changed across trajectories"
                        )
                aggregate = first
            else:
                aggregate = mean_trajectory_candidate_scores(scores_by_method[method])
            primary_scores[method] = {key: aggregate[key] for key in sorted(aggregate)}
            primary_rankings[method] = (
                sorted(aggregate, key=lambda key: (-aggregate[key], key))
                if method == METHODS[0]
                else list(rank_candidate_scores(aggregate))
            )

        worlds.append(
            {
                "world_id": world_id,
                "world_index": int(trajectory_rows[0]["world_index"]),
                "development_partition_sha256": identity[0],
                "blind_bundle_sha256": identity[1],
                "diagnostic_manifest_sha256": identity[2],
                "baseline_release_sha256": identity[3],
                "composite_release_sha256": identity[4],
                "target_labels": list(identity[5]),
                "target_example_count": int(trajectory_rows[0]["target_example_count"]),
                "trajectory_diagnostics": sorted(
                    trajectory_diagnostics, key=lambda item: int(item["trajectory_id"])
                ),
                "primary_candidate_scores": primary_scores,
                "primary_candidate_rankings": primary_rankings,
            }
        )

    output = {
        "schema_version": 1,
        "mode": "m4_blind_localization_aggregate",
        "source_git_sha": source_git_sha,
        "protocol": "research/M4_LOCALIZATION_BASELINE_PROTOCOL.md",
        "protocol_amendment": "research/M4_LOCALIZATION_BASELINE_PROTOCOL_AMENDMENT_1.md",
        "methods": list(METHODS),
        "trajectories": list(M4_TRAJECTORIES),
        "worlds": worlds,
        "world_count": len(worlds),
        "truth_manifest_loaded": False,
        "official_test_split_loaded": False,
        "blind_artifacts_finalized": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".json.sha256").write_text(
        hashlib.sha256(args.output.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )

    print("M4_BLIND_LOCALIZATION_AGGREGATE=COMPLETE")
    print(f"source_git_sha={source_git_sha}")
    print(f"world_count={len(worlds)}")
    print("truth_manifest_loaded=NO")
    print("official_test_split_loaded=NO")
    print(f"artifact={args.output}")


if __name__ == "__main__":
    main()
