from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from model_forensics.exp009_m4_authorization import verify_authorization

METHODS = (
    "B0_deterministic_random",
    "B1_target_label_overlap",
    "B2_lexical_jaccard",
    "B3_final_checkpoint_grad_dot",
    "B4_seven_checkpoint_tracin",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _root_candidate_id(truth_world: dict[str, Any]) -> str:
    truth_manifest = truth_world.get("truth_manifest")
    if not isinstance(truth_manifest, dict):
        raise TypeError("M4 truth artifact is missing truth_manifest")
    rows = truth_manifest.get("truth")
    if not isinstance(rows, list):
        raise TypeError("M4 truth manifest is missing truth rows")
    roots = [
        str(row["candidate_id"])
        for row in rows
        if isinstance(row, dict) and row.get("internal_role") == "root"
    ]
    if len(roots) != 1:
        raise AssertionError(f"expected exactly one root candidate; observed={roots}")
    return roots[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score finalized blind M4 rankings against separately held benchmark truth."
    )
    parser.add_argument("--blind-aggregate", type=Path, required=True)
    parser.add_argument("--blind-sha256-file", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_authorization(Path.cwd())
    if args.output.exists():
        raise FileExistsError("refusing to overwrite M4 evidence")

    blind_bytes = args.blind_aggregate.read_bytes()
    if hashlib.sha256(blind_bytes).hexdigest() != args.blind_sha256_file.read_text().strip():
        raise ValueError("finalized blind aggregate digest mismatch")
    blind = json.loads(blind_bytes)
    if not isinstance(blind, dict):
        raise TypeError("M4 blind aggregate must be a JSON object")
    if blind.get("mode") != "m4_blind_localization_aggregate":
        raise ValueError("unexpected M4 blind aggregate mode")
    if blind.get("blind_artifacts_finalized") is not True:
        raise AssertionError("M4 truth scoring requires finalized blind artifacts")
    if blind.get("truth_manifest_loaded") is not False:
        raise AssertionError("M4 blind aggregate already reports truth access")
    if blind.get("official_test_split_loaded") is not False:
        raise AssertionError("M4 blind aggregate violated official-test embargo")
    if (
        blind["source_git_sha"]
        != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ):
        raise ValueError("blind aggregate source differs from the authorized execution commit")

    worlds = blind.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != 2:
        raise AssertionError("M4 truth scoring requires exactly two frozen worlds")
    if {world["world_id"] for world in worlds} != {"world_00", "world_01"}:
        raise ValueError("frozen truth scoring world set drift")

    scored_worlds: list[dict[str, object]] = []
    top1_counts = {method: 0 for method in METHODS}
    for world in worlds:
        if not isinstance(world, dict):
            raise TypeError("M4 blind world summary must be an object")
        world_id = str(world["world_id"])
        truth_path = args.truth_root / f"{world_id}.json"
        truth = _load_json(truth_path)
        if truth.get("mode") != "m4_truth_only_world":
            raise ValueError(f"{world_id}: unexpected truth-artifact mode")
        if truth.get("world_id") != world_id:
            raise AssertionError(f"{world_id}: truth world identity mismatch")
        for key in (
            "development_partition_sha256",
            "baseline_release_sha256",
            "composite_release_sha256",
        ):
            if truth.get(key) != world.get(key):
                raise AssertionError(f"{world_id}: truth/blind identity mismatch for {key}")

        root_candidate_id = _root_candidate_id(truth)
        rankings = world.get("primary_candidate_rankings")
        if not isinstance(rankings, dict) or set(rankings) != set(METHODS):
            raise AssertionError(f"{world_id}: frozen method ranking set drift")

        method_results: dict[str, object] = {}
        for method in METHODS:
            ranking = rankings[method]
            if not isinstance(ranking, list) or root_candidate_id not in ranking:
                raise AssertionError(f"{world_id}/{method}: root absent from finalized ranking")
            root_rank = ranking.index(root_candidate_id) + 1
            top1 = root_rank == 1
            top1_counts[method] += int(top1)
            method_results[method] = {
                "root_rank": root_rank,
                "top1": top1,
            }

        scored_worlds.append(
            {
                "world_id": world_id,
                "root_candidate_id": root_candidate_id,
                "methods": method_results,
                "truth_artifact_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
            }
        )

    summary = {
        method: {
            "top1_worlds": top1_counts[method],
            "world_count": len(scored_worlds),
            "top1_accuracy": top1_counts[method] / len(scored_worlds),
        }
        for method in METHODS
    }
    output = {
        "schema_version": 1,
        "mode": "m4_truth_evaluation",
        "source_git_sha": blind["source_git_sha"],
        "blind_aggregate_sha256": hashlib.sha256(blind_bytes).hexdigest(),
        "blind_artifacts_finalized_before_truth": True,
        "official_test_split_loaded": False,
        "worlds": scored_worlds,
        "method_summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("M4_TRUTH_EVALUATION=COMPLETE")
    print(f"blind_aggregate_sha256={output['blind_aggregate_sha256']}")
    print("blind_artifacts_finalized_before_truth=YES")
    print("official_test_split_loaded=NO")
    print(f"artifact={args.output}")


if __name__ == "__main__":
    main()
