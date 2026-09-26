from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from model_forensics.exp009_classifier import validate_frozen_development_partition
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_matched_benchmark import (
    MATCHED_BENCHMARK_NAMESPACE,
    build_matched_world,
    rank_matched_pairs,
    select_matched_worlds,
)
from model_forensics.exp009_release import build_clean_release_slots, release_sha256

CLEAN_RUN_IDS = {
    0: "clean_t0000_e7_bs32_lr2e5",
    1: "clean_t0001_e7_bs32_lr2e5",
    2: "clean_t0002_e7_bs32_lr2e5",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _load_jsonl(path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "true_label": str(row["true_label"]),
                "predicted_label": str(row["predicted_label"]),
            }
        )
    return tuple(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the clean-only Exp009 structurally matched benchmark preflight."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            "experiments/009_stochastic_counterfactual_certification/"
            "pilot_evidence/clean"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/matched_benchmark_preflight"),
    )
    args = parser.parse_args()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha256 = validate_frozen_development_partition(partition)
    baseline = build_clean_release_slots(partition.development_train)

    recalls_by_trajectory: dict[int, dict[str, float]] = {}
    predictions_by_trajectory: dict[int, tuple[dict[str, str], ...]] = {}
    clean_evidence: dict[str, object] = {}

    for trajectory_id, run_id in CLEAN_RUN_IDS.items():
        run_root = args.evidence_root / run_id
        summary = _load_json(run_root / "train_summary.json")
        if summary.get("official_test_split_loaded") is not False:
            raise AssertionError(f"{run_id}: official-test embargo metadata drift")
        if summary.get("development_partition_sha256") != partition_sha256:
            raise AssertionError(f"{run_id}: development-partition hash mismatch")

        metrics = summary.get("development_eval_metrics")
        if not isinstance(metrics, dict):
            raise TypeError(f"{run_id}: development_eval_metrics must be an object")
        recalls = metrics.get("per_label_recall")
        if not isinstance(recalls, dict):
            raise TypeError(f"{run_id}: per_label_recall must be an object")

        recalls_by_trajectory[trajectory_id] = {
            str(label): float(value) for label, value in recalls.items()
        }
        prediction_path = run_root / "development_eval_predictions.jsonl"
        predictions_by_trajectory[trajectory_id] = _load_jsonl(prediction_path)
        clean_evidence[str(trajectory_id)] = {
            "run_id": run_id,
            "development_partition_sha256": summary["development_partition_sha256"],
            "official_test_split_loaded": False,
            "prediction_rows": len(predictions_by_trajectory[trajectory_id]),
        }

    ranked = rank_matched_pairs(
        partition,
        per_label_recall_by_trajectory=recalls_by_trajectory,
        prediction_rows_by_trajectory=predictions_by_trajectory,
    )
    worlds = select_matched_worlds(ranked)

    world_summaries: list[dict[str, object]] = []
    for definition in worlds:
        built = build_matched_world(baseline, definition)
        world_dir = args.output_root / f"world_{definition.world_index:02d}"

        _write_json(world_dir / "diagnostic_manifest.json", built.diagnostic_manifest)
        _write_json(world_dir / "truth_manifest.json", built.truth_manifest)
        _write_json(world_dir / "structural_audit.json", built.structural_audit)

        candidate_ids = [
            str(row["candidate_id"])
            for row in built.diagnostic_manifest["candidates"]
        ]
        root_pair = definition.root_pair
        world_summaries.append(
            {
                "world_index": definition.world_index,
                "root_position": definition.root_position,
                "root_pair": [root_pair.label_a, root_pair.label_b],
                "pair_count": len(definition.pairs),
                "pairs": [
                    {
                        **asdict(pair),
                        "ranked_global_position": ranked.index(pair) + 1,
                    }
                    for pair in definition.pairs
                ],
                "candidate_ids": candidate_ids,
                "baseline_release_sha256": release_sha256(baseline),
                "composite_release_sha256": built.structural_audit[
                    "composite_release_sha256"
                ],
                "structural_audit": built.structural_audit,
            }
        )

    summary_payload = {
        "schema_version": 1,
        "benchmark_namespace": MATCHED_BENCHMARK_NAMESPACE,
        "development_partition_sha256": partition_sha256,
        "baseline_release_sha256": release_sha256(baseline),
        "clean_evidence": clean_evidence,
        "eligible_pair_count": len(ranked),
        "selected_world_count": len(worlds),
        "selected_candidate_count": sum(len(world.pairs) for world in worlds),
        "selected_unique_intent_count": len(
            {
                label
                for world in worlds
                for pair in world.pairs
                for label in (pair.label_a, pair.label_b)
            }
        ),
        "worlds": world_summaries,
        "model_training_performed": False,
        "official_test_split_loaded": False,
    }
    _write_json(args.output_root / "summary.json", summary_payload)

    print("===== EXP009 STRUCTURALLY MATCHED BENCHMARK PREFLIGHT =====")
    print(f"development_partition_sha256={partition_sha256}")
    print(f"baseline_release_sha256={release_sha256(baseline)}")
    print(f"eligible_pair_count={len(ranked)}")
    print(f"selected_world_count={len(worlds)}")
    print(
        "selected_candidate_count="
        f"{summary_payload['selected_candidate_count']}"
    )
    print(
        "selected_unique_intent_count="
        f"{summary_payload['selected_unique_intent_count']}"
    )
    for world in world_summaries:
        print(
            f"world={world['world_index']} "
            f"root_position={world['root_position']} "
            f"root_pair={world['root_pair']} "
            f"composite={world['composite_release_sha256']}"
        )
        audit = world["structural_audit"]
        print(
            f"  identical={audit['all_candidate_structures_identical']} "
            f"disjoint_slots={audit['pairwise_changed_slots_disjoint']} "
            f"disjoint_intents={audit['intent_labels_pairwise_disjoint']} "
            f"composite_order_independent={audit['composite_order_independent']}"
        )
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")
    print(f"artifact={args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
