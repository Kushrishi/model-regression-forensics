from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from model_forensics.exp009_classifier import validate_frozen_development_partition
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_m4_bundle import (
    build_m4_blind_world_bundle,
    write_m4_blind_world_bundle,
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reconstruct_worlds(
    *,
    cache_path: Path,
    evidence_root: Path,
) -> tuple[str, object, tuple[object, ...]]:
    records = load_banking77_train(cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha256 = validate_frozen_development_partition(partition)
    baseline = build_clean_release_slots(partition.development_train)

    recalls_by_trajectory: dict[int, dict[str, float]] = {}
    predictions_by_trajectory: dict[int, tuple[dict[str, str], ...]] = {}
    for trajectory_id, run_id in CLEAN_RUN_IDS.items():
        run_root = evidence_root / run_id
        summary = _load_json(run_root / "train_summary.json")
        if summary.get("official_test_split_loaded") is not False:
            raise AssertionError(f"{run_id}: official-test embargo metadata drift")
        if summary.get("development_partition_sha256") != partition_sha256:
            raise AssertionError(f"{run_id}: development-partition hash mismatch")

        metrics = summary.get("development_eval_metrics")
        if not isinstance(metrics, dict) or not isinstance(metrics.get("per_label_recall"), dict):
            raise TypeError(f"{run_id}: missing per-label clean evidence")
        recalls_by_trajectory[trajectory_id] = {
            str(label): float(value) for label, value in metrics["per_label_recall"].items()
        }
        predictions_by_trajectory[trajectory_id] = _load_jsonl(
            run_root / "development_eval_predictions.jsonl"
        )

    ranked = rank_matched_pairs(
        partition,
        per_label_recall_by_trajectory=recalls_by_trajectory,
        prediction_rows_by_trajectory=predictions_by_trajectory,
    )
    definitions = select_matched_worlds(ranked)
    worlds = tuple(build_matched_world(baseline, definition) for definition in definitions)
    return partition_sha256, baseline, worlds


def _assert_matches_frozen_m3(
    *,
    frozen: dict[str, Any],
    partition_sha256: str,
    baseline: object,
    worlds: tuple[object, ...],
) -> None:
    if frozen.get("development_partition_sha256") != partition_sha256:
        raise AssertionError("M4 reconstruction differs from frozen M3 development partition")
    if frozen.get("experiment") != "exp009_structurally_matched_benchmark":
        raise AssertionError("unexpected frozen M3 record")
    if frozen.get("baseline_release_sha256") != release_sha256(baseline):
        raise AssertionError("M4 reconstruction differs from frozen M3 baseline release")

    frozen_worlds = frozen.get("worlds")
    if not isinstance(frozen_worlds, list) or len(frozen_worlds) != len(worlds):
        raise AssertionError("M4 reconstruction differs from frozen M3 world count")

    for built, expected in zip(worlds, frozen_worlds, strict=True):
        if not isinstance(expected, dict):
            raise TypeError("frozen M3 world record must be an object")
        definition = built.definition
        if expected.get("world_index") != definition.world_index:
            raise AssertionError("M4 world index drift from frozen M3")
        if expected.get("composite_release_sha256") != release_sha256(built.composite):
            raise AssertionError(f"world {definition.world_index}: composite release hash drift")
        candidate_ids = [
            str(row["candidate_id"]) for row in built.diagnostic_manifest["candidates"]
        ]
        if expected.get("candidate_ids_sorted") != sorted(candidate_ids):
            raise AssertionError(f"world {definition.world_index}: opaque candidate identity drift")
        for name, payload in (
            ("diagnostic_manifest", built.diagnostic_manifest),
            ("truth_manifest", built.truth_manifest),
            ("structural_audit", built.structural_audit),
        ):
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            key = f"world_{definition.world_index:02d}/{name}.json"
            if hashlib.sha256(encoded).hexdigest() != frozen["artifact_sha256"][key]:
                raise AssertionError(f"M3 artifact identity drift: {key}")
        root_pair = [definition.root_pair.label_a, definition.root_pair.label_b]
        if expected.get("root_pair") != root_pair:
            raise AssertionError(f"world {definition.world_index}: target behavior drift")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the frozen M4 blind localization inputs and separate truth artifact."
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
            "experiments/009_stochastic_counterfactual_certification/pilot_evidence/clean"
        ),
    )
    parser.add_argument(
        "--frozen-m3",
        type=Path,
        default=Path(
            "experiments/009_stochastic_counterfactual_certification/"
            "MATCHED_BENCHMARK_PREFLIGHT_RESULT.json"
        ),
    )
    parser.add_argument(
        "--blind-root",
        type=Path,
        default=Path("artifacts/exp009/m4/blind"),
    )
    parser.add_argument(
        "--truth-root",
        type=Path,
        default=Path("artifacts/exp009/m4/truth"),
    )
    args = parser.parse_args()

    if args.blind_root.resolve() == args.truth_root.resolve():
        raise ValueError("M4 blind and truth roots must be physically distinct")

    partition_sha256, baseline, worlds = _reconstruct_worlds(
        cache_path=args.cache_path,
        evidence_root=args.evidence_root,
    )
    frozen = _load_json(args.frozen_m3)
    _assert_matches_frozen_m3(
        frozen=frozen,
        partition_sha256=partition_sha256,
        baseline=baseline,
        worlds=worlds,
    )

    blind_summary: list[dict[str, object]] = []
    truth_summary: list[dict[str, object]] = []
    for built in worlds:
        definition = built.definition
        target_labels = (definition.root_pair.label_a, definition.root_pair.label_b)
        blind = build_m4_blind_world_bundle(
            world_index=definition.world_index,
            benchmark_namespace=MATCHED_BENCHMARK_NAMESPACE,
            development_partition_sha256=partition_sha256,
            baseline_release=built.baseline,
            composite_release=built.composite,
            diagnostic_manifest=built.diagnostic_manifest,
            target_labels=target_labels,
        )
        blind_path = args.blind_root / f"world_{definition.world_index:02d}.json"
        write_m4_blind_world_bundle(blind_path, blind)

        truth = {
            "schema_version": 1,
            "mode": "m4_truth_only_world",
            "world_id": f"world_{definition.world_index:02d}",
            "world_index": definition.world_index,
            "benchmark_namespace": MATCHED_BENCHMARK_NAMESPACE,
            "development_partition_sha256": partition_sha256,
            "baseline_release_sha256": release_sha256(built.baseline),
            "composite_release_sha256": release_sha256(built.composite),
            "truth_manifest": built.truth_manifest,
            "official_test_split_loaded": False,
            "model_training_performed": False,
        }
        truth_path = args.truth_root / f"world_{definition.world_index:02d}.json"
        _write_json(truth_path, truth)

        blind_summary.append(
            {
                "world_id": blind["world_id"],
                "path": str(blind_path),
                "candidate_count": built.diagnostic_manifest["candidate_count"],
                "target_labels": list(target_labels),
                "composite_release_sha256": release_sha256(built.composite),
                "truth_manifest_loaded": False,
            }
        )
        truth_summary.append(
            {
                "world_id": truth["world_id"],
                "path": str(truth_path),
                "truth_manifest_present": True,
            }
        )

    _write_json(
        args.blind_root / "summary.json",
        {
            "schema_version": 1,
            "mode": "m4_blind_localization_bundle",
            "development_partition_sha256": partition_sha256,
            "worlds": blind_summary,
            "world_count": len(blind_summary),
            "official_test_split_loaded": False,
            "truth_manifest_loaded": False,
            "model_training_performed": False,
        },
    )
    _write_json(
        args.truth_root / "summary.json",
        {
            "schema_version": 1,
            "mode": "m4_truth_only_bundle",
            "development_partition_sha256": partition_sha256,
            "worlds": truth_summary,
            "world_count": len(truth_summary),
            "official_test_split_loaded": False,
            "model_training_performed": False,
        },
    )

    print("M4_BLIND_BUNDLE=READY")
    print(f"development_partition_sha256={partition_sha256}")
    print(f"world_count={len(worlds)}")
    print("official_test_split_loaded=NO")
    print("blind_truth_manifest_loaded=NO")
    print(f"blind_root={args.blind_root}")
    print(f"truth_root={args.truth_root}")


if __name__ == "__main__":
    main()
