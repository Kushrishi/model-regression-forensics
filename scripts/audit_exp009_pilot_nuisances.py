from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from model_forensics.exp009_classifier import validate_frozen_development_partition
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_nuisance import (
    build_balanced_cross_intent_refresh,
    combine_disjoint_release_changes,
    duplicate_model_content_profile,
    nuisance_refresh_audit,
    rank_nuisance_pairs,
    select_disjoint_nuisance_pairs,
)
from model_forensics.exp009_release import (
    build_clean_release_slots,
    build_symmetric_label_swap_candidate,
    changed_slot_ids,
    release_diff_manifest,
    release_sha256,
    restore_release_slots,
)

TARGET_A = "Refund_not_showing_up"
TARGET_B = "request_refund"
PER_DIRECTION = 33
CLEAN_RUN_IDS = {
    0: "clean_t0000_e7_bs32_lr2e5",
    1: "clean_t0001_e7_bs32_lr2e5",
    2: "clean_t0002_e7_bs32_lr2e5",
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(
                {
                    "content_id": str(row["content_id"]),
                    "true_label": str(row["true_label"]),
                    "predicted_label": str(row["predicted_label"]),
                }
            )
    return tuple(rows)


def _write_absent_or_identical(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite different audit artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _text_change_count(baseline, candidate) -> int:
    baseline_by_id = {slot.slot_id: slot for slot in baseline}
    candidate_by_id = {slot.slot_id: slot for slot in candidate}
    return sum(
        baseline_by_id[slot_id].text != candidate_by_id[slot_id].text
        for slot_id in changed_slot_ids(baseline, candidate)
    )


def _label_count_delta(baseline, candidate) -> dict[str, int]:
    delta = Counter(slot.label for slot in candidate)
    delta.subtract(Counter(slot.label for slot in baseline))
    return {label: value for label, value in sorted(delta.items()) if value}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and structurally audit Exp009 pilot nuisance updates"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--clean-runs-root",
        type=Path,
        default=Path("artifacts/exp009/classifier_pilot/runs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/pilot_nuisance_audit"),
    )
    args = parser.parse_args()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha256 = validate_frozen_development_partition(partition)
    baseline = build_clean_release_slots(partition.development_train)

    recall_by_trajectory: dict[int, dict[str, float]] = {}
    predictions_by_trajectory: dict[int, tuple[dict[str, str], ...]] = {}
    for trajectory_id, run_id in CLEAN_RUN_IDS.items():
        run_root = args.clean_runs_root / run_id
        summary = _load_json(run_root / "train_summary.json")
        if summary.get("official_test_split_loaded") is not False:
            raise AssertionError(f"clean trajectory {trajectory_id} test embargo is not intact")
        if summary.get("development_partition_sha256") != partition_sha256:
            raise AssertionError(f"clean trajectory {trajectory_id} partition hash mismatch")
        metrics = summary["development_eval_metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("clean development metrics must be a mapping")
        recalls = metrics["per_label_recall"]
        if not isinstance(recalls, dict):
            raise TypeError("clean per-label recall must be a mapping")
        recall_by_trajectory[trajectory_id] = {
            str(label): float(value) for label, value in recalls.items()
        }
        predictions_by_trajectory[trajectory_id] = _load_jsonl(
            run_root / "development_eval_predictions.jsonl"
        )

    ranked = rank_nuisance_pairs(
        partition,
        per_label_recall_by_trajectory=recall_by_trajectory,
        prediction_rows_by_trajectory=predictions_by_trajectory,
        excluded_labels=frozenset({TARGET_A, TARGET_B}),
    )
    selected = select_disjoint_nuisance_pairs(ranked, count=4)

    root = build_symmetric_label_swap_candidate(
        baseline,
        label_a=TARGET_A,
        label_b=TARGET_B,
        per_direction=PER_DIRECTION,
    )
    root_changed = set(changed_slot_ids(baseline, root))
    root_manifest = release_diff_manifest(baseline, root)
    root_text_changes = _text_change_count(baseline, root)

    nuisance_releases = []
    nuisance_payloads: list[dict[str, object]] = []
    nuisance_changed_sets: list[set[str]] = []

    for nuisance_index, pair in enumerate(selected, start=1):
        candidate, selection = build_balanced_cross_intent_refresh(
            baseline,
            nuisance_index=nuisance_index,
            label_a=pair.label_a,
            label_b=pair.label_b,
            per_direction=PER_DIRECTION,
        )
        audit = nuisance_refresh_audit(baseline, candidate, selection)
        changed = set(changed_slot_ids(baseline, candidate))

        if changed & root_changed:
            raise AssertionError("nuisance changed slots overlap the planted root")
        if any(changed & prior for prior in nuisance_changed_sets):
            raise AssertionError("nuisance changed slots overlap another nuisance")
        nuisance_changed_sets.append(changed)

        expected_transitions = {
            f"{pair.label_a}->{pair.label_b}": PER_DIRECTION,
            f"{pair.label_b}->{pair.label_a}": PER_DIRECTION,
        }
        if audit["label_transitions"] != expected_transitions:
            raise AssertionError("nuisance label transitions do not match the frozen footprint")
        if audit["changed_slot_count"] != 2 * PER_DIRECTION:
            raise AssertionError("nuisance changed-slot count does not match the frozen footprint")
        if audit["text_change_count"] != 2 * PER_DIRECTION:
            raise AssertionError("every nuisance recipient must receive replacement text")
        if audit["aggregate_label_count_delta"] != {}:
            raise AssertionError("nuisance must preserve aggregate label counts")
        if audit["duplicate_model_content_occurrences_beyond_first"] != 2 * PER_DIRECTION:
            raise AssertionError("nuisance reweighting duplicate count is unexpected")

        restored = restore_release_slots(
            candidate,
            baseline,
            restore_slot_ids=tuple(sorted(changed)),
        )
        if restored != baseline:
            raise AssertionError("nuisance restoration does not recover the exact baseline")

        payload: dict[str, object] = {
            "pair_rank_evidence": asdict(pair),
            "construction": audit,
            "restoration_exact_baseline": True,
        }
        nuisance_payloads.append(payload)
        nuisance_releases.append(candidate)

    selected_labels = [label for pair in selected for label in (pair.label_a, pair.label_b)]
    if len(selected_labels) != len(set(selected_labels)):
        raise AssertionError("selected nuisance intent pairs are not disjoint")

    composite = combine_disjoint_release_changes(baseline, root, *nuisance_releases)
    composite_changed = changed_slot_ids(baseline, composite)
    if len(composite_changed) != 5 * 2 * PER_DIRECTION:
        raise AssertionError("five-change composite has an unexpected changed-slot count")
    if _label_count_delta(baseline, composite):
        raise AssertionError("five-change composite must preserve aggregate label counts")

    restored_composite = restore_release_slots(
        composite,
        baseline,
        restore_slot_ids=composite_changed,
    )
    if restored_composite != baseline:
        raise AssertionError("full composite restoration does not recover the exact baseline")

    aggregate: dict[str, object] = {
        "development_partition_sha256": partition_sha256,
        "baseline_release_sha256": release_sha256(baseline),
        "pilot_target_pair": [TARGET_A, TARGET_B],
        "per_direction": PER_DIRECTION,
        "eligible_pair_count": len(ranked),
        "selected_nuisance_pairs": [asdict(pair) for pair in selected],
        "root": {
            "changed_slot_count": root_manifest["changed_slot_count"],
            "changed_slot_ids_sha256": root_manifest["changed_slot_ids_sha256"],
            "candidate_release_sha256": root_manifest["candidate_release_sha256"],
            "label_transitions": root_manifest["label_transitions"],
            "text_change_count": root_text_changes,
            "aggregate_label_count_delta": _label_count_delta(baseline, root),
            **duplicate_model_content_profile(root),
        },
        "nuisances": nuisance_payloads,
        "cross_nuisance_intent_disjoint": True,
        "cross_nuisance_changed_slot_disjoint": True,
        "root_nuisance_changed_slot_disjoint": True,
        "composite": {
            "candidate_release_sha256": release_sha256(composite),
            "changed_slot_count": len(composite_changed),
            "aggregate_label_count_delta": {},
            "restoration_exact_baseline": True,
            **duplicate_model_content_profile(composite),
        },
        "model_training_performed": False,
        "official_test_split_loaded": False,
    }
    _write_absent_or_identical(args.output_root / "summary.json", aggregate)

    print("===== EXP009 PILOT NUISANCE AUDIT =====")
    print(f"development_partition_sha256={partition_sha256}")
    print(f"baseline_release_sha256={release_sha256(baseline)}")
    print(f"eligible_pair_count={len(ranked)}")
    print(f"root_changed_slots={len(root_changed)}")
    print(f"root_text_change_count={root_text_changes}")
    for nuisance_index, payload in enumerate(nuisance_payloads, start=1):
        pair = payload["pair_rank_evidence"]
        construction = payload["construction"]
        print(
            f"nuisance_{nuisance_index}="
            f"{pair['label_a']}<->{pair['label_b']} "
            f"mean_bidirectional_confusion_rate="
            f"{pair['mean_bidirectional_confusion_rate']:.6f} "
            f"mutual_confusion_count={pair['mutual_confusion_count']} "
            f"lexical_jaccard={pair['lexical_jaccard']:.6f} "
            f"changed_slots={construction['changed_slot_count']} "
            f"text_changes={construction['text_change_count']} "
            f"duplicates_added="
            f"{construction['duplicate_model_content_occurrences_beyond_first']}"
        )
    print("cross_nuisance_intent_disjoint=YES")
    print("cross_nuisance_changed_slot_disjoint=YES")
    print("root_nuisance_changed_slot_disjoint=YES")
    print(f"composite_changed_slots={len(composite_changed)}")
    print(f"composite_release_sha256={release_sha256(composite)}")
    print("composite_restoration_exact_baseline=YES")
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")
    print(f"artifact={args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
