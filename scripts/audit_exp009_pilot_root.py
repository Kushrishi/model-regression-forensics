from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.exp009_classifier import validate_frozen_development_partition
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
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
DOSES = {
    "1_8": 16,
    "1_4": 33,
    "3_8": 49,
    "1_2": 65,
}


def _write_absent_or_identical(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite different audit artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Exp009 pilot symmetric root construction")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/pilot_root_audit"),
    )
    args = parser.parse_args()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha256 = validate_frozen_development_partition(partition)
    baseline = build_clean_release_slots(partition.development_train)
    baseline_sha256 = release_sha256(baseline)

    previous_changed: set[str] = set()
    dose_payloads: dict[str, dict[str, object]] = {}

    for dose_name, per_direction in DOSES.items():
        candidate = build_symmetric_label_swap_candidate(
            baseline,
            label_a=TARGET_A,
            label_b=TARGET_B,
            per_direction=per_direction,
        )
        changed = changed_slot_ids(baseline, candidate)
        changed_set = set(changed)
        if previous_changed and not previous_changed < changed_set:
            raise AssertionError(f"pilot dose {dose_name} is not a strict superset of the prior dose")
        previous_changed = changed_set

        restored = restore_release_slots(
            candidate,
            baseline,
            restore_slot_ids=changed,
        )
        if restored != baseline:
            raise AssertionError(f"pilot dose {dose_name} did not restore exactly to baseline")

        manifest = release_diff_manifest(baseline, candidate)
        expected_changed = 2 * per_direction
        if manifest["changed_slot_count"] != expected_changed:
            raise AssertionError(
                f"pilot dose {dose_name} changed {manifest['changed_slot_count']} slots; "
                f"expected {expected_changed}"
            )

        expected_transitions = {
            f"{TARGET_A}->{TARGET_B}": per_direction,
            f"{TARGET_B}->{TARGET_A}": per_direction,
        }
        if manifest["label_transitions"] != expected_transitions:
            raise AssertionError(f"pilot dose {dose_name} has unexpected label transitions")

        payload: dict[str, object] = {
            "development_partition_sha256": partition_sha256,
            "target_a": TARGET_A,
            "target_b": TARGET_B,
            "dose_name": dose_name,
            "per_direction": per_direction,
            "total_changed_slots": expected_changed,
            "baseline_release_sha256": baseline_sha256,
            "candidate_release_sha256": manifest["candidate_release_sha256"],
            "changed_slot_ids_sha256": manifest["changed_slot_ids_sha256"],
            "label_transitions": manifest["label_transitions"],
            "slot_count": manifest["slot_count"],
            "restoration_exact_baseline": True,
            "model_training_performed": False,
            "official_test_split_loaded": False,
        }
        dose_payloads[dose_name] = payload
        _write_absent_or_identical(args.output_root / f"dose_{dose_name}.json", payload)

    aggregate: dict[str, object] = {
        "development_partition_sha256": partition_sha256,
        "baseline_release_sha256": baseline_sha256,
        "slot_count": len(baseline),
        "target_a": TARGET_A,
        "target_b": TARGET_B,
        "dose_order": list(DOSES),
        "doses": dose_payloads,
        "dose_sets_strictly_nested": True,
        "all_restorations_exact_baseline": True,
        "dose_selected": False,
        "model_training_performed": False,
        "official_test_split_loaded": False,
    }
    _write_absent_or_identical(args.output_root / "summary.json", aggregate)

    print("===== EXP009 PILOT ROOT AUDIT =====")
    print(f"development_partition_sha256={partition_sha256}")
    print(f"baseline_release_sha256={baseline_sha256}")
    print(f"slot_count={len(baseline)}")
    print(f"target_A={TARGET_A}")
    print(f"target_B={TARGET_B}")
    for dose_name, payload in dose_payloads.items():
        print(
            f"dose={dose_name} "
            f"per_direction={payload['per_direction']} "
            f"total_changed={payload['total_changed_slots']} "
            f"candidate_release_sha256={payload['candidate_release_sha256']} "
            f"changed_slot_ids_sha256={payload['changed_slot_ids_sha256']}"
        )
    print("dose_sets_strictly_nested=YES")
    print("all_restorations_exact_baseline=YES")
    print("dose_selected=NO")
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")
    print(f"artifact={args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
