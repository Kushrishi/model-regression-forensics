from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.exp009_classifier import Exp009ClassifierPilotConfig
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_release import (
    build_clean_release_slots,
    build_symmetric_label_swap_candidate,
)
from model_forensics.exp009_release_training import (
    train_versioned_classifier_pilot,
    versioned_release_preflight,
)

TARGET_A = "Refund_not_showing_up"
TARGET_B = "request_refund"
DOSES = {
    "1_8": 16,
    "1_4": 33,
    "3_8": 49,
    "1_2": 65,
}


def _config_from_args(args: argparse.Namespace) -> Exp009ClassifierPilotConfig:
    return Exp009ClassifierPilotConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
    )


def _clean_reference_path(trajectory_id: int, *, epochs: int) -> Path:
    run_id = f"clean_t{trajectory_id:04d}_e{epochs}_bs32_lr2e5"
    return Path("artifacts/exp009/classifier_pilot/runs") / run_id / "train_summary.json"


def _load_clean_reference(trajectory_id: int, *, epochs: int) -> dict[str, object] | None:
    path = _clean_reference_path(trajectory_id, epochs=epochs)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Exp009 development candidate from a versioned Banking77 release"
    )
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--dose", choices=tuple(DOSES), required=True)
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/release_pilot/runs"),
    )
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    if args.trajectory_id < 0:
        raise ValueError("--trajectory-id must be non-negative")

    config = _config_from_args(args)
    config.validate()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    baseline = build_clean_release_slots(partition.development_train)
    candidate = build_symmetric_label_swap_candidate(
        baseline,
        label_a=TARGET_A,
        label_b=TARGET_B,
        per_direction=DOSES[args.dose],
    )

    preflight = versioned_release_preflight(
        partition,
        release_slots=candidate,
        trajectory_id=args.trajectory_id,
        config=config,
        target_labels=(TARGET_A, TARGET_B),
    )
    clean_reference = _load_clean_reference(args.trajectory_id, epochs=config.epochs)
    paired_schedule_match: bool | None = None
    if clean_reference is not None:
        paired_schedule_match = (
            clean_reference["slot_schedule_sha256"] == preflight["slot_schedule_sha256"]
        )
        if not paired_schedule_match:
            raise RuntimeError("candidate slot schedule does not match paired clean trajectory")

    if args.mode == "preflight":
        release = preflight["release"]
        diff = release["diff"]
        print("===== EXP009 VERSIONED CANDIDATE PREFLIGHT =====")
        print(f"dose={args.dose}")
        print(f"per_direction={DOSES[args.dose]}")
        print(f"trajectory_id={args.trajectory_id}")
        print(f"development_partition_sha256={preflight['development_partition_sha256']}")
        print(f"baseline_release_sha256={release['baseline_release_sha256']}")
        print(f"candidate_release_sha256={release['release_sha256']}")
        print(f"changed_slot_count={diff['changed_slot_count']}")
        print(f"changed_slot_ids_sha256={diff['changed_slot_ids_sha256']}")
        print(f"slot_schedule_sha256={preflight['slot_schedule_sha256']}")
        print(f"paired_clean_reference_found={clean_reference is not None}")
        print(f"paired_clean_schedule_match={paired_schedule_match}")
        print("training_performed=NO")
        print("official_test_split_loaded=NO")
        return

    if args.run_id is None:
        raise ValueError("--run-id is required with --mode train")

    summary = train_versioned_classifier_pilot(
        partition,
        release_slots=candidate,
        trajectory_id=args.trajectory_id,
        config=config,
        target_labels=(TARGET_A, TARGET_B),
        run_id=args.run_id,
        output_root=args.output_root,
    )

    paired_initial_state_match: bool | None = None
    if clean_reference is not None:
        paired_initial_state_match = (
            clean_reference["model"]["initial_model_state_sha256"]
            == summary["model"]["initial_model_state_sha256"]
        )
        if not paired_initial_state_match:
            raise RuntimeError(
                "candidate initial model state does not match paired clean trajectory"
            )

    metrics = summary["development_eval_metrics"]
    slices = summary["behavior_slice_metrics"]
    optimization = summary["optimization"]
    release = summary["release"]
    diff = release["diff"]

    print("===== EXP009 VERSIONED CANDIDATE PILOT =====")
    print(f"run_id={summary['run_id']}")
    print(f"dose={args.dose}")
    print(f"per_direction={DOSES[args.dose]}")
    print(f"trajectory_id={summary['trajectory_id']}")
    print(f"development_partition_sha256={summary['development_partition_sha256']}")
    print(f"candidate_release_sha256={release['release_sha256']}")
    print(f"changed_slot_count={diff['changed_slot_count']}")
    print(f"slot_schedule_sha256={summary['slot_schedule_sha256']}")
    print(f"initial_model_state_sha256={summary['model']['initial_model_state_sha256']}")
    print(f"paired_clean_schedule_match={paired_schedule_match}")
    print(f"paired_clean_initial_state_match={paired_initial_state_match}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_recall={metrics['macro_recall']:.6f}")
    print(f"target_macro_recall={slices['target_macro_recall']:.6f}")
    print(f"target_A_recall={slices['target_per_label_recall'][TARGET_A]:.6f}")
    print(f"target_B_recall={slices['target_per_label_recall'][TARGET_B]:.6f}")
    print(f"protected_macro_recall={slices['protected_macro_recall']:.6f}")
    print(f"protected_worst_intent_recall={slices['protected_worst_intent_recall']:.6f}")
    print(f"elapsed_seconds={optimization['elapsed_seconds']:.3f}")
    print("confirmatory_result=NO")
    print("official_test_split_loaded=NO")
    print(f"summary={args.output_root / args.run_id / 'train_summary.json'}")


if __name__ == "__main__":
    main()
