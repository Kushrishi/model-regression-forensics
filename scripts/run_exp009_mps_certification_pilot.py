from __future__ import annotations

import argparse
from pathlib import Path

from model_forensics.exp009_classifier import (
    Exp009ClassifierPilotConfig,
    select_device,
)
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_pilot_states import (
    STATES,
    TARGET_A,
    TARGET_B,
    build_pilot_state_bundle,
)
from model_forensics.exp009_release import build_clean_release_slots
from model_forensics.exp009_release_training import (
    train_versioned_classifier_pilot,
    versioned_release_preflight,
)


def _config() -> Exp009ClassifierPilotConfig:
    return Exp009ClassifierPilotConfig(
        epochs=7,
        batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.10,
        max_length=128,
        max_grad_norm=1.0,
    )


def _run_id(trajectory_id: int, state: str) -> str:
    return f"mps_v2_t{trajectory_id:04d}_{state}_e7_bs32_lr2e5"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen Exp009 MPS certification-pilot state"
    )
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--trajectory-id", type=int, required=True)
    parser.add_argument("--state", choices=STATES, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/mps_certification_pilot_v2/runs"),
    )
    args = parser.parse_args()

    if args.trajectory_id not in {0, 1, 2}:
        raise ValueError("development certification pilot trajectory must be 0, 1, or 2")

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    baseline = build_clean_release_slots(partition.development_train)
    states = build_pilot_state_bundle(baseline).state_map()
    release = states[args.state]
    config = _config()

    preflight = versioned_release_preflight(
        partition,
        release_slots=release,
        trajectory_id=args.trajectory_id,
        config=config,
        target_labels=(TARGET_A, TARGET_B),
    )

    if args.mode == "preflight":
        release_info = preflight["release"]
        diff = release_info["diff"]
        print("===== EXP009 MPS CERTIFICATION PILOT STATE PREFLIGHT =====")
        print(f"trajectory_id={args.trajectory_id}")
        print(f"state={args.state}")
        print(f"run_id={_run_id(args.trajectory_id, args.state)}")
        print(f"development_partition_sha256={preflight['development_partition_sha256']}")
        print(f"baseline_release_sha256={release_info['baseline_release_sha256']}")
        print(f"release_sha256={release_info['release_sha256']}")
        print(f"changed_slot_count={diff['changed_slot_count']}")
        print(f"changed_slot_ids_sha256={diff['changed_slot_ids_sha256']}")
        print(f"slot_schedule_sha256={preflight['slot_schedule_sha256']}")
        print("training_performed=NO")
        print("official_test_split_loaded=NO")
        return

    import torch

    device = select_device(torch)
    if device != "mps":
        raise RuntimeError(f"refusing result-bearing pilot training on non-MPS device: {device}")

    run_id = _run_id(args.trajectory_id, args.state)
    summary = train_versioned_classifier_pilot(
        partition,
        release_slots=release,
        trajectory_id=args.trajectory_id,
        config=config,
        target_labels=(TARGET_A, TARGET_B),
        run_id=run_id,
        output_root=args.output_root,
    )

    runtime = summary["runtime"]
    if runtime["device"] != "mps":
        raise AssertionError("result-bearing run did not use MPS")

    metrics = summary["development_eval_metrics"]
    slices = summary["behavior_slice_metrics"]
    release_info = summary["release"]
    diff = release_info["diff"]

    print("===== EXP009 MPS CERTIFICATION PILOT RUN =====")
    print(f"trajectory_id={summary['trajectory_id']}")
    print(f"state={args.state}")
    print(f"run_id={summary['run_id']}")
    print(f"device={runtime['device']}")
    print(f"release_sha256={release_info['release_sha256']}")
    print(f"changed_slot_count={diff['changed_slot_count']}")
    print(f"slot_schedule_sha256={summary['slot_schedule_sha256']}")
    print(f"initial_model_state_sha256={summary['model']['initial_model_state_sha256']}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_recall={metrics['macro_recall']:.6f}")
    print(f"target_macro_recall={slices['target_macro_recall']:.6f}")
    print(f"target_A_recall={slices['target_per_label_recall'][TARGET_A]:.6f}")
    print(f"target_B_recall={slices['target_per_label_recall'][TARGET_B]:.6f}")
    print(f"protected_macro_recall={slices['protected_macro_recall']:.6f}")
    print(f"protected_worst_intent_recall={slices['protected_worst_intent_recall']:.6f}")
    print(f"elapsed_seconds={summary['optimization']['elapsed_seconds']:.3f}")
    print("confirmatory_result=NO")
    print("official_test_split_loaded=NO")
    print(f"summary={args.output_root / run_id / 'train_summary.json'}")


if __name__ == "__main__":
    main()
