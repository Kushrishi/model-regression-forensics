from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.exp009_data import build_development_partition, load_banking77_train
from model_forensics.exp009_trajectory import (
    build_stable_training_slots,
    trajectory_manifest,
)


def _write_if_absent_or_identical(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(
                f"refusing to overwrite different trajectory audit artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Exp009 stable slots and trajectory schedules"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/trajectory_audit"),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Audit-only epoch count; this does not freeze the later training configuration.",
    )
    parser.add_argument(
        "--trajectory-ids",
        type=int,
        nargs="+",
        default=[0, 1],
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if len(args.trajectory_ids) != len(set(args.trajectory_ids)):
        raise ValueError("--trajectory-ids must be unique")

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    slots = build_stable_training_slots(partition.development_train)

    manifests = [
        trajectory_manifest(
            slots,
            trajectory_id=trajectory_id,
            epochs=args.epochs,
        )
        for trajectory_id in args.trajectory_ids
    ]

    for manifest in manifests:
        trajectory_id = int(manifest["trajectory_id"])
        content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        _write_if_absent_or_identical(
            args.output_root / f"trajectory_{trajectory_id:04d}.json",
            content,
        )

    slot_manifest = "".join(
        json.dumps(
            {
                "slot_id": slot.slot_id,
                "content_id": slot.content_id,
                "label": slot.label,
            },
            sort_keys=True,
        )
        + "\n"
        for slot in slots
    )
    _write_if_absent_or_identical(args.output_root / "baseline_slots.jsonl", slot_manifest)

    print("===== EXP009 TRAJECTORY AUDIT =====")
    print(f"development_train_records={len(partition.development_train)}")
    print(f"stable_slots={len(slots)}")
    print(f"audit_epochs={args.epochs}")
    print("training_configuration_frozen=NO")
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")
    for manifest in manifests:
        print(
            f"trajectory_{int(manifest['trajectory_id']):04d}_schedule_sha256="
            f"{manifest['slot_schedule_sha256']}"
        )
    print(f"output_root={args.output_root}")


if __name__ == "__main__":
    main()
