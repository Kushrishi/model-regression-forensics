from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.exp007 import (
    EXP007_CHANGES_PER_SHARD,
    EXP007_FROZEN_MANIFEST_SHA256,
    EXP007_SHARD_IDS,
    build_exp007_data,
    build_exp007_restoration_train,
)
from model_forensics.inference import file_sha256
from model_forensics.task import Exp003TaskExample

EVAL_DATASETS = (
    "circle_small_eval",
    "circle_large_eval",
    "square_small_eval",
    "square_large_eval",
    "triangle_small_eval",
    "triangle_large_eval",
    "all_eval",
)
ORDER_NAMESPACE = "exp007-order-control-a"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _records(
    examples: tuple[Exp003TaskExample, ...],
) -> list[dict[str, str]]:
    return [example.to_sft_record() for example in examples]


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_selection(
    payload: dict[str, Any],
    *,
    config_path: Path,
    target_dose: int,
) -> None:
    if payload.get("experiment_id") != "exp007":
        raise ValueError("selection artifact is not from Experiment 007")
    if payload.get("selection_rule") != "minimum_target_dose_meeting_gate":
        raise ValueError("selection artifact uses the wrong selection rule")
    if payload.get("certification_authorized") is not True:
        raise ValueError("selection artifact does not authorize certification")
    if payload.get("selected_target_dose") != target_dose:
        raise ValueError("certification target dose differs from frozen selection")
    if payload.get("manifest_sha256") != EXP007_FROZEN_MANIFEST_SHA256:
        raise ValueError("selection artifact manifest hash mismatch")
    if payload.get("config_sha256") != _config_sha256(config_path):
        raise ValueError("selection artifact config hash mismatch")


def _validate_candidate_gate(
    payload: dict[str, Any],
    *,
    target_dose: int,
    world_index: int,
) -> None:
    if payload.get("experiment_id") != "exp007":
        raise ValueError("candidate gate is not from Experiment 007")
    if payload.get("gate") != "candidate_localized_regression":
        raise ValueError("unexpected candidate gate type")
    if payload.get("phase") != "certification":
        raise ValueError("restoration requires a certification candidate gate")
    if payload.get("target_dose") != target_dose:
        raise ValueError("candidate gate target dose mismatch")
    if payload.get("world_index") != world_index:
        raise ValueError("candidate gate world index mismatch")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("candidate gate lacks world provenance")
    if provenance.get("manifest_sha256") != EXP007_FROZEN_MANIFEST_SHA256:
        raise ValueError("candidate gate manifest provenance mismatch")

    result = payload.get("result")
    if not isinstance(result, dict) or result.get("all_passed") is not True:
        raise ValueError("restoration preparation requires passed localized-regression gate")


def _order_control(
    examples: tuple[Exp003TaskExample, ...],
    *,
    seed: int,
) -> tuple[Exp003TaskExample, ...]:
    return tuple(
        sorted(
            examples,
            key=lambda example: hashlib.sha256(
                f"{ORDER_NAMESPACE}|{seed}|{example.example_id}".encode()
            ).hexdigest(),
        )
    )


def _copy_eval_datasets(
    source: Path,
    destination: Path,
) -> None:
    for split in EVAL_DATASETS:
        source_path = source / "datasets" / f"{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"missing frozen evaluation dataset: {source_path}")

        target = destination / "datasets" / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Prepare gated Experiment 007 restoration and order-control datasets.")
    )
    parser.add_argument("--config", default="configs/exp007.yaml")
    parser.add_argument("--target-dose", type=int, required=True)
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate-gate", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    if config.experiment_id != "exp007":
        raise ValueError("certification preparation requires Experiment 007 config")

    selection = _read_json(Path(args.selection))
    _validate_selection(
        selection,
        config_path=config_path,
        target_dose=args.target_dose,
    )

    candidate_gate = _read_json(Path(args.candidate_gate))
    _validate_candidate_gate(
        candidate_gate,
        target_dose=args.target_dose,
        world_index=args.world_index,
    )

    prepared = Path(args.prepared)
    output = Path(args.output)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    data = build_exp007_data(
        seed=config.seed,
        phase="certification",
        target_dose=args.target_dose,
        world_index=args.world_index,
    )

    baseline_records = _records(data.baseline_train)
    candidate_records = _records(data.candidate_train)

    if baseline_records != _read_jsonl(prepared / "datasets" / "baseline_train.jsonl"):
        raise ValueError("generated baseline does not match frozen certification world")

    if candidate_records != _read_jsonl(prepared / "datasets" / "candidate_train.jsonl"):
        raise ValueError("generated candidate does not match frozen certification world")

    restorations = {
        candidate_id: build_exp007_restoration_train(
            candidate_id,
            seed=config.seed,
            phase="certification",
            target_dose=args.target_dose,
            world_index=args.world_index,
        )
        for candidate_id in EXP007_SHARD_IDS
    }

    primary = output / "primary"
    order_control = output / "order_control_a"

    _write_jsonl(
        primary / "datasets" / "baseline_train.jsonl",
        baseline_records,
    )
    _write_jsonl(
        primary / "datasets" / "candidate_train.jsonl",
        candidate_records,
    )

    for candidate_id, examples in restorations.items():
        _write_jsonl(
            primary / "datasets" / f"restoration_{candidate_id}_train.jsonl",
            _records(examples),
        )

    _copy_eval_datasets(prepared, primary)

    ordered_baseline = _order_control(
        data.baseline_train,
        seed=config.seed,
    )
    ordered_candidate = _order_control(
        data.candidate_train,
        seed=config.seed,
    )
    ordered_restorations = {
        candidate_id: _order_control(examples, seed=config.seed)
        for candidate_id, examples in restorations.items()
    }

    _write_jsonl(
        order_control / "datasets" / "baseline_train.jsonl",
        _records(ordered_baseline),
    )
    _write_jsonl(
        order_control / "datasets" / "candidate_train.jsonl",
        _records(ordered_candidate),
    )

    for candidate_id, examples in ordered_restorations.items():
        _write_jsonl(
            order_control / "datasets" / f"restoration_{candidate_id}_train.jsonl",
            _records(examples),
        )

    _copy_eval_datasets(prepared, order_control)

    baseline_ids = [example.example_id for example in ordered_baseline]
    candidate_ids = [example.example_id for example in ordered_candidate]

    if candidate_ids != baseline_ids:
        raise ValueError("order-control clean/candidate example order differs")

    if any(
        [example.example_id for example in examples] != baseline_ids
        for examples in ordered_restorations.values()
    ):
        raise ValueError("order-control restoration example order differs")

    summary = {
        "experiment_id": "exp007",
        "phase": "certification",
        "target_dose": args.target_dose,
        "world_index": args.world_index,
        "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
        "primary_restoration_count": len(EXP007_SHARD_IDS),
        "restored_records_per_candidate": EXP007_CHANGES_PER_SHARD,
        "localized_regression_gate_required": True,
        "order_control_a": {
            "namespace": ORDER_NAMESPACE,
            "seed": config.seed,
            "example_count": len(baseline_ids),
            "identical_example_order_across_siblings": True,
        },
        "hashes": {
            "primary_baseline": file_sha256(primary / "datasets" / "baseline_train.jsonl"),
            "primary_candidate": file_sha256(primary / "datasets" / "candidate_train.jsonl"),
            "order_baseline": file_sha256(order_control / "datasets" / "baseline_train.jsonl"),
            "order_candidate": file_sha256(order_control / "datasets" / "candidate_train.jsonl"),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
