from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.exp008 import (
    EXP008_CHANGES_PER_SHARD,
    EXP008_FROZEN_MANIFEST_SHA256,
    EXP008_SHARD_IDS,
    build_exp008_data,
    build_exp008_restoration_train,
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

ORDER_NAMESPACE = "exp008-order-control-a"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(
    path: Path,
    records: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _records(
    examples: tuple[Exp003TaskExample, ...],
) -> list[dict[str, str]]:
    return [example.to_sft_record() for example in examples]


def _validate_candidate_gate(
    payload: dict[str, Any],
    *,
    world_index: int,
) -> None:
    if payload.get("experiment_id") != "exp008":
        raise ValueError("candidate gate is not Exp008")
    if payload.get("gate") != "candidate_localized_regression":
        raise ValueError("candidate gate type mismatch")
    if payload.get("world_index") != world_index:
        raise ValueError("candidate gate world mismatch")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("candidate gate lacks provenance")
    if provenance.get("manifest_sha256") != EXP008_FROZEN_MANIFEST_SHA256:
        raise ValueError("candidate gate manifest mismatch")

    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("candidate gate result malformed")
    if result.get("all_passed") is not True:
        raise ValueError("restoration preparation requires passed candidate gate")


def _order_control(
    examples: tuple[Exp003TaskExample, ...],
    *,
    seed: int,
) -> tuple[Exp003TaskExample, ...]:
    return tuple(
        sorted(
            examples,
            key=lambda example: hashlib.sha256(
                (f"{ORDER_NAMESPACE}|{seed}|{example.example_id}").encode()
            ).hexdigest(),
        )
    )


def _copy_eval(
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
        description=("Prepare gated Experiment 008 restoration and alternative-order datasets.")
    )
    parser.add_argument(
        "--config",
        default="configs/exp008.yaml",
    )
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--candidate-gate", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.experiment_id != "exp008":
        raise ValueError("certification preparation requires Exp008 config")

    candidate_gate = _read_json(Path(args.candidate_gate))
    _validate_candidate_gate(
        candidate_gate,
        world_index=args.world_index,
    )

    prepared = Path(args.prepared)
    prepared_summary = _read_json(prepared / "summary.json")

    if prepared_summary.get("manifest_sha256") != (EXP008_FROZEN_MANIFEST_SHA256):
        raise ValueError("prepared world manifest mismatch")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    data = build_exp008_data(
        seed=config.seed,
        world_index=args.world_index,
    )

    baseline_records = _records(data.baseline_train)
    candidate_records = _records(data.candidate_train)

    if baseline_records != _read_jsonl(prepared / "datasets" / "baseline_train.jsonl"):
        raise ValueError("generated baseline differs from frozen prepared world")

    if candidate_records != _read_jsonl(prepared / "datasets" / "candidate_train.jsonl"):
        raise ValueError("generated candidate differs from frozen prepared world")

    restorations = {
        candidate_id: build_exp008_restoration_train(
            candidate_id,
            seed=config.seed,
            world_index=args.world_index,
        )
        for candidate_id in EXP008_SHARD_IDS
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

    _copy_eval(prepared, primary)

    ordered_baseline = _order_control(
        data.baseline_train,
        seed=config.seed,
    )
    ordered_candidate = _order_control(
        data.candidate_train,
        seed=config.seed,
    )
    ordered_restorations = {
        candidate_id: _order_control(
            examples,
            seed=config.seed,
        )
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

    _copy_eval(prepared, order_control)

    baseline_ids = [example.example_id for example in ordered_baseline]
    candidate_ids = [example.example_id for example in ordered_candidate]

    if candidate_ids != baseline_ids:
        raise ValueError("order-control baseline/candidate order differs")

    if any(
        [example.example_id for example in examples] != baseline_ids
        for examples in ordered_restorations.values()
    ):
        raise ValueError("order-control restoration order differs")

    summary = {
        "experiment_id": "exp008",
        "world_index": args.world_index,
        "manifest_sha256": EXP008_FROZEN_MANIFEST_SHA256,
        "localized_regression_gate_required": True,
        "primary_restoration_count": len(EXP008_SHARD_IDS),
        "restored_records_per_candidate": (EXP008_CHANGES_PER_SHARD),
        "order_control_a": {
            "namespace": ORDER_NAMESPACE,
            "seed": config.seed,
            "example_count": len(baseline_ids),
        },
        "hashes": {
            "primary_baseline": file_sha256(primary / "datasets" / "baseline_train.jsonl"),
            "primary_candidate": file_sha256(primary / "datasets" / "candidate_train.jsonl"),
            "order_baseline": file_sha256(order_control / "datasets" / "baseline_train.jsonl"),
            "order_candidate": file_sha256(order_control / "datasets" / "candidate_train.jsonl"),
        },
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
