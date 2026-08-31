from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.inference import file_sha256
from model_forensics.task import (
    EXP005_SHARD_IDS,
    Exp003TaskExample,
    build_exp005_data,
    build_exp005_restoration_train,
)

EVAL_DATASETS = (
    "circle_small_eval",
    "circle_large_eval",
    "square_small_eval",
    "square_large_eval",
    "triangle_small_eval",
    "triangle_large_eval",
    "all_eval",
)
ORDER_NAMESPACE = "exp005-order-control-a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare private Experiment 005 restoration and order-control datasets."
    )
    parser.add_argument("--config", default="configs/exp005.yaml")
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _records(examples: tuple[Exp003TaskExample, ...]) -> list[dict[str, str]]:
    return [example.to_sft_record() for example in examples]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


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


def _copy_eval_datasets(source: Path, destination: Path) -> None:
    for split in EVAL_DATASETS:
        source_path = source / "datasets" / f"{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"missing frozen evaluation dataset: {source_path}")
        target = destination / "datasets" / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if config.experiment_id != "exp005":
        raise ValueError("certification preparation requires Experiment 005 config")

    prepared = Path(args.prepared)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty certification output: {output}")

    data = build_exp005_data(config.seed, args.attempt_index)
    baseline_records = _records(data.baseline_train)
    candidate_records = _records(data.candidate_train)

    frozen_baseline = _read_jsonl(prepared / "datasets" / "baseline_train.jsonl")
    frozen_candidate = _read_jsonl(prepared / "datasets" / "candidate_train.jsonl")
    if baseline_records != frozen_baseline:
        raise ValueError("generated baseline does not match the already-frozen prepared world")
    if candidate_records != frozen_candidate:
        raise ValueError("generated candidate does not match the already-frozen prepared world")

    restorations = {
        candidate_id: build_exp005_restoration_train(
            candidate_id,
            seed=config.seed,
            attempt_index=args.attempt_index,
        )
        for candidate_id in EXP005_SHARD_IDS
    }

    primary = output / "primary"
    order_control = output / "order_control_a"

    _write_jsonl(primary / "datasets" / "baseline_train.jsonl", baseline_records)
    _write_jsonl(primary / "datasets" / "candidate_train.jsonl", candidate_records)
    for candidate_id, examples in restorations.items():
        _write_jsonl(
            primary / "datasets" / f"restoration_{candidate_id}_train.jsonl",
            _records(examples),
        )
    _copy_eval_datasets(prepared, primary)

    ordered_baseline = _order_control(data.baseline_train, seed=config.seed)
    ordered_candidate = _order_control(data.candidate_train, seed=config.seed)
    _write_jsonl(
        order_control / "datasets" / "baseline_train.jsonl",
        _records(ordered_baseline),
    )
    _write_jsonl(
        order_control / "datasets" / "candidate_train.jsonl",
        _records(ordered_candidate),
    )

    ordered_restorations: dict[str, tuple[Exp003TaskExample, ...]] = {}
    for candidate_id, examples in restorations.items():
        ordered = _order_control(examples, seed=config.seed)
        ordered_restorations[candidate_id] = ordered
        _write_jsonl(
            order_control / "datasets" / f"restoration_{candidate_id}_train.jsonl",
            _records(ordered),
        )
    _copy_eval_datasets(prepared, order_control)

    baseline_ids = [example.example_id for example in ordered_baseline]
    candidate_ids = [example.example_id for example in ordered_candidate]
    if baseline_ids != candidate_ids:
        raise ValueError("order-control clean/candidate example order differs")
    if any(
        [example.example_id for example in ordered_restorations[candidate_id]] != baseline_ids
        for candidate_id in EXP005_SHARD_IDS
    ):
        raise ValueError("order-control restoration example order differs")

    summary = {
        "experiment_id": "exp005",
        "primary_restoration_count": len(EXP005_SHARD_IDS),
        "restored_records_per_candidate": 24,
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
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
