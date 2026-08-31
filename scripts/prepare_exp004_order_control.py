from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.task import (
    EXP004_SHARD_IDS,
    build_exp004_data,
    build_exp004_intervention_train,
    sft_examples_sha256,
    write_sft_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp004.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/exp004/postmortem/order_control/prepared",
    )
    return parser.parse_args()


def _order_key(example_id: str, seed: int) -> str:
    payload = f"exp004-order-control-a|{seed}|{example_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _permute(examples, seed: int):
    return tuple(
        sorted(
            examples,
            key=lambda example: _order_key(example.example_id, seed),
        )
    )


def _id_order_sha256(examples) -> str:
    payload = "\n".join(example.example_id for example in examples)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    output = Path(args.output)
    datasets = output / "datasets"

    data = build_exp004_data(seed=config.seed)

    variants = {
        "baseline_permuted_train": data.baseline_train,
        "candidate_permuted_train": data.candidate_train,
    }

    for shard_id in EXP004_SHARD_IDS:
        variants[f"restore_{shard_id}_permuted_train"] = build_exp004_intervention_train(
            shard_id, seed=config.seed
        )

    permuted = {name: _permute(examples, config.seed) for name, examples in variants.items()}

    orders = {
        name: tuple(example.example_id for example in examples)
        for name, examples in permuted.items()
    }
    first_order = next(iter(orders.values()))
    original_order = tuple(example.example_id for example in data.baseline_train)

    assert all(order == first_order for order in orders.values())
    assert set(first_order) == set(original_order)
    assert first_order != original_order

    for name, examples in permuted.items():
        write_sft_jsonl(examples, datasets / f"{name}.jsonl")

    write_sft_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_sft_jsonl(data.control_eval, datasets / "control_eval.jsonl")
    write_sft_jsonl(data.all_eval, datasets / "all_eval.jsonl")
    for slice_id, examples in data.eval_by_slice.items():
        write_sft_jsonl(examples, datasets / f"{slice_id}_eval.jsonl")

    summary = {
        "experiment_id": config.experiment_id,
        "analysis_type": "post_hoc_exploratory_order_control",
        "permutation_id": "order_control_a",
        "seed": config.seed,
        "training_records": len(first_order),
        "same_example_membership": True,
        "same_permutation_across_variants": True,
        "different_from_original_order": True,
        "permuted_id_order_sha256": _id_order_sha256(permuted["baseline_permuted_train"]),
        "variants": {},
    }

    for name, examples in permuted.items():
        counts = Counter(example.response for example in examples)
        summary["variants"][name] = {
            "records": len(examples),
            "label_counts": dict(sorted(counts.items())),
            "sft_sha256": sft_examples_sha256(examples),
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
