from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from model_forensics.certification import (
    evaluate_exp007_candidate_gate as evaluate_candidate_gate,
)
from model_forensics.config import load_experiment_config
from model_forensics.exp008 import (
    EXP008_FROZEN_MANIFEST_SHA256,
    EXP008_WORLD_COUNT,
    derive_exp008_world_seed,
)

EVAL_SPLITS = (
    "circle_small_eval",
    "circle_large_eval",
    "square_small_eval",
    "square_large_eval",
    "triangle_small_eval",
    "triangle_large_eval",
    "all_eval",
)

_RUNTIME_COMPARISON_KEYS = (
    "device",
    "platform_system",
    "platform_machine",
    "torch_num_threads",
    "torch_num_interop_threads",
    "torch",
    "transformers",
    "peft",
)


def _load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(
    payload: dict[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing {key}")
    return value


def _validate_prepared(
    prepared: Path,
    *,
    world_index: int,
) -> dict[str, Any]:
    summary = _load(prepared / "summary.json")

    if summary.get("experiment_id") != "exp008":
        raise ValueError("prepared world is not Experiment 008")
    if summary.get("world_index") != world_index:
        raise ValueError("prepared world index mismatch")
    if summary.get("manifest_sha256") != EXP008_FROZEN_MANIFEST_SHA256:
        raise ValueError("prepared world manifest mismatch")

    construction = _mapping(
        summary,
        "construction_gates",
        label="prepared summary",
    )
    if construction.get("all_passed") is not True:
        raise ValueError("prepared construction gates did not pass")

    return summary


def _validate_training(
    summary: dict[str, Any],
    *,
    label: str,
    expected_split: str,
    expected_file: Path,
) -> str:
    if summary.get("experiment_id") != "exp008":
        raise ValueError(f"{label} training summary is not Exp008")
    if summary.get("run_kind") != "lora_sft":
        raise ValueError(f"{label} training run kind mismatch")
    if summary.get("train_split") != expected_split:
        raise ValueError(f"{label} training split mismatch")

    prepared_input = _mapping(
        summary,
        "prepared_input",
        label=f"{label} training summary",
    )

    if prepared_input.get("file_sha256") != _sha256(expected_file):
        raise ValueError(f"{label} adapter was not trained on this prepared dataset")

    adapter_path = summary.get("adapter_path")
    if not isinstance(adapter_path, str) or not adapter_path:
        raise ValueError(f"{label} training summary lacks adapter path")

    return str(Path(adapter_path).resolve())


def _runtime_signature(
    summary: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    runtime = _mapping(
        summary,
        "runtime",
        label=f"{label} training summary",
    )

    missing = [key for key in _RUNTIME_COMPARISON_KEYS if key not in runtime]
    if missing:
        raise ValueError(f"{label} runtime provenance lacks fields: {missing}")

    return {key: runtime[key] for key in _RUNTIME_COMPARISON_KEYS}


def _validate_eval(
    summary: dict[str, Any],
    *,
    label: str,
    expected_adapter: str,
    prepared: Path,
) -> None:
    if summary.get("experiment_id") != "exp008":
        raise ValueError(f"{label} evaluation is not Exp008")
    if summary.get("run_kind") != "adapter_eval":
        raise ValueError(f"{label} evaluation run kind mismatch")

    adapter = summary.get("adapter")
    if not isinstance(adapter, str):
        raise ValueError(f"{label} evaluation lacks adapter")

    if str(Path(adapter).resolve()) != expected_adapter:
        raise ValueError(f"{label} evaluation does not match trained adapter")

    prepared_inputs = _mapping(
        summary,
        "prepared_inputs",
        label=f"{label} evaluation summary",
    )

    expected = {
        f"{split}_file_sha256": _sha256(prepared / "datasets" / f"{split}.jsonl")
        for split in EVAL_SPLITS
    }

    if prepared_inputs != expected:
        raise ValueError(f"{label} evaluation inputs differ from frozen world")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check one Experiment 008 localized-regression gate."
    )
    parser.add_argument(
        "--config",
        default="configs/exp008.yaml",
    )
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument(
        "--baseline-train-summary",
        required=True,
    )
    parser.add_argument(
        "--candidate-train-summary",
        required=True,
    )
    parser.add_argument(
        "--baseline-summary",
        required=True,
    )
    parser.add_argument(
        "--candidate-summary",
        required=True,
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if not 0 <= args.world_index < EXP008_WORLD_COUNT:
        raise ValueError("world index outside frozen Exp008 range")

    config = load_experiment_config(args.config)
    if config.experiment_id != "exp008":
        raise ValueError("candidate gate requires Exp008 config")

    derive_exp008_world_seed(
        config.seed,
        args.world_index,
    )

    prepared = Path(args.prepared)
    prepared_summary = _validate_prepared(
        prepared,
        world_index=args.world_index,
    )

    baseline_train = _load(args.baseline_train_summary)
    candidate_train = _load(args.candidate_train_summary)

    baseline_adapter = _validate_training(
        baseline_train,
        label="baseline",
        expected_split="baseline_train",
        expected_file=(prepared / "datasets" / "baseline_train.jsonl"),
    )
    candidate_adapter = _validate_training(
        candidate_train,
        label="candidate",
        expected_split="candidate_train",
        expected_file=(prepared / "datasets" / "candidate_train.jsonl"),
    )

    baseline_runtime = _runtime_signature(
        baseline_train,
        label="baseline",
    )
    candidate_runtime = _runtime_signature(
        candidate_train,
        label="candidate",
    )

    if baseline_runtime != candidate_runtime:
        raise ValueError("baseline/candidate execution backends are not comparable")

    baseline_eval = _load(args.baseline_summary)
    candidate_eval = _load(args.candidate_summary)

    _validate_eval(
        baseline_eval,
        label="baseline",
        expected_adapter=baseline_adapter,
        prepared=prepared,
    )
    _validate_eval(
        candidate_eval,
        label="candidate",
        expected_adapter=candidate_adapter,
        prepared=prepared,
    )

    result = evaluate_candidate_gate(
        baseline_summary=baseline_eval,
        candidate_summary=candidate_eval,
        minimum_baseline_score=(config.evaluation.minimum_baseline_score),
        minimum_regression_delta=(config.evaluation.minimum_regression_delta),
        maximum_unrelated_delta=(config.evaluation.maximum_unrelated_delta),
    )

    payload = {
        "experiment_id": "exp008",
        "gate": "candidate_localized_regression",
        "world_index": args.world_index,
        "thresholds": {
            "minimum_baseline_score": (config.evaluation.minimum_baseline_score),
            "minimum_regression_delta": (config.evaluation.minimum_regression_delta),
            "maximum_unrelated_delta": (config.evaluation.maximum_unrelated_delta),
        },
        "provenance": {
            "manifest_sha256": EXP008_FROZEN_MANIFEST_SHA256,
            "prepared_summary_sha256": _sha256(prepared / "summary.json"),
            "baseline_train_sha256": _sha256(prepared / "datasets" / "baseline_train.jsonl"),
            "candidate_train_sha256": _sha256(prepared / "datasets" / "candidate_train.jsonl"),
            "runtime": baseline_runtime,
            "prepared_dataset_sft_sha256": prepared_summary["dataset_sft_sha256"],
        },
        "result": result.to_dict(),
    }

    rendered = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    print(rendered, end="")


if __name__ == "__main__":
    main()
