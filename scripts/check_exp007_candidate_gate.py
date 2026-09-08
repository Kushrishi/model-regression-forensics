from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from model_forensics.certification import evaluate_exp007_candidate_gate
from model_forensics.config import load_experiment_config
from model_forensics.exp007 import (
    EXP007_FROZEN_MANIFEST_SHA256,
    derive_exp007_world_seed,
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


def _load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_mapping(
    payload: dict[str, Any],
    key: str,
    *,
    label: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing {key}")
    return value


def _validate_prepared_world(
    prepared: Path,
    *,
    phase: str,
    target_dose: int,
    world_index: int,
) -> None:
    summary = _load(prepared / "summary.json")

    if summary.get("experiment_id") != "exp007":
        raise ValueError("prepared world is not from Experiment 007")
    if summary.get("phase") != phase:
        raise ValueError("prepared world phase mismatch")
    if summary.get("target_dose") != target_dose:
        raise ValueError("prepared world target dose mismatch")
    if summary.get("world_index") != world_index:
        raise ValueError("prepared world index mismatch")
    if summary.get("manifest_sha256") != EXP007_FROZEN_MANIFEST_SHA256:
        raise ValueError("prepared world manifest hash mismatch")

    construction = _require_mapping(
        summary,
        "construction_gates",
        label="prepared summary",
    )
    if construction.get("all_passed") is not True:
        raise ValueError("prepared world construction gates did not pass")


def _validate_training_summary(
    summary: dict[str, Any],
    *,
    label: str,
    expected_split: str,
    expected_file: Path,
) -> str:
    if summary.get("experiment_id") != "exp007":
        raise ValueError(f"{label} training summary is not from Experiment 007")
    if summary.get("run_kind") != "lora_sft":
        raise ValueError(f"{label} training summary has wrong run kind")
    if summary.get("train_split") != expected_split:
        raise ValueError(f"{label} training summary has wrong train split")

    prepared_input = _require_mapping(
        summary,
        "prepared_input",
        label=f"{label} training summary",
    )

    expected_sha = _file_sha256(expected_file)
    if prepared_input.get("file_sha256") != expected_sha:
        raise ValueError(f"{label} adapter was not trained on this prepared world")

    adapter_path = summary.get("adapter_path")
    if not isinstance(adapter_path, str) or not adapter_path:
        raise ValueError(f"{label} training summary lacks adapter path")

    return str(Path(adapter_path).resolve())


def _validate_evaluation_summary(
    summary: dict[str, Any],
    *,
    label: str,
    expected_adapter: str,
    prepared: Path,
) -> None:
    if summary.get("experiment_id") != "exp007":
        raise ValueError(f"{label} evaluation summary is not from Experiment 007")
    if summary.get("run_kind") != "adapter_eval":
        raise ValueError(f"{label} evaluation summary has wrong run kind")

    adapter = summary.get("adapter")
    if not isinstance(adapter, str):
        raise ValueError(f"{label} evaluation summary lacks adapter path")
    if str(Path(adapter).resolve()) != expected_adapter:
        raise ValueError(f"{label} evaluation does not use the matching trained adapter")

    prepared_inputs = _require_mapping(
        summary,
        "prepared_inputs",
        label=f"{label} evaluation summary",
    )

    expected = {
        f"{split_name}_file_sha256": _file_sha256(prepared / "datasets" / f"{split_name}.jsonl")
        for split_name in EVAL_SPLITS
    }

    if prepared_inputs != expected:
        raise ValueError(f"{label} evaluation inputs do not match this prepared world")


def _validate_provenance(
    *,
    prepared: Path,
    phase: str,
    target_dose: int,
    world_index: int,
    baseline_train_summary: dict[str, Any],
    candidate_train_summary: dict[str, Any],
    baseline_eval_summary: dict[str, Any],
    candidate_eval_summary: dict[str, Any],
) -> dict[str, Any]:
    _validate_prepared_world(
        prepared,
        phase=phase,
        target_dose=target_dose,
        world_index=world_index,
    )

    baseline_train_file = prepared / "datasets" / "baseline_train.jsonl"
    candidate_train_file = prepared / "datasets" / "candidate_train.jsonl"

    baseline_adapter = _validate_training_summary(
        baseline_train_summary,
        label="baseline",
        expected_split="baseline_train",
        expected_file=baseline_train_file,
    )
    candidate_adapter = _validate_training_summary(
        candidate_train_summary,
        label="candidate",
        expected_split="candidate_train",
        expected_file=candidate_train_file,
    )

    _validate_evaluation_summary(
        baseline_eval_summary,
        label="baseline",
        expected_adapter=baseline_adapter,
        prepared=prepared,
    )
    _validate_evaluation_summary(
        candidate_eval_summary,
        label="candidate",
        expected_adapter=candidate_adapter,
        prepared=prepared,
    )

    return {
        "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
        "prepared_summary_sha256": _file_sha256(prepared / "summary.json"),
        "baseline_train_sha256": _file_sha256(baseline_train_file),
        "candidate_train_sha256": _file_sha256(candidate_train_file),
        "evaluation_inputs": {
            split_name: _file_sha256(prepared / "datasets" / f"{split_name}.jsonl")
            for split_name in EVAL_SPLITS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check one frozen Experiment 007 localized-regression gate."
    )
    parser.add_argument("--config", default="configs/exp007.yaml")
    parser.add_argument("--phase", choices=("calibration", "certification"), required=True)
    parser.add_argument("--target-dose", type=int, required=True)
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--baseline-train-summary", required=True)
    parser.add_argument("--candidate-train-summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.experiment_id != "exp007":
        raise ValueError("candidate gate requires Experiment 007 config")

    derive_exp007_world_seed(
        config.seed,
        args.phase,
        args.target_dose,
        args.world_index,
    )

    baseline_eval = _load(args.baseline_summary)
    candidate_eval = _load(args.candidate_summary)

    provenance = _validate_provenance(
        prepared=Path(args.prepared),
        phase=args.phase,
        target_dose=args.target_dose,
        world_index=args.world_index,
        baseline_train_summary=_load(args.baseline_train_summary),
        candidate_train_summary=_load(args.candidate_train_summary),
        baseline_eval_summary=baseline_eval,
        candidate_eval_summary=candidate_eval,
    )

    result = evaluate_exp007_candidate_gate(
        baseline_summary=baseline_eval,
        candidate_summary=candidate_eval,
        minimum_baseline_score=config.evaluation.minimum_baseline_score,
        minimum_regression_delta=config.evaluation.minimum_regression_delta,
        maximum_unrelated_delta=config.evaluation.maximum_unrelated_delta,
    )

    payload = {
        "experiment_id": "exp007",
        "gate": "candidate_localized_regression",
        "phase": args.phase,
        "target_dose": args.target_dose,
        "world_index": args.world_index,
        "thresholds": {
            "minimum_baseline_score": config.evaluation.minimum_baseline_score,
            "minimum_regression_delta": config.evaluation.minimum_regression_delta,
            "maximum_unrelated_delta": config.evaluation.maximum_unrelated_delta,
        },
        "provenance": provenance,
        "result": result.to_dict(),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    print(rendered, end="")


if __name__ == "__main__":
    main()
