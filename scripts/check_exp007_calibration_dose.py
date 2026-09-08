from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.exp007 import EXP007_TARGET_DOSES


def _load(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate two frozen Experiment 007 calibration-world gates."
    )
    parser.add_argument("--config", default="configs/exp007.yaml")
    parser.add_argument("--target-dose", type=int, choices=EXP007_TARGET_DOSES, required=True)
    parser.add_argument("--world-gate", action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if config.experiment_id != "exp007" or config.calibration is None:
        raise ValueError("calibration aggregation requires Experiment 007 config")

    if len(args.world_gate) != config.calibration.calibration_world_count:
        raise ValueError("exactly two frozen calibration-world gates are required")

    results: dict[str, bool] = {}
    indices: set[int] = set()

    for path in args.world_gate:
        payload = _load(path)

        if payload.get("experiment_id") != "exp007":
            raise ValueError("calibration gate is not from Experiment 007")
        if payload.get("gate") != "candidate_localized_regression":
            raise ValueError("unexpected calibration gate type")
        if payload.get("phase") != "calibration":
            raise ValueError("received non-calibration gate")
        if payload.get("target_dose") != args.target_dose:
            raise ValueError("calibration gate target dose mismatch")

        index = payload.get("world_index")
        result = payload.get("result")

        if not isinstance(index, int):
            raise ValueError("invalid calibration world index")
        if not isinstance(result, dict) or not isinstance(result.get("all_passed"), bool):
            raise ValueError("invalid calibration gate result")

        indices.add(index)
        results[str(index)] = result["all_passed"]

    expected = set(range(config.calibration.calibration_world_count))
    if indices != expected:
        raise ValueError("calibration gates do not cover frozen world indices")

    passing = sum(results.values())
    qualified = passing >= config.calibration.minimum_passing_worlds

    payload = {
        "experiment_id": "exp007",
        "gate": "calibration_target_dose",
        "target_dose": args.target_dose,
        "world_results": dict(sorted(results.items())),
        "passing_worlds": passing,
        "required_passing_worlds": config.calibration.minimum_passing_worlds,
        "qualified": qualified,
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    print(rendered, end="")


if __name__ == "__main__":
    main()
