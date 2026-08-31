from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.certification import evaluate_exp005_candidate_gate
from model_forensics.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the frozen Experiment 005 clean/candidate behavioral gates."
    )
    parser.add_argument("--config", default="configs/exp005.yaml")
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def _load_json(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    result = evaluate_exp005_candidate_gate(
        baseline_summary=_load_json(args.baseline_summary),
        candidate_summary=_load_json(args.candidate_summary),
        minimum_baseline_score=config.evaluation.minimum_baseline_score,
        minimum_regression_delta=config.evaluation.minimum_regression_delta,
        maximum_unrelated_delta=config.evaluation.maximum_unrelated_delta,
    )

    payload = {
        "experiment_id": config.experiment_id,
        "gate": "candidate_localized_regression",
        "thresholds": {
            "minimum_baseline_score": config.evaluation.minimum_baseline_score,
            "minimum_regression_delta": config.evaluation.minimum_regression_delta,
            "maximum_unrelated_delta": config.evaluation.maximum_unrelated_delta,
        },
        "result": result.to_dict(),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    print(rendered, end="")


if __name__ == "__main__":
    main()
