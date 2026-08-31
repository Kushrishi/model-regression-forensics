from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.inference import evaluate_lora_adapter_run

EVAL_SPLITS = {
    "circle_small": "circle_small_eval",
    "circle_large": "circle_large_eval",
    "square_small": "square_small_eval",
    "square_large": "square_large_eval",
    "triangle_small": "triangle_small_eval",
    "triangle_large": "triangle_large_eval",
    "all": "all_eval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one Experiment 005 explicit-policy adapter."
    )
    parser.add_argument("--config", default="configs/exp005.yaml")
    parser.add_argument(
        "--prepared",
        default="artifacts/exp005/private/attempt_00/prepared",
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/exp005/private/attempt_00/runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_lora_adapter_run(
        config=load_experiment_config(args.config),
        prepared=args.prepared,
        adapter=args.adapter,
        run_id=args.run_id,
        output_root=args.output_root,
        eval_splits=EVAL_SPLITS,
        preparation_command="scripts/prepare_exp005.py",
    )


if __name__ == "__main__":
    main()
