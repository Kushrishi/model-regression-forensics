from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.inference import evaluate_lora_adapter_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Experiment 002 LoRA adapter.")
    parser.add_argument("--config", default="configs/exp002.yaml")
    parser.add_argument("--prepared", default="artifacts/exp002/prepared")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="artifacts/exp002/runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_lora_adapter_run(
        config=load_experiment_config(args.config),
        prepared=args.prepared,
        adapter=args.adapter,
        run_id=args.run_id,
        output_root=args.output_root,
        eval_splits={"target": "target_eval", "control": "control_eval", "all": "all_eval"},
        preparation_command="scripts/prepare_exp002.py",
    )


if __name__ == "__main__":
    main()
