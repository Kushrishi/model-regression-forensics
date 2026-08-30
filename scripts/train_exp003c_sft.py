from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.training import train_lora_sft_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Experiment 003-C selected-slot lookup baseline."
    )
    parser.add_argument("--config", default="configs/exp003c.yaml")
    parser.add_argument("--prepared", default="artifacts/exp003c/prepared")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="artifacts/exp003c/checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_lora_sft_run(
        config=load_experiment_config(args.config),
        prepared=args.prepared,
        train_split="baseline_train",
        run_id=args.run_id,
        output_root=args.output_root,
        preparation_command="scripts/prepare_exp003c.py",
    )


if __name__ == "__main__":
    main()
