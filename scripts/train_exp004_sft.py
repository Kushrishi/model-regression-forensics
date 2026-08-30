from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.training import train_lora_sft_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Experiment 004 LoRA SFT sibling run.")
    parser.add_argument("--config", default="configs/exp004.yaml")
    parser.add_argument("--prepared", default="artifacts/exp004/prepared")
    parser.add_argument(
        "--train-split",
        choices=("baseline_train", "candidate_train", "intervention_train"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="artifacts/exp004/checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_lora_sft_run(
        config=load_experiment_config(args.config),
        prepared=args.prepared,
        train_split=args.train_split,
        run_id=args.run_id,
        output_root=args.output_root,
        preparation_command="scripts/prepare_exp004.py",
    )


if __name__ == "__main__":
    main()
