from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.task import EXP004_SHARD_IDS
from model_forensics.training import train_lora_sft_run

TRAIN_SPLITS = tuple(f"restore_{shard_id}_train" for shard_id in EXP004_SHARD_IDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train one post-hoc Experiment 004 single-shard restoration sibling.")
    )
    parser.add_argument("--config", default="configs/exp004.yaml")
    parser.add_argument(
        "--prepared",
        default="artifacts/exp004/postmortem/prepared",
    )
    parser.add_argument(
        "--train-split",
        choices=TRAIN_SPLITS,
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/exp004/postmortem/checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_lora_sft_run(
        config=load_experiment_config(args.config),
        prepared=args.prepared,
        train_split=args.train_split,
        run_id=args.run_id,
        output_root=args.output_root,
        preparation_command=("scripts/prepare_exp004_postmortem_sweep.py"),
    )


if __name__ == "__main__":
    main()
