from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.task import EXP005_SHARD_IDS
from model_forensics.training import train_lora_sft_run

RESTORATION_SPLITS = tuple(f"restoration_{candidate_id}_train" for candidate_id in EXP005_SHARD_IDS)
TRAIN_SPLITS = ("baseline_train", "candidate_train", *RESTORATION_SPLITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Experiment 005 clean, candidate, or restoration LoRA SFT sibling."
    )
    parser.add_argument("--config", default="configs/exp005.yaml")
    parser.add_argument(
        "--prepared",
        default="artifacts/exp005/private/attempt_00/prepared",
    )
    parser.add_argument(
        "--train-split",
        choices=TRAIN_SPLITS,
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/exp005/private/attempt_00/checkpoints",
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
        preparation_command="scripts/prepare_exp005.py / scripts/prepare_exp005_certification.py",
    )


if __name__ == "__main__":
    main()
