from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.task import EXP006_SHARD_IDS
from model_forensics.training import train_lora_sft_run

RESTORATION_SPLITS = tuple(f"restoration_{candidate_id}_train" for candidate_id in EXP006_SHARD_IDS)
TRAIN_SPLITS = ("baseline_train", "candidate_train", *RESTORATION_SPLITS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Experiment 006 clean, candidate, or restoration LoRA SFT sibling."
    )
    parser.add_argument("--config", default="configs/exp006.yaml")
    parser.add_argument("--prepared", required=True)
    parser.add_argument(
        "--train-split",
        choices=TRAIN_SPLITS,
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if config.experiment_id != "exp006":
        raise ValueError("training requires Experiment 006 config")

    train_lora_sft_run(
        config=config,
        prepared=args.prepared,
        train_split=args.train_split,
        run_id=args.run_id,
        output_root=args.output_root,
        preparation_command=("scripts/prepare_exp006.py / scripts/prepare_exp006_certification.py"),
    )


if __name__ == "__main__":
    main()
