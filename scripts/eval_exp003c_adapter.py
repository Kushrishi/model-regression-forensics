from __future__ import annotations

import argparse

from model_forensics.config import load_experiment_config
from model_forensics.inference import evaluate_lora_adapter_run

EVAL_SPLITS = {
    "slot_a": "slot_a_eval",
    "slot_b": "slot_b_eval",
    "slot_c": "slot_c_eval",
    "slot_d": "slot_d_eval",
    "slot_e": "slot_e_eval",
    "slot_f": "slot_f_eval",
    "all": "all_eval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Experiment 003-C selected-slot lookup adapter."
    )
    parser.add_argument("--config", default="configs/exp003c.yaml")
    parser.add_argument("--prepared", default="artifacts/exp003c/prepared")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="artifacts/exp003c/runs")
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
        preparation_command="scripts/prepare_exp003c.py",
    )


if __name__ == "__main__":
    main()
