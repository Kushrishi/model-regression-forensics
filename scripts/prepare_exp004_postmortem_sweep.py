from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.task import (
    EXP004_LABEL_CHANGES_PER_SHARD,
    EXP004_SHARD_IDS,
    build_exp004_data,
    build_exp004_intervention_train,
    sft_examples_sha256,
    write_sft_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the post-hoc Experiment 004 single-shard restoration sweep. "
            "This is exploratory and does not modify the frozen Exp004 result."
        )
    )
    parser.add_argument("--config", default="configs/exp004.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/exp004/postmortem/prepared",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    output = Path(args.output)
    datasets = output / "datasets"

    data = build_exp004_data(seed=config.seed)

    # Recreate the exact frozen held-out evaluation inputs in the isolated
    # exploratory prepared namespace.
    write_sft_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_sft_jsonl(data.control_eval, datasets / "control_eval.jsonl")
    write_sft_jsonl(data.all_eval, datasets / "all_eval.jsonl")

    for slice_id, examples in data.eval_by_slice.items():
        write_sft_jsonl(
            examples,
            datasets / f"{slice_id}_eval.jsonl",
        )

    baseline = data.baseline_train
    candidate = data.candidate_train

    baseline_by_id = {example.example_id: example for example in baseline}
    candidate_by_id = {example.example_id: example for example in candidate}

    candidate_corruptions = sum(
        baseline_by_id[example.example_id].response != candidate_by_id[example.example_id].response
        for example in candidate
    )

    expected_candidate_corruptions = len(EXP004_SHARD_IDS) * EXP004_LABEL_CHANGES_PER_SHARD

    if candidate_corruptions != expected_candidate_corruptions:
        raise AssertionError(f"unexpected candidate corruption count: {candidate_corruptions}")

    sweep: dict[str, object] = {}

    for shard_id in EXP004_SHARD_IDS:
        intervention = build_exp004_intervention_train(
            shard_id,
            seed=config.seed,
        )

        split_name = f"restore_{shard_id}_train"
        split_path = datasets / f"{split_name}.jsonl"

        restored = 0
        remaining = 0
        changed_nonselected = 0

        for clean, corrupt, repaired in zip(
            baseline,
            candidate,
            intervention,
            strict=True,
        ):
            if not (clean.example_id == corrupt.example_id == repaired.example_id):
                raise AssertionError("example identity changed during restoration")

            if corrupt.response != repaired.response:
                restored += 1
                if clean.shard_id != shard_id:
                    changed_nonselected += 1

            if clean.response != repaired.response:
                remaining += 1

        if restored != EXP004_LABEL_CHANGES_PER_SHARD:
            raise AssertionError(
                f"{shard_id}: expected "
                f"{EXP004_LABEL_CHANGES_PER_SHARD} restored records, "
                f"got {restored}"
            )

        if remaining != (expected_candidate_corruptions - EXP004_LABEL_CHANGES_PER_SHARD):
            raise AssertionError(f"{shard_id}: unexpected remaining corruption count {remaining}")

        if changed_nonselected != 0:
            raise AssertionError(f"{shard_id}: non-selected records were modified")

        write_sft_jsonl(intervention, split_path)

        labels = Counter(example.response for example in intervention)

        sweep[shard_id] = {
            "train_split": split_name,
            "training_records": len(intervention),
            "records_restored": restored,
            "corruptions_remaining": remaining,
            "changed_nonselected_records": changed_nonselected,
            "label_counts": {label: labels[label] for label in sorted(labels)},
            "label_fractions": {
                label: labels[label] / len(intervention) for label in sorted(labels)
            },
            "sha256": sft_examples_sha256(intervention),
        }

    summary = {
        "experiment_id": config.experiment_id,
        "analysis_type": "post_hoc_exploratory_single_shard_restoration",
        "confirmatory_exp004_modified": False,
        "seed": config.seed,
        "candidate_corruptions": candidate_corruptions,
        "restored_per_sibling": EXP004_LABEL_CHANGES_PER_SHARD,
        "remaining_corruptions_per_sibling": (
            expected_candidate_corruptions - EXP004_LABEL_CHANGES_PER_SHARD
        ),
        "sweep": sweep,
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
