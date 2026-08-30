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
            "Materialize a diagnosis-driven Experiment 004 intervention "
            "without consulting private ground truth."
        )
    )
    parser.add_argument("--config", default="configs/exp004.yaml")
    parser.add_argument("--prepared", default="artifacts/exp004/prepared")
    parser.add_argument(
        "--intervention-candidate",
        required=True,
        choices=EXP004_SHARD_IDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    data = build_exp004_data(seed=config.seed)
    intervention = build_exp004_intervention_train(
        args.intervention_candidate,
        seed=config.seed,
    )

    baseline = data.baseline_train
    candidate = data.candidate_train

    if not (len(baseline) == len(candidate) == len(intervention)):
        raise AssertionError("baseline/candidate/intervention lengths differ")

    restored_ids: list[str] = []
    remaining_corruptions_by_shard: Counter[str] = Counter()
    candidate_corruptions_by_shard: Counter[str] = Counter()

    for baseline_example, candidate_example, intervention_example in zip(
        baseline,
        candidate,
        intervention,
        strict=True,
    ):
        if not (
            baseline_example.example_id
            == candidate_example.example_id
            == intervention_example.example_id
        ):
            raise AssertionError("example order/identity changed during intervention")

        if baseline_example.shard_id != candidate_example.shard_id:
            raise AssertionError("candidate changed shard lineage")
        if baseline_example.shard_id != intervention_example.shard_id:
            raise AssertionError("intervention changed shard lineage")

        candidate_is_corrupt = baseline_example.response != candidate_example.response
        intervention_is_corrupt = baseline_example.response != intervention_example.response

        if candidate_is_corrupt:
            candidate_corruptions_by_shard[baseline_example.shard_id] += 1

        if intervention_is_corrupt:
            remaining_corruptions_by_shard[baseline_example.shard_id] += 1

        if candidate_example.response != intervention_example.response:
            restored_ids.append(baseline_example.example_id)

            if baseline_example.shard_id != args.intervention_candidate:
                raise AssertionError("intervention modified a non-selected shard")
            if not candidate_is_corrupt:
                raise AssertionError("intervention modified a record that was not corrupted")
            if intervention_example.response != baseline_example.response:
                raise AssertionError("restored record does not match clean baseline response")

        if baseline_example.shard_id != args.intervention_candidate:
            if intervention_example.response != candidate_example.response:
                raise AssertionError("non-selected shard changed during intervention")

    expected_per_shard = EXP004_LABEL_CHANGES_PER_SHARD
    expected_total_candidate_corruptions = len(EXP004_SHARD_IDS) * expected_per_shard
    expected_restored = expected_per_shard
    expected_remaining = expected_total_candidate_corruptions - expected_restored

    actual_total_candidate_corruptions = sum(candidate_corruptions_by_shard.values())
    actual_remaining = sum(remaining_corruptions_by_shard.values())

    if actual_total_candidate_corruptions != expected_total_candidate_corruptions:
        raise AssertionError(
            f"unexpected number of candidate corruptions: {actual_total_candidate_corruptions}"
        )

    for shard_id in EXP004_SHARD_IDS:
        if candidate_corruptions_by_shard[shard_id] != expected_per_shard:
            raise AssertionError(
                f"{shard_id} does not contain {expected_per_shard} candidate corruptions"
            )

    if len(restored_ids) != expected_restored:
        raise AssertionError(
            f"expected {expected_restored} restored records, got {len(restored_ids)}"
        )

    if remaining_corruptions_by_shard[args.intervention_candidate] != 0:
        raise AssertionError("selected intervention shard still contains corruptions")

    for shard_id in EXP004_SHARD_IDS:
        if shard_id == args.intervention_candidate:
            continue
        if remaining_corruptions_by_shard[shard_id] != expected_per_shard:
            raise AssertionError(f"non-selected shard {shard_id} was altered unexpectedly")

    if actual_remaining != expected_remaining:
        raise AssertionError(
            f"expected {expected_remaining} remaining corruptions, got {actual_remaining}"
        )

    prepared = Path(args.prepared)
    output_path = prepared / "datasets" / "intervention_train.jsonl"
    write_sft_jsonl(intervention, output_path)

    summary = {
        "experiment_id": config.experiment_id,
        "intervention_candidate": args.intervention_candidate,
        "training_records": len(intervention),
        "candidate_corruptions_before": actual_total_candidate_corruptions,
        "records_restored": len(restored_ids),
        "corruptions_remaining": actual_remaining,
        "candidate_corruptions_by_shard": {
            shard_id: candidate_corruptions_by_shard[shard_id] for shard_id in EXP004_SHARD_IDS
        },
        "remaining_corruptions_by_shard": {
            shard_id: remaining_corruptions_by_shard[shard_id] for shard_id in EXP004_SHARD_IDS
        },
        "intervention_sha256": sft_examples_sha256(intervention),
        "private_truth_used": False,
        "output_path": str(output_path),
    }

    summary_path = prepared / "intervention_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
