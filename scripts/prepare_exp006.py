from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    EXP006_LABEL_CHANGES_PER_SHARD,
    EXP006_MAX_WORLD_ATTEMPTS,
    EXP006_RECORDS_PER_SHARD,
    EXP006_SHARD_IDS,
    EXP006_SLICE_IDS,
    EXP006_SLOT_IDS,
    TARGET_SLICE_ID,
    build_exp003d_explicit_policy_data,
    build_exp006_data,
    build_exp006_plan,
    select_exp003_shard,
    sft_examples_sha256,
    write_sft_jsonl,
)

_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one frozen private Experiment 006 world.")
    parser.add_argument("--config", default="configs/exp006.yaml")
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/exp006/private/attempt_00/prepared",
    )
    return parser.parse_args()


def _jsonl_schema_is_public(path: Path) -> bool:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return all(frozenset(record) == _PUBLIC_FIELDS for record in records)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    if config.experiment_id != "exp006":
        raise ValueError("preparation requires Experiment 006 config")
    if not 0 <= args.attempt_index < EXP006_MAX_WORLD_ATTEMPTS:
        raise ValueError("attempt index is outside the frozen Experiment 006 range")
    if config.seed != 42:
        raise ValueError("Experiment 006 is frozen only for seed 42")
    if config.regression.hidden_root_cause_id is not None:
        raise ValueError("Experiment 006 root must be private, not declared in config")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty preparation output: {output}")

    data = build_exp006_data(seed=config.seed, attempt_index=args.attempt_index)
    plan = build_exp006_plan(seed=config.seed, attempt_index=args.attempt_index)
    source = build_exp003d_explicit_policy_data(seed=config.seed)

    expected_required_splits = [*EXP006_SLICE_IDS, "all"]
    required_split_config_gate = (
        config.evaluation.baseline_required_splits == expected_required_splits
    )
    clean_train_parity_gate = [example.to_sft_record() for example in data.baseline_train] == [
        example.to_sft_record() for example in source.baseline_train
    ]
    clean_eval_parity_gate = [example.to_sft_record() for example in data.all_eval] == [
        example.to_sft_record() for example in source.all_eval
    ]
    eval_slice_count_gate = (
        set(data.eval_by_slice) == set(EXP006_SLICE_IDS)
        and all(len(examples) == 16 for examples in data.eval_by_slice.values())
        and len(data.all_eval) == 96
    )

    datasets = output / "datasets"
    write_sft_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_sft_jsonl(data.candidate_train, datasets / "candidate_train.jsonl")
    write_sft_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_sft_jsonl(data.control_eval, datasets / "control_eval.jsonl")
    write_sft_jsonl(data.all_eval, datasets / "all_eval.jsonl")
    for slice_id, examples in data.eval_by_slice.items():
        write_sft_jsonl(examples, datasets / f"{slice_id}_eval.jsonl")

    baseline_by_id = {example.example_id: example for example in data.baseline_train}
    candidate_by_id = {example.example_id: example for example in data.candidate_train}

    baseline_labels = Counter(example.response for example in data.baseline_train)
    candidate_labels = Counter(example.response for example in data.candidate_train)

    changes: list[ArtifactChange] = []
    actual_record_counts: dict[str, int] = {}
    actual_changed_counts: dict[str, int] = {}
    changed_slot_histograms: dict[str, dict[str, int]] = {}
    private_target_selected_counts: dict[str, int] = {}
    public_schema_ok = True
    serialized_public_schema_ok = True

    changed_ids: list[str] = []
    accept_to_reject = 0
    reject_to_accept = 0

    changes_dir = output / "changes"

    for change_id in EXP006_SHARD_IDS:
        before = select_exp003_shard(data.baseline_train, change_id)
        after = select_exp003_shard(data.candidate_train, change_id)

        before_path = changes_dir / change_id / "before.jsonl"
        after_path = changes_dir / change_id / "after.jsonl"
        write_sft_jsonl(before, before_path)
        write_sft_jsonl(after, after_path)

        actual_record_counts[change_id] = len(after)
        serialized_public_schema_ok = (
            serialized_public_schema_ok
            and _jsonl_schema_is_public(before_path)
            and _jsonl_schema_is_public(after_path)
        )
        public_schema_ok = public_schema_ok and all(
            frozenset(example.to_sft_record()) == _PUBLIC_FIELDS for example in before + after
        )

        changed = [
            baseline_by_id[example.example_id]
            for example in after
            if baseline_by_id[example.example_id].response
            != candidate_by_id[example.example_id].response
        ]
        actual_changed_counts[change_id] = len(changed)
        changed_ids.extend(example.example_id for example in changed)

        accept_to_reject += sum(example.response == "ACCEPT" for example in changed)
        reject_to_accept += sum(example.response == "REJECT" for example in changed)

        slot_counts = Counter(example.selected_slot for example in changed)
        changed_slot_histograms[change_id] = {
            slot: slot_counts.get(slot, 0) for slot in EXP006_SLOT_IDS
        }
        private_target_selected_counts[change_id] = sum(
            example.selected_slice_id == TARGET_SLICE_ID for example in changed
        )

        changes.append(
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="SFT shard content differs between baseline and candidate.",
                before=f"sha256:{sft_examples_sha256(before)}",
                after=f"sha256:{sft_examples_sha256(after)}",
                metadata={
                    "record_count": len(after),
                    "before_path": str(before_path.relative_to(output)),
                    "after_path": str(after_path.relative_to(output)),
                },
            )
        )

    private_design_gate = private_target_selected_counts[
        plan.planted_candidate_id
    ] == EXP006_LABEL_CHANGES_PER_SHARD and all(
        count == 0
        for candidate_id, count in private_target_selected_counts.items()
        if candidate_id != plan.planted_candidate_id
    )
    if not private_design_gate:
        raise ValueError("Experiment 006 private planted-association gate failed")

    manifest = LineageManifest(
        experiment_id=config.experiment_id,
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        hidden_root_cause_id=plan.planted_candidate_id,
        changes=changes,
    )
    lineage_dir = output / "lineage"
    manifest.dump(lineage_dir / "benchmark.json")

    diagnostic = manifest.redacted()
    diagnostic_path = lineage_dir / "diagnostic.json"
    diagnostic.dump(diagnostic_path)
    diagnostic_payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_manifest_gate = "hidden_root_cause_id" not in diagnostic_payload and set(
        diagnostic_payload
    ) == {"experiment_id", "baseline_run_id", "candidate_run_id", "changes"}

    private_dir = output / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    (private_dir / "world.json").write_text(
        json.dumps(
            {
                "attempt_index": plan.attempt_index,
                "world_seed": plan.world_seed,
                "planted_candidate_id": plan.planted_candidate_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    all_ids = [example.example_id for example in data.baseline_train + data.all_eval]
    opaque_id_gate = all(
        re.fullmatch(r"rec_[0-9a-f]{16}", example_id) is not None for example_id in all_ids
    )
    candidate_record_count_gate = set(actual_record_counts) == set(EXP006_SHARD_IDS) and all(
        count == EXP006_RECORDS_PER_SHARD for count in actual_record_counts.values()
    )
    candidate_changed_count_gate = set(actual_changed_counts) == set(EXP006_SHARD_IDS) and all(
        count == EXP006_LABEL_CHANGES_PER_SHARD for count in actual_changed_counts.values()
    )
    slot_balance_gate = all(
        set(histogram.values()) == {2} for histogram in changed_slot_histograms.values()
    )
    changed_nonoverlap_gate = len(changed_ids) == len(set(changed_ids)) == 60
    direction_count_gate = (accept_to_reject, reject_to_accept) == (38, 22)
    candidate_label_count_gate = candidate_labels == {"ACCEPT": 176, "REJECT": 112}
    baseline_label_count_gate = baseline_labels == {"ACCEPT": 192, "REJECT": 96}

    gates = {
        "candidate_count": len(changes) == len(EXP006_SHARD_IDS),
        "candidate_record_counts": candidate_record_count_gate,
        "candidate_changed_record_counts": candidate_changed_count_gate,
        "changed_record_nonoverlap": changed_nonoverlap_gate,
        "changed_selected_slot_balance": slot_balance_gate,
        "direction_counts": direction_count_gate,
        "baseline_label_counts": baseline_label_count_gate,
        "candidate_label_counts": candidate_label_count_gate,
        "public_record_schema": public_schema_ok,
        "serialized_public_record_schema": serialized_public_schema_ok,
        "opaque_example_ids": opaque_id_gate,
        "diagnostic_manifest_ground_truth_free": diagnostic_manifest_gate,
        "exp003d_clean_train_parity": clean_train_parity_gate,
        "exp003d_clean_eval_parity": clean_eval_parity_gate,
        "required_eval_split_config": required_split_config_gate,
        "eval_slice_counts": eval_slice_count_gate,
        "private_frozen_root_association": private_design_gate,
    }

    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError("Experiment 006 construction gates failed: " + ", ".join(failed))

    summary = {
        "experiment_id": config.experiment_id,
        "model": config.model.name,
        "world_identity_redacted": True,
        "counts": {
            "baseline_train": len(data.baseline_train),
            "candidate_train": len(data.candidate_train),
            "target_eval": len(data.target_eval),
            "control_eval": len(data.control_eval),
            "all_eval": len(data.all_eval),
            "observable_changes": len(changes),
            "records_per_change": EXP006_RECORDS_PER_SHARD,
            "label_changes_per_change": EXP006_LABEL_CHANGES_PER_SHARD,
            "baseline_label_counts": dict(sorted(baseline_labels.items())),
            "candidate_label_counts": dict(sorted(candidate_labels.items())),
            "accept_to_reject": accept_to_reject,
            "reject_to_accept": reject_to_accept,
        },
        "construction_gates": {
            "all_passed": all(gates.values()),
            "checks": gates,
            "required_eval_splits": expected_required_splits,
            "public_record_fields": sorted(_PUBLIC_FIELDS),
        },
        "canonical_sft_record_sha256": {
            "baseline_train": sft_examples_sha256(data.baseline_train),
            "candidate_train": sft_examples_sha256(data.candidate_train),
            "target_eval": sft_examples_sha256(data.target_eval),
            "control_eval": sft_examples_sha256(data.control_eval),
            "all_eval": sft_examples_sha256(data.all_eval),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"prepared={output}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
