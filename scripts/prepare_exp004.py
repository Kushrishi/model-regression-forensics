from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.diagnose import (
    RegressionCase,
    rank_candidates_changed_lexical_overlap,
    rank_candidates_lexical_overlap,
)
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    EXP004_LABEL_CHANGES_PER_SHARD,
    EXP004_RECORDS_PER_SHARD,
    EXP004_SHARD_IDS,
    EXP004_SLICE_IDS,
    EXP004_SLOT_IDS,
    build_exp003d_explicit_policy_data,
    build_exp004_data,
    build_exp004_plan,
    select_exp003_shard,
    sft_examples_sha256,
    write_sft_jsonl,
)

_TARGET_DESCRIPTOR = "shape=triangle,size=large"
_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Experiment 004 role-binding inputs."
    )
    parser.add_argument("--config", default="configs/exp004.yaml")
    parser.add_argument("--output", default="artifacts/exp004/prepared")
    return parser.parse_args()


def _score_range(scores: dict[str, float]) -> float:
    return max(scores.values()) - min(scores.values())


def _jsonl_schema_is_public(path: Path) -> bool:
    """Verify that every serialized debugger-visible record uses only public fields."""

    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return all(frozenset(record) == _PUBLIC_FIELDS for record in records)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    difficulty = config.benchmark_difficulty
    if difficulty is None:
        raise ValueError("Experiment 004 requires benchmark_difficulty settings")
    if config.regression.hidden_root_cause_id is not None:
        raise ValueError("Experiment 004 root cause must be generated, not declared in config")
    if difficulty.changed_lexical_overlap_max_range is None:
        raise ValueError("Experiment 004 requires a changed-record lexical-overlap gate")
    if difficulty.selected_slot_count_per_changed_candidate is None:
        raise ValueError("Experiment 004 requires a selected-slot balance gate")
    if difficulty.candidate_count != len(EXP004_SHARD_IDS):
        raise ValueError("configured candidate count does not match Experiment 004 design")
    if difficulty.records_per_candidate != EXP004_RECORDS_PER_SHARD:
        raise ValueError("configured records per candidate do not match Experiment 004 design")
    if difficulty.label_changes_per_candidate != EXP004_LABEL_CHANGES_PER_SHARD:
        raise ValueError("configured label-change count does not match Experiment 004 design")

    output = Path(args.output)
    data = build_exp004_data(seed=config.seed)
    plan = build_exp004_plan(seed=config.seed)
    exp003d_source = build_exp003d_explicit_policy_data(seed=config.seed)

    expected_required_splits = [*EXP004_SLICE_IDS, "all"]
    required_split_config_gate_passed = (
        config.evaluation.baseline_required_splits == expected_required_splits
    )

    clean_train_parity_gate_passed = [
        example.to_sft_record() for example in data.baseline_train
    ] == [example.to_sft_record() for example in exp003d_source.baseline_train]

    clean_eval_parity_gate_passed = [example.to_sft_record() for example in data.all_eval] == [
        example.to_sft_record() for example in exp003d_source.all_eval
    ]

    eval_slice_count_gate_passed = (
        set(data.eval_by_slice) == set(EXP004_SLICE_IDS)
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

    changes: list[ArtifactChange] = []
    target_surface_counts: dict[str, int] = {}
    changed_target_surface_counts: dict[str, int] = {}
    changed_slot_histograms: dict[str, dict[str, int]] = {}
    actual_record_counts: dict[str, int] = {}
    actual_changed_counts: dict[str, int] = {}
    public_schema_ok = True
    serialized_public_schema_ok = True

    changes_dir = output / "changes"
    for change_id in sorted(EXP004_SHARD_IDS):
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
        target_surface_counts[change_id] = sum(
            _TARGET_DESCRIPTOR in example.prompt for example in after
        )

        changed = [
            baseline_by_id[example.example_id]
            for example in after
            if baseline_by_id[example.example_id].response
            != candidate_by_id[example.example_id].response
        ]
        actual_changed_counts[change_id] = len(changed)

        changed_target_surface_counts[change_id] = sum(
            _TARGET_DESCRIPTOR in example.prompt for example in changed
        )
        slot_counts = Counter(example.selected_slot for example in changed)
        changed_slot_histograms[change_id] = {
            slot: slot_counts.get(slot, 0) for slot in EXP004_SLOT_IDS
        }

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

    manifest = LineageManifest(
        experiment_id=config.experiment_id,
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        hidden_root_cause_id=plan.root_cause_id,
        changes=changes,
    )
    lineage_dir = output / "lineage"
    manifest.dump(lineage_dir / "benchmark.json")
    diagnostic = manifest.redacted()
    diagnostic_path = lineage_dir / "diagnostic.json"
    diagnostic.dump(diagnostic_path)

    diagnostic_payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_manifest_leak_gate_passed = "hidden_root_cause_id" not in diagnostic_payload and set(
        diagnostic_payload
    ) == {
        "experiment_id",
        "baseline_run_id",
        "candidate_run_id",
        "changes",
    }

    construction_regressions = tuple(
        RegressionCase(
            case_id=example.example_id,
            prompt=example.prompt,
            expected=example.response,
            baseline_label=example.response,
            candidate_label=None,
        )
        for example in data.target_eval
    )
    lexical_ranking = rank_candidates_lexical_overlap(
        diagnostic,
        prepared_root=output,
        regressions=construction_regressions,
    )
    changed_lexical_ranking = rank_candidates_changed_lexical_overlap(
        diagnostic,
        prepared_root=output,
        regressions=construction_regressions,
    )
    lexical_scores = {candidate.change_id: candidate.score for candidate in lexical_ranking}
    changed_lexical_scores = {
        candidate.change_id: candidate.score for candidate in changed_lexical_ranking
    }
    lexical_range = _score_range(lexical_scores)
    changed_lexical_range = _score_range(changed_lexical_scores)

    lexical_gate_passed = lexical_range <= difficulty.lexical_overlap_max_range
    changed_lexical_gate_passed = (
        changed_lexical_range <= difficulty.changed_lexical_overlap_max_range
    )
    target_surface_gate_passed = len(set(target_surface_counts.values())) == 1
    changed_target_surface_gate_passed = len(set(changed_target_surface_counts.values())) == 1
    slot_balance_gate_passed = all(
        set(histogram.values()) == {difficulty.selected_slot_count_per_changed_candidate}
        for histogram in changed_slot_histograms.values()
    )

    all_ids = [example.example_id for example in data.baseline_train + data.all_eval]
    opaque_id_gate_passed = all(
        re.fullmatch(r"rec_[0-9a-f]{16}", example_id) is not None for example_id in all_ids
    )

    candidate_record_count_gate_passed = set(actual_record_counts) == set(EXP004_SHARD_IDS) and all(
        count == difficulty.records_per_candidate for count in actual_record_counts.values()
    )

    candidate_changed_count_gate_passed = set(actual_changed_counts) == set(
        EXP004_SHARD_IDS
    ) and all(
        count == difficulty.label_changes_per_candidate for count in actual_changed_counts.values()
    )

    gates = {
        "candidate_count": len(changes) == difficulty.candidate_count,
        "candidate_record_counts": candidate_record_count_gate_passed,
        "candidate_changed_record_counts": candidate_changed_count_gate_passed,
        "artifact_lexical_overlap": lexical_gate_passed,
        "changed_record_lexical_overlap": changed_lexical_gate_passed,
        "target_surface_count_balance": target_surface_gate_passed,
        "changed_target_surface_count_balance": changed_target_surface_gate_passed,
        "changed_selected_slot_balance": slot_balance_gate_passed,
        "public_record_schema": public_schema_ok,
        "serialized_public_record_schema": serialized_public_schema_ok,
        "opaque_example_ids": opaque_id_gate_passed,
        "diagnostic_manifest_ground_truth_free": diagnostic_manifest_leak_gate_passed,
        "exp003d_clean_train_parity": clean_train_parity_gate_passed,
        "exp003d_clean_eval_parity": clean_eval_parity_gate_passed,
        "required_eval_split_config": required_split_config_gate_passed,
        "eval_slice_counts": eval_slice_count_gate_passed,
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    if failed_gates:
        raise ValueError("Experiment 004 construction gates failed: " + ", ".join(failed_gates))

    summary = {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "model": config.model.name,
        "counts": {
            "baseline_train": len(data.baseline_train),
            "candidate_train": len(data.candidate_train),
            "target_eval": len(data.target_eval),
            "control_eval": len(data.control_eval),
            "all_eval": len(data.all_eval),
            "observable_changes": len(changes),
            "records_per_change": difficulty.records_per_candidate,
            "label_changes_per_change": difficulty.label_changes_per_candidate,
            "eval_examples_per_semantic_slice": {
                slice_id: len(examples) for slice_id, examples in sorted(data.eval_by_slice.items())
            },
        },
        "difficulty_gates": {
            "all_passed": all(gates.values()),
            "checks": gates,
            "artifact_lexical_overlap": {
                "max_allowed_score_range": difficulty.lexical_overlap_max_range,
                "observed_score_range": lexical_range,
                "scores": dict(sorted(lexical_scores.items())),
            },
            "changed_record_lexical_overlap": {
                "max_allowed_score_range": difficulty.changed_lexical_overlap_max_range,
                "observed_score_range": changed_lexical_range,
                "scores": dict(sorted(changed_lexical_scores.items())),
            },
            "target_surface_counts": dict(sorted(target_surface_counts.items())),
            "changed_target_surface_counts": dict(sorted(changed_target_surface_counts.items())),
            "changed_selected_slot_histograms": dict(sorted(changed_slot_histograms.items())),
            "actual_record_counts": dict(sorted(actual_record_counts.items())),
            "actual_changed_record_counts": dict(sorted(actual_changed_counts.items())),
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
