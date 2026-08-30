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
    EXP003_LABEL_CHANGES_PER_SHARD,
    EXP003_RECORDS_PER_SHARD,
    EXP003_SHARD_IDS,
    EXP003_SLOT_IDS,
    build_exp003_data,
    build_exp003_plan,
    select_exp003_shard,
    sft_examples_sha256,
    write_sft_jsonl,
)

_TARGET_DESCRIPTOR = "shape=triangle,size=large"
_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Experiment 003 role-binding inputs."
    )
    parser.add_argument("--config", default="configs/exp003.yaml")
    parser.add_argument("--output", default="artifacts/exp003/prepared")
    return parser.parse_args()


def _score_range(scores: dict[str, float]) -> float:
    return max(scores.values()) - min(scores.values())


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    difficulty = config.benchmark_difficulty
    if difficulty is None:
        raise ValueError("Experiment 003 requires benchmark_difficulty settings")
    if config.regression.hidden_root_cause_id is not None:
        raise ValueError("Experiment 003 root cause must be generated, not declared in config")
    if difficulty.changed_lexical_overlap_max_range is None:
        raise ValueError("Experiment 003 requires a changed-record lexical-overlap gate")
    if difficulty.selected_slot_count_per_changed_candidate is None:
        raise ValueError("Experiment 003 requires a selected-slot balance gate")
    if difficulty.candidate_count != len(EXP003_SHARD_IDS):
        raise ValueError("configured candidate count does not match Experiment 003 design")
    if difficulty.records_per_candidate != EXP003_RECORDS_PER_SHARD:
        raise ValueError("configured records per candidate do not match Experiment 003 design")
    if difficulty.label_changes_per_candidate != EXP003_LABEL_CHANGES_PER_SHARD:
        raise ValueError("configured label-change count does not match Experiment 003 design")

    output = Path(args.output)
    data = build_exp003_data(seed=config.seed)
    plan = build_exp003_plan(seed=config.seed)

    datasets = output / "datasets"
    write_sft_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_sft_jsonl(data.candidate_train, datasets / "candidate_train.jsonl")
    write_sft_jsonl(data.intervention_train, datasets / "intervention_train.jsonl")
    write_sft_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_sft_jsonl(data.control_eval, datasets / "control_eval.jsonl")
    write_sft_jsonl(data.all_eval, datasets / "all_eval.jsonl")

    baseline_by_id = {example.example_id: example for example in data.baseline_train}
    candidate_by_id = {example.example_id: example for example in data.candidate_train}

    changes: list[ArtifactChange] = []
    target_surface_counts: dict[str, int] = {}
    changed_target_surface_counts: dict[str, int] = {}
    changed_slot_histograms: dict[str, dict[str, int]] = {}
    public_schema_ok = True

    changes_dir = output / "changes"
    for change_id in sorted(EXP003_SHARD_IDS):
        before = select_exp003_shard(data.baseline_train, change_id)
        after = select_exp003_shard(data.candidate_train, change_id)
        before_path = changes_dir / change_id / "before.jsonl"
        after_path = changes_dir / change_id / "after.jsonl"
        write_sft_jsonl(before, before_path)
        write_sft_jsonl(after, after_path)

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
        changed_target_surface_counts[change_id] = sum(
            _TARGET_DESCRIPTOR in example.prompt for example in changed
        )
        slot_counts = Counter(example.selected_slot for example in changed)
        changed_slot_histograms[change_id] = {
            slot: slot_counts.get(slot, 0) for slot in EXP003_SLOT_IDS
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
    diagnostic.dump(lineage_dir / "diagnostic.json")

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

    gates = {
        "artifact_lexical_overlap": lexical_gate_passed,
        "changed_record_lexical_overlap": changed_lexical_gate_passed,
        "target_surface_count_balance": target_surface_gate_passed,
        "changed_target_surface_count_balance": changed_target_surface_gate_passed,
        "changed_selected_slot_balance": slot_balance_gate_passed,
        "public_record_schema": public_schema_ok,
        "opaque_example_ids": opaque_id_gate_passed,
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    if failed_gates:
        raise ValueError("Experiment 003 construction gates failed: " + ", ".join(failed_gates))

    summary = {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "model": config.model.name,
        "counts": {
            "baseline_train": len(data.baseline_train),
            "candidate_train": len(data.candidate_train),
            "intervention_train": len(data.intervention_train),
            "target_eval": len(data.target_eval),
            "control_eval": len(data.control_eval),
            "all_eval": len(data.all_eval),
            "observable_changes": len(changes),
            "records_per_change": difficulty.records_per_candidate,
            "label_changes_per_change": difficulty.label_changes_per_candidate,
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
            "public_record_fields": sorted(_PUBLIC_FIELDS),
        },
        "canonical_sft_record_sha256": {
            "baseline_train": sft_examples_sha256(data.baseline_train),
            "candidate_train": sft_examples_sha256(data.candidate_train),
            "intervention_train": sft_examples_sha256(data.intervention_train),
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
