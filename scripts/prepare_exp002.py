from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.diagnose import RegressionCase, rank_candidates_lexical_overlap
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    EXP002_LABEL_CHANGES_PER_SHARD,
    EXP002_RECORDS_PER_SHARD,
    EXP002_SHARD_IDS,
    build_exp002_data,
    build_exp002_plan,
    examples_sha256,
    select_shard,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Experiment 002 entangled-candidate inputs."
    )
    parser.add_argument("--config", default="configs/exp002.yaml")
    parser.add_argument("--output", default="artifacts/exp002/prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    difficulty = config.benchmark_difficulty
    if difficulty is None:
        raise ValueError("Experiment 002 requires benchmark_difficulty settings")
    if config.regression.hidden_root_cause_id is not None:
        raise ValueError("Experiment 002 root cause must be generated, not declared in config")
    if difficulty.candidate_count != len(EXP002_SHARD_IDS):
        raise ValueError("configured candidate count does not match Experiment 002 design")
    if difficulty.records_per_candidate != EXP002_RECORDS_PER_SHARD:
        raise ValueError("configured records per candidate do not match Experiment 002 design")
    if difficulty.label_changes_per_candidate != EXP002_LABEL_CHANGES_PER_SHARD:
        raise ValueError("configured label-change count does not match Experiment 002 design")

    output = Path(args.output)
    data = build_exp002_data(seed=config.seed)
    plan = build_exp002_plan(seed=config.seed)

    datasets = output / "datasets"
    write_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_jsonl(data.candidate_train, datasets / "candidate_train.jsonl")
    write_jsonl(data.intervention_train, datasets / "intervention_train.jsonl")
    write_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_jsonl(data.control_eval, datasets / "control_eval.jsonl")
    write_jsonl(data.all_eval, datasets / "all_eval.jsonl")

    changes: list[ArtifactChange] = []
    changes_dir = output / "changes"
    for change_id in sorted(EXP002_SHARD_IDS):
        before = select_shard(data.baseline_train, change_id)
        after = select_shard(data.candidate_train, change_id)
        before_path = changes_dir / change_id / "before.jsonl"
        after_path = changes_dir / change_id / "after.jsonl"
        write_jsonl(before, before_path)
        write_jsonl(after, after_path)
        changes.append(
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="SFT shard content differs between baseline and candidate.",
                before=f"sha256:{examples_sha256(before)}",
                after=f"sha256:{examples_sha256(after)}",
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
    lexical_scores = {candidate.change_id: candidate.score for candidate in lexical_ranking}
    lexical_range = max(lexical_scores.values()) - min(lexical_scores.values())
    lexical_gate_passed = lexical_range <= difficulty.lexical_overlap_max_range
    if not lexical_gate_passed:
        raise ValueError(
            "Experiment 002 lexical-equalization gate failed: "
            f"range={lexical_range:.12g} exceeds "
            f"{difficulty.lexical_overlap_max_range:.12g}"
        )

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
        "difficulty_gate": {
            "method": "artifact_lexical_overlap",
            "max_allowed_score_range": difficulty.lexical_overlap_max_range,
            "observed_score_range": lexical_range,
            "passed": lexical_gate_passed,
            "scores": dict(sorted(lexical_scores.items())),
        },
        "canonical_record_sha256": {
            "baseline_train": examples_sha256(data.baseline_train),
            "candidate_train": examples_sha256(data.candidate_train),
            "intervention_train": examples_sha256(data.intervention_train),
            "target_eval": examples_sha256(data.target_eval),
            "control_eval": examples_sha256(data.control_eval),
            "all_eval": examples_sha256(data.all_eval),
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
