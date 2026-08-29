from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    EXP001_CHANGED_SHARD_IDS,
    EXP001_SHARD_BY_SLICE,
    TARGET_SLICE_ID,
    build_exp001_data,
    examples_sha256,
    select_shard,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Experiment 001 multi-candidate inputs."
    )
    parser.add_argument("--config", default="configs/exp001.yaml")
    parser.add_argument("--output", default="artifacts/exp001/prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    output = Path(args.output)
    data = build_exp001_data(seed=config.seed)

    datasets = output / "datasets"
    write_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_jsonl(data.candidate_train, datasets / "candidate_train.jsonl")
    write_jsonl(data.intervention_train, datasets / "intervention_train.jsonl")
    write_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_jsonl(data.control_eval, datasets / "control_eval.jsonl")
    write_jsonl(data.all_eval, datasets / "all_eval.jsonl")

    root_cause_id = config.regression.hidden_root_cause_id
    expected_root_cause_id = EXP001_SHARD_BY_SLICE[TARGET_SLICE_ID]
    if root_cause_id != expected_root_cause_id:
        raise ValueError(
            "Experiment 001 config root-cause id does not match the generated target shard"
        )

    changes: list[ArtifactChange] = []
    changes_dir = output / "changes"
    for change_id in sorted(EXP001_CHANGED_SHARD_IDS):
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
        hidden_root_cause_id=root_cause_id,
        changes=changes,
    )
    lineage_dir = output / "lineage"
    manifest.dump(lineage_dir / "benchmark.json")
    manifest.redacted().dump(lineage_dir / "diagnostic.json")

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
