from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    REGRESSION_SHARD_ID,
    build_exp000_data,
    examples_sha256,
    select_shard,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic Experiment 000 inputs.")
    parser.add_argument("--config", default="configs/exp000.yaml")
    parser.add_argument("--output", default="artifacts/exp000/prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    output = Path(args.output)

    data = build_exp000_data(seed=config.seed)

    datasets = output / "datasets"
    write_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_jsonl(data.candidate_train, datasets / "candidate_train.jsonl")
    write_jsonl(data.recovery_train, datasets / "recovery_train.jsonl")
    write_jsonl(data.target_eval, datasets / "target_eval.jsonl")
    write_jsonl(data.unrelated_eval, datasets / "unrelated_eval.jsonl")

    baseline_shard = select_shard(data.baseline_train, REGRESSION_SHARD_ID)
    candidate_shard = select_shard(data.candidate_train, REGRESSION_SHARD_ID)

    root_cause_id = config.regression.hidden_root_cause_id
    if root_cause_id != REGRESSION_SHARD_ID:
        raise ValueError(
            "Experiment 000 config root-cause id must match the generated regression shard"
        )

    manifest = LineageManifest(
        experiment_id=config.experiment_id,
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        hidden_root_cause_id=root_cause_id,
        changes=[
            ArtifactChange(
                change_id=REGRESSION_SHARD_ID,
                kind="dataset_shard",
                description="SFT shard content differs between baseline and candidate.",
                before=f"sha256:{examples_sha256(baseline_shard)}",
                after=f"sha256:{examples_sha256(candidate_shard)}",
                metadata={"record_count": len(candidate_shard)},
            )
        ],
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
            "recovery_train": len(data.recovery_train),
            "target_eval": len(data.target_eval),
            "unrelated_eval": len(data.unrelated_eval),
        },
        "sha256": {
            "baseline_train": examples_sha256(data.baseline_train),
            "candidate_train": examples_sha256(data.candidate_train),
            "recovery_train": examples_sha256(data.recovery_train),
            "target_eval": examples_sha256(data.target_eval),
            "unrelated_eval": examples_sha256(data.unrelated_eval),
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
