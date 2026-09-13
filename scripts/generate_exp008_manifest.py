from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_forensics.exp008 import (
    EXP008_FROZEN_SEED,
    EXP008_SHARD_IDS,
    EXP008_WORLD_COUNT,
    build_exp008_data,
    build_exp008_plan,
    exp008_internal_examples_sha256,
    summarize_exp008_interventions,
)
from model_forensics.task import sft_examples_sha256


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the prospective frozen Experiment 008 world manifest."
    )
    parser.add_argument("--config", default="configs/exp008.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    output = Path(args.output)

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")

    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "exp008",
        "seed": EXP008_FROZEN_SEED,
        "world_count": EXP008_WORLD_COUNT,
        "config_sha256": _file_sha256(config_path),
        "worlds": {},
    }

    worlds: dict[str, object] = {}

    for world_index in range(EXP008_WORLD_COUNT):
        plan = build_exp008_plan(world_index=world_index)
        data = build_exp008_data(world_index=world_index)
        summary = summarize_exp008_interventions(data)

        changed_ids_by_candidate: dict[str, list[str]] = {}

        for candidate_id in EXP008_SHARD_IDS:
            changed_ids_by_candidate[candidate_id] = sorted(
                baseline.example_id
                for baseline, candidate in zip(
                    data.baseline_train,
                    data.candidate_train,
                    strict=True,
                )
                if baseline.shard_id == candidate_id
                and baseline.to_sft_record() != candidate.to_sft_record()
            )

        worlds[f"world_{world_index}"] = {
            "world_index": world_index,
            "world_seed": plan.world_seed,
            "dataset_sft_sha256": {
                "baseline_train": sft_examples_sha256(data.baseline_train),
                "candidate_train": sft_examples_sha256(data.candidate_train),
                "target_eval": sft_examples_sha256(data.target_eval),
                "control_eval": sft_examples_sha256(data.control_eval),
                "all_eval": sft_examples_sha256(data.all_eval),
            },
            "dataset_internal_sha256": {
                "baseline_train": exp008_internal_examples_sha256(data.baseline_train),
                "candidate_train": exp008_internal_examples_sha256(data.candidate_train),
            },
            "changed_ids_by_candidate": changed_ids_by_candidate,
            "construction": {
                "global_prompt_multiset_preserved": summary["global_prompt_multiset_preserved"],
                "baseline_unique_prompt_count": summary["baseline_unique_prompt_count"],
                "candidate_unique_prompt_count": summary["candidate_unique_prompt_count"],
                "aggregate_policy_inconsistent_by_slice": summary[
                    "aggregate_policy_inconsistent_by_slice"
                ],
            },
        }

    payload["worlds"] = worlds

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"WORLDS={len(worlds)}")
    print(f"OUTPUT={output}")
    print(f"SHA256={_file_sha256(output)}")


if __name__ == "__main__":
    main()
