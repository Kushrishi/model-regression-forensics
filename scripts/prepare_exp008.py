from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.exp008 import (
    EXP008_CHANGES_PER_SHARD,
    EXP008_FROZEN_MANIFEST_SHA256,
    EXP008_FROZEN_SEED,
    EXP008_RECORDS_PER_SHARD,
    EXP008_SHARD_IDS,
    EXP008_WORLD_COUNT,
    build_exp008_data,
    build_exp008_plan,
    exp008_internal_examples_sha256,
    summarize_exp008_interventions,
)
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    select_exp003_shard,
    sft_examples_sha256,
    write_sft_jsonl,
)

_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})
_MANIFEST_PATH = Path("src/model_forensics/data/exp008_frozen_worlds.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _public_jsonl(path: Path) -> bool:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return all(frozenset(record) == _PUBLIC_FIELDS for record in records)


def _frozen_world(
    *,
    config_path: Path,
    world_index: int,
) -> dict[str, Any]:
    if _sha256(_MANIFEST_PATH) != EXP008_FROZEN_MANIFEST_SHA256:
        raise ValueError("Experiment 008 frozen manifest hash mismatch")

    manifest = _load_json(_MANIFEST_PATH)

    if manifest.get("schema_version") != 1:
        raise ValueError("Experiment 008 manifest schema mismatch")
    if manifest.get("experiment_id") != "exp008":
        raise ValueError("Experiment 008 manifest identity mismatch")
    if manifest.get("seed") != EXP008_FROZEN_SEED:
        raise ValueError("Experiment 008 manifest seed mismatch")
    if manifest.get("world_count") != EXP008_WORLD_COUNT:
        raise ValueError("Experiment 008 manifest world-count mismatch")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise ValueError("Experiment 008 frozen config hash mismatch")

    worlds = manifest.get("worlds")
    if not isinstance(worlds, dict):
        raise ValueError("Experiment 008 manifest worlds are malformed")

    world = worlds.get(f"world_{world_index}")
    if not isinstance(world, dict):
        raise ValueError("Experiment 008 requested world is not frozen")

    return world


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one frozen Experiment 008 world.")
    parser.add_argument("--config", default="configs/exp008.yaml")
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not 0 <= args.world_index < EXP008_WORLD_COUNT:
        raise ValueError("Experiment 008 world index is outside frozen range")

    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    if config.experiment_id != "exp008":
        raise ValueError("preparation requires Experiment 008 config")
    if config.seed != EXP008_FROZEN_SEED:
        raise ValueError("Experiment 008 is frozen only for seed 42")

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    frozen = _frozen_world(
        config_path=config_path,
        world_index=args.world_index,
    )

    plan = build_exp008_plan(
        seed=config.seed,
        world_index=args.world_index,
    )
    data = build_exp008_data(
        seed=config.seed,
        world_index=args.world_index,
    )
    audit = summarize_exp008_interventions(data)

    if frozen.get("world_index") != args.world_index:
        raise ValueError("Experiment 008 frozen world index mismatch")
    if frozen.get("world_seed") != plan.world_seed:
        raise ValueError("Experiment 008 frozen world seed mismatch")

    expected_sft = frozen.get("dataset_sft_sha256")
    if not isinstance(expected_sft, dict):
        raise ValueError("Experiment 008 frozen SFT hashes are malformed")

    actual_sft = {
        "baseline_train": sft_examples_sha256(data.baseline_train),
        "candidate_train": sft_examples_sha256(data.candidate_train),
        "target_eval": sft_examples_sha256(data.target_eval),
        "control_eval": sft_examples_sha256(data.control_eval),
        "all_eval": sft_examples_sha256(data.all_eval),
    }

    if actual_sft != expected_sft:
        raise ValueError("Experiment 008 generated SFT data differs from frozen world")

    expected_internal = frozen.get("dataset_internal_sha256")
    if not isinstance(expected_internal, dict):
        raise ValueError("Experiment 008 frozen internal hashes are malformed")

    actual_internal = {
        "baseline_train": exp008_internal_examples_sha256(data.baseline_train),
        "candidate_train": exp008_internal_examples_sha256(data.candidate_train),
    }

    if actual_internal != expected_internal:
        raise ValueError("Experiment 008 generated internal data differs from frozen world")

    datasets = output / "datasets"

    write_sft_jsonl(
        data.baseline_train,
        datasets / "baseline_train.jsonl",
    )
    write_sft_jsonl(
        data.candidate_train,
        datasets / "candidate_train.jsonl",
    )
    write_sft_jsonl(
        data.target_eval,
        datasets / "target_eval.jsonl",
    )
    write_sft_jsonl(
        data.control_eval,
        datasets / "control_eval.jsonl",
    )
    write_sft_jsonl(
        data.all_eval,
        datasets / "all_eval.jsonl",
    )

    for slice_id, examples in data.eval_by_slice.items():
        write_sft_jsonl(
            examples,
            datasets / f"{slice_id}_eval.jsonl",
        )

    baseline_by_id = {example.example_id: example for example in data.baseline_train}
    candidate_by_id = {example.example_id: example for example in data.candidate_train}

    changes: list[ArtifactChange] = []
    changed_ids: list[str] = []
    changed_counts: dict[str, int] = {}
    record_counts: dict[str, int] = {}
    public_schema = True
    serialized_schema = True

    frozen_changed = frozen.get("changed_ids_by_candidate")
    if not isinstance(frozen_changed, dict):
        raise ValueError("Experiment 008 frozen changed-ID map is malformed")

    for candidate_id in EXP008_SHARD_IDS:
        before = select_exp003_shard(
            data.baseline_train,
            candidate_id,
        )
        after = select_exp003_shard(
            data.candidate_train,
            candidate_id,
        )

        before_path = output / "changes" / candidate_id / "before.jsonl"
        after_path = output / "changes" / candidate_id / "after.jsonl"

        write_sft_jsonl(before, before_path)
        write_sft_jsonl(after, after_path)

        changed = [
            baseline_by_id[example.example_id]
            for example in after
            if baseline_by_id[example.example_id].to_sft_record()
            != candidate_by_id[example.example_id].to_sft_record()
        ]

        actual_changed_ids = sorted(example.example_id for example in changed)

        if actual_changed_ids != frozen_changed.get(candidate_id):
            raise ValueError(f"Experiment 008 changed IDs differ from frozen world: {candidate_id}")

        record_counts[candidate_id] = len(after)
        changed_counts[candidate_id] = len(changed)
        changed_ids.extend(actual_changed_ids)

        public_schema = public_schema and all(
            frozenset(example.to_sft_record()) == _PUBLIC_FIELDS for example in before + after
        )
        serialized_schema = (
            serialized_schema and _public_jsonl(before_path) and _public_jsonl(after_path)
        )

        changes.append(
            ArtifactChange(
                change_id=candidate_id,
                kind="dataset_shard",
                description=("SFT shard content differs between baseline and candidate."),
                before=f"sha256:{sft_examples_sha256(before)}",
                after=f"sha256:{sft_examples_sha256(after)}",
                metadata={
                    "record_count": len(after),
                    "changed_record_count": len(changed),
                    "before_path": str(before_path.relative_to(output)),
                    "after_path": str(after_path.relative_to(output)),
                },
            )
        )

    manifest = LineageManifest(
        experiment_id="exp008",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        hidden_root_cause_id=plan.planted_candidate_id,
        changes=changes,
    )

    manifest.dump(output / "lineage" / "benchmark.json")
    manifest.redacted().dump(output / "lineage" / "diagnostic.json")

    diagnostic = _load_json(output / "lineage" / "diagnostic.json")

    diagnostic_gate = "hidden_root_cause_id" not in diagnostic and set(diagnostic) == {
        "experiment_id",
        "baseline_run_id",
        "candidate_run_id",
        "changes",
    }

    construction = {
        "candidate_count": len(changes) == 5,
        "candidate_record_counts": all(
            value == EXP008_RECORDS_PER_SHARD for value in record_counts.values()
        ),
        "candidate_changed_record_counts": all(
            value == EXP008_CHANGES_PER_SHARD for value in changed_counts.values()
        ),
        "changed_record_nonoverlap": (
            len(changed_ids) == len(set(changed_ids)) == 5 * EXP008_CHANGES_PER_SHARD
        ),
        "global_prompt_multiset_preserved": (audit["global_prompt_multiset_preserved"] is True),
        "baseline_unique_prompts": (audit["baseline_unique_prompt_count"] == 288),
        "candidate_unique_prompts": (audit["candidate_unique_prompt_count"] == 288),
        "only_target_policy_inconsistent": (
            audit["aggregate_policy_inconsistent_by_slice"] == {"triangle_large": 36}
        ),
        "public_training_schema": public_schema,
        "serialized_training_schema": serialized_schema,
        "diagnostic_manifest_ground_truth_free": diagnostic_gate,
        "frozen_sft_hashes_match": actual_sft == expected_sft,
        "frozen_internal_hashes_match": (actual_internal == expected_internal),
    }

    construction["all_passed"] = all(construction.values())

    if not construction["all_passed"]:
        failed = [key for key, value in construction.items() if value is not True]
        raise ValueError(f"Experiment 008 construction gates failed: {failed}")

    private = output / "private"
    private.mkdir(parents=True, exist_ok=True)

    (private / "world.json").write_text(
        json.dumps(
            {
                "world_index": args.world_index,
                "world_seed": plan.world_seed,
                "planted_candidate_id": plan.planted_candidate_id,
                "manifest_sha256": EXP008_FROZEN_MANIFEST_SHA256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (private / "construction_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment_id": "exp008",
        "status": "prepared_model_not_run",
        "world_index": args.world_index,
        "world_seed_sha256": hashlib.sha256(str(plan.world_seed).encode()).hexdigest(),
        "world_identity_redacted": True,
        "manifest_sha256": EXP008_FROZEN_MANIFEST_SHA256,
        "config_sha256": _sha256(config_path),
        "dataset_sft_sha256": actual_sft,
        "dataset_internal_sha256": actual_internal,
        "construction_gates": construction,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
