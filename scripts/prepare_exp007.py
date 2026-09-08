from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.exp007 import (
    EXP007_CALIBRATION_MATERIALS,
    EXP007_CERTIFICATION_MATERIALS,
    EXP007_CHANGES_PER_SHARD,
    EXP007_FROZEN_MANIFEST_SHA256,
    EXP007_FROZEN_SEED,
    EXP007_RECORDS_PER_SHARD,
    EXP007_SHARD_IDS,
    EXP007_SLICE_IDS,
    build_exp007_data,
    build_exp007_plan,
)
from model_forensics.lineage import ArtifactChange, LineageManifest
from model_forensics.task import (
    TARGET_SLICE_ID,
    build_exp003d_explicit_policy_data,
    select_exp003_shard,
    sft_examples_sha256,
    write_sft_jsonl,
)

_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_jsonl(path: Path) -> bool:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return all(frozenset(record) == _PUBLIC_FIELDS for record in records)


def _validate_selection(
    path: Path,
    *,
    config_path: Path,
    target_dose: int,
) -> None:
    payload = _load_json(path)

    if payload.get("experiment_id") != "exp007":
        raise ValueError("selection artifact is not from Experiment 007")
    if payload.get("selection_rule") != "minimum_target_dose_meeting_gate":
        raise ValueError("selection artifact uses the wrong selection rule")
    if payload.get("certification_authorized") is not True:
        raise ValueError("selection artifact does not authorize certification")
    if payload.get("selected_target_dose") != target_dose:
        raise ValueError("requested target dose does not match frozen selection")
    if payload.get("manifest_sha256") != EXP007_FROZEN_MANIFEST_SHA256:
        raise ValueError("selection artifact manifest hash mismatch")
    if payload.get("config_sha256") != _sha256(config_path):
        raise ValueError("selection artifact config hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one frozen Experiment 007 calibration or certification world."
    )
    parser.add_argument("--config", default="configs/exp007.yaml")
    parser.add_argument(
        "--phase",
        choices=("calibration", "certification"),
        required=True,
    )
    parser.add_argument("--target-dose", type=int, required=True)
    parser.add_argument("--world-index", type=int, required=True)
    parser.add_argument("--selection")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    if config.experiment_id != "exp007":
        raise ValueError("preparation requires Experiment 007 config")
    if config.seed != EXP007_FROZEN_SEED:
        raise ValueError("Experiment 007 is frozen only for seed 42")
    if config.regression.hidden_root_cause_id is not None:
        raise ValueError("Experiment 007 root must remain private")
    if config.calibration is None:
        raise ValueError("Experiment 007 calibration protocol is missing")

    if args.phase == "calibration":
        if args.selection is not None:
            raise ValueError("calibration must not consume a selection artifact")
    else:
        if args.selection is None:
            raise ValueError("certification preparation requires a frozen calibration selection")
        _validate_selection(
            Path(args.selection),
            config_path=config_path,
            target_dose=args.target_dose,
        )

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")

    data = build_exp007_data(
        seed=config.seed,
        phase=args.phase,
        target_dose=args.target_dose,
        world_index=args.world_index,
    )
    plan = build_exp007_plan(
        seed=config.seed,
        phase=args.phase,
        target_dose=args.target_dose,
        world_index=args.world_index,
    )
    clean = build_exp003d_explicit_policy_data(config.seed)

    expected_materials = (
        set(EXP007_CALIBRATION_MATERIALS)
        if args.phase == "calibration"
        else set(EXP007_CERTIFICATION_MATERIALS)
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
    train_materials = tuple(sorted({example.material for example in data.baseline_train}))

    changes: list[ArtifactChange] = []
    record_counts: dict[str, int] = {}
    changed_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    direction_counts: dict[str, tuple[int, int]] = {}
    material_histograms: dict[str, tuple[int, ...]] = {}
    changed_ids: list[str] = []
    public_schema = True
    serialized_schema = True

    for candidate_id in EXP007_SHARD_IDS:
        before = select_exp003_shard(data.baseline_train, candidate_id)
        after = select_exp003_shard(data.candidate_train, candidate_id)

        before_path = output / "changes" / candidate_id / "before.jsonl"
        after_path = output / "changes" / candidate_id / "after.jsonl"

        write_sft_jsonl(before, before_path)
        write_sft_jsonl(after, after_path)

        changed = [
            baseline_by_id[example.example_id]
            for example in after
            if baseline_by_id[example.example_id].response
            != candidate_by_id[example.example_id].response
        ]

        record_counts[candidate_id] = len(after)
        changed_counts[candidate_id] = len(changed)
        changed_ids.extend(example.example_id for example in changed)

        target_counts[candidate_id] = sum(
            example.selected_slice_id == TARGET_SLICE_ID for example in changed
        )

        direction_counts[candidate_id] = (
            sum(example.response == "ACCEPT" for example in changed),
            sum(example.response == "REJECT" for example in changed),
        )

        material_histograms[candidate_id] = tuple(
            Counter(example.material for example in changed)[material]
            for material in train_materials
        )

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
                    "before_path": str(before_path.relative_to(output)),
                    "after_path": str(after_path.relative_to(output)),
                },
            )
        )

    private_root_gate = target_counts[plan.planted_candidate_id] == args.target_dose and all(
        value == 0
        for candidate_id, value in target_counts.items()
        if candidate_id != plan.planted_candidate_id
    )
    if not private_root_gate:
        raise ValueError("Experiment 007 private target-dose association failed")

    manifest = LineageManifest(
        experiment_id="exp007",
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

    private = output / "private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "world.json").write_text(
        json.dumps(
            {
                "phase": args.phase,
                "target_dose": args.target_dose,
                "world_index": args.world_index,
                "world_seed": plan.world_seed,
                "planted_candidate_id": plan.planted_candidate_id,
                "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    gates = {
        "candidate_count": len(changes) == 5,
        "candidate_record_counts": all(
            value == EXP007_RECORDS_PER_SHARD for value in record_counts.values()
        ),
        "candidate_changed_record_counts": all(
            value == EXP007_CHANGES_PER_SHARD for value in changed_counts.values()
        ),
        "changed_record_nonoverlap": (len(changed_ids) == len(set(changed_ids)) == 180),
        "direction_counts_per_candidate": all(
            value == (24, 12) for value in direction_counts.values()
        ),
        "identical_material_histogram": (len(set(material_histograms.values())) == 1),
        "material_bounds": all(
            min(histogram) >= 2 and max(histogram) <= 4
            for histogram in material_histograms.values()
        ),
        "baseline_label_counts": (
            Counter(example.response for example in data.baseline_train)
            == {"ACCEPT": 192, "REJECT": 96}
        ),
        "candidate_label_counts": (
            Counter(example.response for example in data.candidate_train)
            == {"ACCEPT": 132, "REJECT": 156}
        ),
        "clean_train_parity": (
            [example.to_sft_record() for example in data.baseline_train]
            == [example.to_sft_record() for example in clean.baseline_train]
        ),
        "phase_eval_materials": (
            {example.material for example in data.all_eval} == expected_materials
        ),
        "eval_slice_counts": (
            len(data.all_eval) == 96
            and set(data.eval_by_slice) == set(EXP007_SLICE_IDS)
            and all(len(examples) == 16 for examples in data.eval_by_slice.values())
        ),
        "required_eval_split_config": (
            config.evaluation.baseline_required_splits == [*EXP007_SLICE_IDS, "all"]
        ),
        "public_record_schema": public_schema,
        "serialized_public_record_schema": serialized_schema,
        "opaque_example_ids": all(
            re.fullmatch(r"rec_[0-9a-f]{16}", example.example_id)
            for example in data.baseline_train + data.all_eval
        ),
        "diagnostic_manifest_ground_truth_free": diagnostic_gate,
        "private_target_dose_association": private_root_gate,
    }

    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError("Experiment 007 construction gates failed: " + ", ".join(failed))

    summary = {
        "experiment_id": "exp007",
        "phase": args.phase,
        "target_dose": args.target_dose,
        "world_index": args.world_index,
        "world_identity_redacted": True,
        "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
        "counts": {
            "baseline_train": len(data.baseline_train),
            "candidate_train": len(data.candidate_train),
            "all_eval": len(data.all_eval),
            "observable_changes": len(changes),
            "records_per_change": EXP007_RECORDS_PER_SHARD,
            "label_changes_per_change": EXP007_CHANGES_PER_SHARD,
        },
        "construction_gates": {
            "all_passed": all(gates.values()),
            "checks": gates,
        },
        "canonical_sft_record_sha256": {
            "baseline_train": sft_examples_sha256(data.baseline_train),
            "candidate_train": sft_examples_sha256(data.candidate_train),
            "all_eval": sft_examples_sha256(data.all_eval),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
