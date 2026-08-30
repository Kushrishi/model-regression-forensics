from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.task import (
    EXP003_SLOT_IDS,
    EXP003D_POLICY,
    EXP003D_POLICY_TEXT,
    EXP003D_SLICE_IDS,
    build_exp003_data,
    build_exp003d_explicit_policy_data,
    sft_examples_sha256,
    write_sft_jsonl,
)

_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})
_ALL_DESCRIPTORS = tuple(
    f"shape={shape},size={size}"
    for shape in ("circle", "square", "triangle")
    for size in ("small", "large")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the deterministic Experiment 003-D explicit-policy diagnostic."
    )
    parser.add_argument("--config", default="configs/exp003d.yaml")
    parser.add_argument("--output", default="artifacts/exp003d/prepared")
    return parser.parse_args()


def _label_counts(examples: tuple) -> dict[str, int]:
    return dict(sorted(Counter(example.response for example in examples).items()))


def _slice_counts(examples: tuple) -> dict[str, int]:
    counts = Counter(example.selected_slice_id for example in examples)
    return {slice_id: counts.get(slice_id, 0) for slice_id in EXP003D_SLICE_IDS}


def _slot_counts(examples: tuple) -> dict[str, int]:
    counts = Counter(example.selected_slot for example in examples)
    return {slot: counts.get(slot, 0) for slot in EXP003_SLOT_IDS}


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    diagnostic = config.capability_diagnostic
    if diagnostic is None or diagnostic.kind != "explicit_policy_role_binding":
        raise ValueError("Experiment 003-D requires explicit_policy_role_binding settings")
    if config.regression.kind != "none":
        raise ValueError("Experiment 003-D is a capability diagnostic, not a regression run")
    if config.training.response_loss_weights is not None:
        raise ValueError("Experiment 003-D must preserve the original unweighted SFT loss")

    expected = {
        "source_experiment_id": "exp003",
        "slot_count": 6,
        "train_example_count": 288,
        "eval_example_count": 96,
        "policy": EXP003D_POLICY,
    }
    observed = diagnostic.model_dump()
    observed.pop("kind")
    if observed != expected:
        raise ValueError(f"Experiment 003-D config does not match frozen design: {observed}")

    source = build_exp003_data(seed=config.seed)
    data = build_exp003d_explicit_policy_data(seed=config.seed)
    output = Path(args.output)
    datasets = output / "datasets"
    write_sft_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_sft_jsonl(data.all_eval, datasets / "all_eval.jsonl")
    for slice_id, examples in data.eval_by_slice.items():
        write_sft_jsonl(examples, datasets / f"{slice_id}_eval.jsonl")

    paired_train = tuple(zip(source.baseline_train, data.baseline_train, strict=True))
    paired_eval = tuple(zip(source.all_eval, data.all_eval, strict=True))
    paired = paired_train + paired_eval

    one_factor_parity_ok = all(
        replace(exp003d, prompt=exp003.prompt) == exp003 for exp003, exp003d in paired
    )
    prompt_prefix_only_ok = all(
        exp003d.prompt == f"{EXP003D_POLICY_TEXT} {exp003.prompt}" for exp003, exp003d in paired
    )

    train_labels = _label_counts(data.baseline_train)
    eval_labels = _label_counts(data.all_eval)
    train_slice_counts = _slice_counts(data.baseline_train)
    eval_slice_counts = _slice_counts(data.all_eval)
    train_slot_counts = _slot_counts(data.baseline_train)
    eval_slot_counts = _slot_counts(data.all_eval)

    all_examples = data.baseline_train + data.all_eval
    policy_response_ok = all(
        EXP003D_POLICY[example.selected_slice_id.split("_", maxsplit=1)[0]] == example.response
        for example in all_examples
    )
    all_descriptors_ok = all(
        all(example.prompt.count(descriptor) == 1 for descriptor in _ALL_DESCRIPTORS)
        for example in all_examples
    )
    public_schema_ok = all(
        frozenset(example.to_sft_record()) == _PUBLIC_FIELDS for example in all_examples
    )
    opaque_ids_ok = all(
        re.fullmatch(r"rec_[0-9a-f]{16}", example.example_id) is not None
        for example in all_examples
    )
    source_train_materials = {example.material for example in source.baseline_train}
    source_eval_materials = {example.material for example in source.all_eval}

    gates = {
        "dataset_counts": len(data.baseline_train) == diagnostic.train_example_count
        and len(data.all_eval) == diagnostic.eval_example_count,
        "one_factor_exp003_parity": one_factor_parity_ok,
        "prompt_change_is_policy_prefix_only": prompt_prefix_only_ok,
        "train_label_distribution_preserved": train_labels == {"ACCEPT": 192, "REJECT": 96},
        "eval_label_distribution_preserved": eval_labels == {"ACCEPT": 64, "REJECT": 32},
        "train_slice_balance": set(train_slice_counts.values()) == {48},
        "eval_slice_balance": set(eval_slice_counts.values()) == {16},
        "train_selected_slot_balance": set(train_slot_counts.values()) == {48},
        "eval_selected_slot_balance": set(eval_slot_counts.values()) == {16},
        "train_eval_materials_disjoint": source_train_materials.isdisjoint(source_eval_materials),
        "all_six_descriptors_once_per_prompt": all_descriptors_ok,
        "explicit_policy_matches_responses": policy_response_ok,
        "public_record_schema": public_schema_ok,
        "opaque_example_ids": opaque_ids_ok,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError("Experiment 003-D construction gates failed: " + ", ".join(failed))

    summary = {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "model": config.model.name,
        "source_experiment_id": diagnostic.source_experiment_id,
        "explicit_policy": diagnostic.policy,
        "counts": {
            "baseline_train": len(data.baseline_train),
            "all_eval": len(data.all_eval),
            "slice_eval": {
                slice_id: len(examples) for slice_id, examples in sorted(data.eval_by_slice.items())
            },
        },
        "balance": {
            "train_labels": train_labels,
            "eval_labels": eval_labels,
            "train_slice_counts": train_slice_counts,
            "eval_slice_counts": eval_slice_counts,
            "train_selected_slot_counts": train_slot_counts,
            "eval_selected_slot_counts": eval_slot_counts,
        },
        "construction_gates": {"all_passed": all(gates.values()), "checks": gates},
        "canonical_sft_record_sha256": {
            "baseline_train": sft_examples_sha256(data.baseline_train),
            "all_eval": sft_examples_sha256(data.all_eval),
            **{
                f"{slice_id}_eval": sft_examples_sha256(examples)
                for slice_id, examples in sorted(data.eval_by_slice.items())
            },
        },
        "source_exp003_sft_record_sha256": {
            "baseline_train": sft_examples_sha256(source.baseline_train),
            "all_eval": sft_examples_sha256(source.all_eval),
        },
        "public_record_fields": sorted(_PUBLIC_FIELDS),
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
