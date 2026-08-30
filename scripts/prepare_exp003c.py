from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.task import (
    EXP003C_EVAL_CONTEXTS,
    EXP003C_SLOT_IDS,
    EXP003C_TRAIN_CONTEXTS,
    build_exp003c_lookup_data,
    sft_examples_sha256,
    write_sft_jsonl,
)

_PUBLIC_FIELDS = frozenset({"example_id", "prompt", "response"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the deterministic Experiment 003-C lookup diagnostic."
    )
    parser.add_argument("--config", default="configs/exp003c.yaml")
    parser.add_argument("--output", default="artifacts/exp003c/prepared")
    return parser.parse_args()


def _label_counts(examples: tuple) -> dict[str, int]:
    return dict(sorted(Counter(example.response for example in examples).items()))


def _selected_slot_counts(examples: tuple) -> dict[str, int]:
    counts = Counter(example.selected_slot for example in examples)
    return {slot: counts.get(slot, 0) for slot in EXP003C_SLOT_IDS}


def _slot_label_counts(examples: tuple) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for slot in EXP003C_SLOT_IDS:
        counts = Counter(example.response for example in examples if example.selected_slot == slot)
        result[slot] = {label: counts.get(label, 0) for label in ("ACCEPT", "REJECT")}
    return result


def _pattern_counts(examples: tuple) -> dict[str, int]:
    return dict(sorted(Counter(example.pattern_id for example in examples).items()))


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    diagnostic = config.capability_diagnostic
    if diagnostic is None or diagnostic.kind != "selected_slot_lookup":
        raise ValueError("Experiment 003-C requires selected_slot_lookup settings")
    if config.regression.kind != "none":
        raise ValueError("Experiment 003-C is a capability diagnostic, not a regression run")
    if config.training.response_loss_weights is not None:
        raise ValueError("Experiment 003-C must use the original unweighted SFT loss")

    expected = {
        "slot_count": 6,
        "accept_per_prompt": 3,
        "train_pattern_count": 16,
        "eval_pattern_count": 4,
        "train_contexts_per_pattern": 3,
        "eval_contexts_per_pattern": 4,
    }
    observed = diagnostic.model_dump()
    observed.pop("kind")
    if observed != expected:
        raise ValueError(f"Experiment 003-C config does not match frozen design: {observed}")

    data = build_exp003c_lookup_data(seed=config.seed)
    output = Path(args.output)
    datasets = output / "datasets"
    write_sft_jsonl(data.baseline_train, datasets / "baseline_train.jsonl")
    write_sft_jsonl(data.all_eval, datasets / "all_eval.jsonl")
    for slot, examples in data.eval_by_slot.items():
        write_sft_jsonl(examples, datasets / f"{slot}_eval.jsonl")

    train_pattern_set = set(data.train_patterns)
    eval_pattern_set = set(data.eval_patterns)
    all_patterns = train_pattern_set | eval_pattern_set
    train_labels = _label_counts(data.baseline_train)
    eval_labels = _label_counts(data.all_eval)
    train_slot_counts = _selected_slot_counts(data.baseline_train)
    eval_slot_counts = _selected_slot_counts(data.all_eval)
    train_slot_labels = _slot_label_counts(data.baseline_train)
    eval_slot_labels = _slot_label_counts(data.all_eval)
    train_pattern_counts = _pattern_counts(data.baseline_train)
    eval_pattern_counts = _pattern_counts(data.all_eval)

    prompt_balance_ok = all(
        example.prompt.count("decision=ACCEPT") == diagnostic.accept_per_prompt
        and example.prompt.count("decision=REJECT")
        == diagnostic.slot_count - diagnostic.accept_per_prompt
        for example in data.baseline_train + data.all_eval
    )
    public_schema_ok = all(
        frozenset(example.to_sft_record()) == _PUBLIC_FIELDS
        for example in data.baseline_train + data.all_eval
    )
    opaque_ids_ok = all(
        re.fullmatch(r"rec_[0-9a-f]{16}", example.example_id) is not None
        for example in data.baseline_train + data.all_eval
    )

    gates = {
        "twenty_unique_balanced_patterns": len(all_patterns) == 20
        and all(len(pattern) == 3 for pattern in all_patterns),
        "pattern_split_counts": len(train_pattern_set) == diagnostic.train_pattern_count
        and len(eval_pattern_set) == diagnostic.eval_pattern_count,
        "held_out_patterns_disjoint": train_pattern_set.isdisjoint(eval_pattern_set),
        "dataset_counts": len(data.baseline_train) == 288 and len(data.all_eval) == 96,
        "train_label_balance": train_labels == {"ACCEPT": 144, "REJECT": 144},
        "eval_label_balance": eval_labels == {"ACCEPT": 48, "REJECT": 48},
        "within_prompt_balance": prompt_balance_ok,
        "train_selected_slot_balance": set(train_slot_counts.values()) == {48},
        "eval_selected_slot_balance": set(eval_slot_counts.values()) == {16},
        "train_slot_label_balance": all(
            counts == {"ACCEPT": 24, "REJECT": 24} for counts in train_slot_labels.values()
        ),
        "eval_slot_label_balance": all(
            counts == {"ACCEPT": 8, "REJECT": 8} for counts in eval_slot_labels.values()
        ),
        "context_vocabularies_disjoint": set(EXP003C_TRAIN_CONTEXTS).isdisjoint(
            EXP003C_EVAL_CONTEXTS
        ),
        "pattern_record_balance": set(train_pattern_counts.values()) == {18}
        and set(eval_pattern_counts.values()) == {24},
        "public_record_schema": public_schema_ok,
        "opaque_example_ids": opaque_ids_ok,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError("Experiment 003-C construction gates failed: " + ", ".join(failed))

    summary = {
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "model": config.model.name,
        "counts": {
            "baseline_train": len(data.baseline_train),
            "all_eval": len(data.all_eval),
            "train_patterns": len(data.train_patterns),
            "eval_patterns": len(data.eval_patterns),
            "slot_eval": {slot: len(examples) for slot, examples in data.eval_by_slot.items()},
        },
        "balance": {
            "train_labels": train_labels,
            "eval_labels": eval_labels,
            "train_selected_slot_counts": train_slot_counts,
            "eval_selected_slot_counts": eval_slot_counts,
            "train_slot_label_counts": train_slot_labels,
            "eval_slot_label_counts": eval_slot_labels,
            "train_pattern_record_counts": train_pattern_counts,
            "eval_pattern_record_counts": eval_pattern_counts,
        },
        "pattern_split": {
            "train": ["".join(slot[-1] for slot in pattern) for pattern in data.train_patterns],
            "eval": ["".join(slot[-1] for slot in pattern) for pattern in data.eval_patterns],
        },
        "construction_gates": {"all_passed": all(gates.values()), "checks": gates},
        "canonical_sft_record_sha256": {
            "baseline_train": sft_examples_sha256(data.baseline_train),
            "all_eval": sft_examples_sha256(data.all_eval),
            **{
                f"{slot}_eval": sft_examples_sha256(examples)
                for slot, examples in sorted(data.eval_by_slot.items())
            },
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
