from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from model_forensics.exp009_classifier import validate_frozen_development_partition
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_nuisance import (
    build_balanced_cross_intent_refresh,
    nuisance_refresh_audit,
)
from model_forensics.exp009_release import (
    build_clean_release_slots,
    build_symmetric_label_swap_candidate,
    changed_slot_ids,
)

TARGET_A = "Refund_not_showing_up"
TARGET_B = "request_refund"
PER_DIRECTION = 33
MINIMUM_CLEAN_RECALL = 0.90
MINIMUM_EVAL_EXAMPLES = 20
MINIMUM_TRAIN_EXAMPLES = 66

CLEAN_RUN_IDS = {
    0: "clean_t0000_e7_bs32_lr2e5",
    1: "clean_t0001_e7_bs32_lr2e5",
    2: "clean_t0002_e7_bs32_lr2e5",
}


@dataclass(frozen=True)
class ProposedPair:
    label_a: str
    label_b: str
    mean_symmetric_confusion_rate: float
    mutual_confusion_count: int
    lexical_jaccard: float
    minimum_clean_recall: float
    train_count_a: int
    train_count_b: int
    eval_count_a: int
    eval_count_b: int
    a_to_b_count: int
    b_to_a_count: int


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "true_label": str(row["true_label"]),
                "predicted_label": str(row["predicted_label"]),
            }
        )
    return tuple(rows)


def _intent_tokens(label: str) -> frozenset[str]:
    return frozenset(token for token in label.lower().split("_") if token)


def _lexical_jaccard(label_a: str, label_b: str) -> float:
    a_tokens = _intent_tokens(label_a)
    b_tokens = _intent_tokens(label_b)
    union = a_tokens | b_tokens
    return 0.0 if not union else len(a_tokens & b_tokens) / len(union)


def _text_change_count(baseline, candidate) -> int:
    baseline_by_id = {slot.slot_id: slot for slot in baseline}
    candidate_by_id = {slot.slot_id: slot for slot in candidate}
    return sum(
        baseline_by_id[slot_id].text != candidate_by_id[slot_id].text
        for slot_id in changed_slot_ids(baseline, candidate)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only selection audit for proposed Exp009 nuisance rule v2"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            "experiments/009_stochastic_counterfactual_certification/pilot_evidence/clean"
        ),
    )
    args = parser.parse_args()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha256 = validate_frozen_development_partition(partition)

    train_counts = Counter(record.label for record in partition.development_train)
    eval_counts = Counter(record.label for record in partition.development_eval)

    recalls_by_trajectory: dict[int, dict[str, float]] = {}
    predictions_by_trajectory: dict[int, tuple[dict[str, str], ...]] = {}

    for trajectory_id, run_id in CLEAN_RUN_IDS.items():
        root = args.evidence_root / run_id
        summary = _load_json(root / "train_summary.json")
        if summary.get("official_test_split_loaded") is not False:
            raise AssertionError("official-test embargo metadata is not intact")
        if summary.get("development_partition_sha256") != partition_sha256:
            raise AssertionError("development partition hash mismatch")

        metrics = summary["development_eval_metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("development_eval_metrics must be a mapping")
        recalls = metrics["per_label_recall"]
        if not isinstance(recalls, dict):
            raise TypeError("per_label_recall must be a mapping")

        recalls_by_trajectory[trajectory_id] = {
            str(label): float(value) for label, value in recalls.items()
        }
        predictions_by_trajectory[trajectory_id] = _load_jsonl(
            root / "development_eval_predictions.jsonl"
        )

    true_counts: Counter[str] = Counter()
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for rows in predictions_by_trajectory.values():
        for row in rows:
            truth = row["true_label"]
            prediction = row["predicted_label"]
            true_counts[truth] += 1
            if truth != prediction:
                confusion_counts[(truth, prediction)] += 1

    excluded = {TARGET_A, TARGET_B}
    candidate_labels = sorted(label for label in train_counts if label not in excluded)

    eligible: list[ProposedPair] = []
    for label_a, label_b in combinations(candidate_labels, 2):
        if train_counts[label_a] < MINIMUM_TRAIN_EXAMPLES:
            continue
        if train_counts[label_b] < MINIMUM_TRAIN_EXAMPLES:
            continue
        if eval_counts[label_a] < MINIMUM_EVAL_EXAMPLES:
            continue
        if eval_counts[label_b] < MINIMUM_EVAL_EXAMPLES:
            continue

        minimum_clean_recall = min(
            min(recalls[label_a], recalls[label_b]) for recalls in recalls_by_trajectory.values()
        )
        if minimum_clean_recall < MINIMUM_CLEAN_RECALL:
            continue

        lexical_jaccard = _lexical_jaccard(label_a, label_b)
        if lexical_jaccard <= 0.0:
            continue

        a_to_b = confusion_counts[(label_a, label_b)]
        b_to_a = confusion_counts[(label_b, label_a)]
        if a_to_b + b_to_a <= 0:
            continue

        if true_counts[label_a] <= 0 or true_counts[label_b] <= 0:
            raise AssertionError("eligible label missing pooled clean truth count")

        mean_rate = 0.5 * (a_to_b / true_counts[label_a] + b_to_a / true_counts[label_b])

        eligible.append(
            ProposedPair(
                label_a=label_a,
                label_b=label_b,
                mean_symmetric_confusion_rate=mean_rate,
                mutual_confusion_count=a_to_b + b_to_a,
                lexical_jaccard=lexical_jaccard,
                minimum_clean_recall=minimum_clean_recall,
                train_count_a=train_counts[label_a],
                train_count_b=train_counts[label_b],
                eval_count_a=eval_counts[label_a],
                eval_count_b=eval_counts[label_b],
                a_to_b_count=a_to_b,
                b_to_a_count=b_to_a,
            )
        )

    eligible.sort(
        key=lambda pair: (
            -pair.mean_symmetric_confusion_rate,
            -pair.mutual_confusion_count,
            -pair.lexical_jaccard,
            pair.label_a,
            pair.label_b,
        )
    )

    selected: list[ProposedPair] = []
    used_labels: set[str] = set()
    for pair in eligible:
        pair_labels = {pair.label_a, pair.label_b}
        if pair_labels & used_labels:
            continue
        selected.append(pair)
        used_labels.update(pair_labels)
        if len(selected) == 4:
            break

    baseline = build_clean_release_slots(partition.development_train)
    root_candidate = build_symmetric_label_swap_candidate(
        baseline,
        label_a=TARGET_A,
        label_b=TARGET_B,
        per_direction=PER_DIRECTION,
    )

    root_changed = set(changed_slot_ids(baseline, root_candidate))
    root_text_changes = _text_change_count(baseline, root_candidate)

    print("===== EXP009 PROPOSED NUISANCE RULE V2 — READ-ONLY AUDIT =====")
    print(f"development_partition_sha256={partition_sha256}")
    print(f"eligible_pair_count={len(eligible)}")
    print(f"greedy_selected_count={len(selected)}")
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")
    print()

    print("===== ALL ELIGIBLE PAIRS IN PROPOSED RANK ORDER =====")
    for rank, pair in enumerate(eligible, start=1):
        print(
            f"{rank:02d}. {pair.label_a}<->{pair.label_b} "
            f"mean_symmetric_confusion_rate={pair.mean_symmetric_confusion_rate:.8f} "
            f"mutual_confusion_count={pair.mutual_confusion_count} "
            f"a_to_b={pair.a_to_b_count} "
            f"b_to_a={pair.b_to_a_count} "
            f"lexical_jaccard={pair.lexical_jaccard:.6f} "
            f"min_clean_recall={pair.minimum_clean_recall:.6f} "
            f"train=({pair.train_count_a},{pair.train_count_b}) "
            f"eval=({pair.eval_count_a},{pair.eval_count_b})"
        )
    print()

    print("===== GREEDY FOUR =====")
    for index, pair in enumerate(selected, start=1):
        print(
            f"nuisance_{index}={pair.label_a}<->{pair.label_b} "
            f"mean_symmetric_confusion_rate={pair.mean_symmetric_confusion_rate:.8f} "
            f"mutual_confusion_count={pair.mutual_confusion_count} "
            f"a_to_b={pair.a_to_b_count} "
            f"b_to_a={pair.b_to_a_count} "
            f"lexical_jaccard={pair.lexical_jaccard:.6f}"
        )

    if len(selected) != 4:
        print("PROPOSED_V2_SELECTION=FAIL_NOT_FOUR_DISJOINT_PAIRS")
        raise SystemExit(1)

    if len(used_labels) != 8:
        raise AssertionError("selected v2 nuisance pairs are not intent-disjoint")

    print("PROPOSED_V2_SELECTION=PASS_FOUR_DISJOINT_PAIRS")
    print()

    print("===== STRUCTURAL FOOTPRINT AUDIT =====")
    print(f"root_changed_slots={len(root_changed)}")
    print(f"root_text_changes={root_text_changes}")

    nuisance_changed_sets: list[set[str]] = []
    for index, pair in enumerate(selected, start=1):
        candidate, selection = build_balanced_cross_intent_refresh(
            baseline,
            nuisance_index=index,
            label_a=pair.label_a,
            label_b=pair.label_b,
            per_direction=PER_DIRECTION,
        )
        audit = nuisance_refresh_audit(baseline, candidate, selection)
        changed = set(changed_slot_ids(baseline, candidate))

        if changed & root_changed:
            raise AssertionError("selected nuisance overlaps root changed slots")
        if any(changed & prior for prior in nuisance_changed_sets):
            raise AssertionError("selected nuisances overlap changed slots")
        nuisance_changed_sets.append(changed)

        print(
            f"nuisance_{index}_changed_slots={audit['changed_slot_count']} "
            f"text_changes={audit['text_change_count']} "
            f"label_count_delta={audit['aggregate_label_count_delta']} "
            f"duplicates_added="
            f"{audit['duplicate_model_content_occurrences_beyond_first']}"
        )

    print()
    if root_text_changes == 0:
        print("STRUCTURAL_LEAKAGE_ROOT_TEXT_CHANGES=0")
    print("STRUCTURAL_LEAKAGE_NUISANCE_TEXT_CHANGES=66_EACH")
    print("STRUCTURAL_MATCH_FOR_DIAGNOSIS=FAIL")
    print("STRUCTURAL_MATCH_FOR_CERTIFICATION_ONLY=NOT_REQUIRED")
    print()
    print("PROPOSAL_STATUS=NOT_FROZEN")
    print("MODEL_TRAINING_RUN=NO")
    print("OFFICIAL_BANKING77_TEST_TOUCHED=NO")


if __name__ == "__main__":
    main()
