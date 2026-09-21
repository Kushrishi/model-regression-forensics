from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from model_forensics.exp009_classifier import validate_frozen_development_partition
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)

TARGET_A = "Refund_not_showing_up"
TARGET_B = "request_refund"
EXCLUDED_LABELS = frozenset({TARGET_A, TARGET_B})
MINIMUM_CLEAN_RECALL = 0.90
MINIMUM_EVAL_EXAMPLES = 20
MINIMUM_TRAIN_EXAMPLES = 66

CLEAN_RUN_IDS = {
    0: "clean_t0000_e7_bs32_lr2e5",
    1: "clean_t0001_e7_bs32_lr2e5",
    2: "clean_t0002_e7_bs32_lr2e5",
}


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
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def _greedy_disjoint_count(pairs: list[tuple[str, str]]) -> int:
    selected = 0
    used: set[str] = set()
    for label_a, label_b in sorted(pairs):
        if label_a in used or label_b in used:
            continue
        used.add(label_a)
        used.add(label_b)
        selected += 1
    return selected


def _pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}<->{pair[1]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only autopsy of the frozen Exp009 nuisance-selection rule v1"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
        help="Pinned Banking77 train cache. Downloaded from the frozen source if absent.",
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
    candidate_labels = sorted(label for label in train_counts if label not in EXCLUDED_LABELS)
    all_pairs = list(combinations(candidate_labels, 2))

    recall_by_trajectory: dict[int, dict[str, float]] = {}
    predictions_by_trajectory: dict[int, tuple[dict[str, str], ...]] = {}

    for trajectory_id, run_id in CLEAN_RUN_IDS.items():
        run_root = args.evidence_root / run_id
        summary = _load_json(run_root / "train_summary.json")

        if summary.get("official_test_split_loaded") is not False:
            raise AssertionError(
                f"trajectory {trajectory_id}: official-test embargo metadata is not intact"
            )
        if summary.get("development_partition_sha256") != partition_sha256:
            raise AssertionError(f"trajectory {trajectory_id}: development partition hash mismatch")

        metrics = summary.get("development_eval_metrics")
        if not isinstance(metrics, dict):
            raise TypeError("development_eval_metrics must be a mapping")
        recalls = metrics.get("per_label_recall")
        if not isinstance(recalls, dict):
            raise TypeError("per_label_recall must be a mapping")

        recall_by_trajectory[trajectory_id] = {
            str(label): float(value) for label, value in recalls.items()
        }

        rows = _load_jsonl(run_root / "development_eval_predictions.jsonl")
        if len(rows) != len(partition.development_eval):
            raise AssertionError(
                f"trajectory {trajectory_id}: prediction row count mismatch "
                f"expected={len(partition.development_eval)} observed={len(rows)}"
            )
        predictions_by_trajectory[trajectory_id] = rows

    true_counts: Counter[str] = Counter()
    confusion_counts: Counter[tuple[str, str]] = Counter()
    for rows in predictions_by_trajectory.values():
        for row in rows:
            truth = row["true_label"]
            prediction = row["predicted_label"]
            true_counts[truth] += 1
            if truth != prediction:
                confusion_counts[(truth, prediction)] += 1

    def train_gate(pair: tuple[str, str]) -> bool:
        a, b = pair
        return train_counts[a] >= MINIMUM_TRAIN_EXAMPLES and train_counts[b] >= MINIMUM_TRAIN_EXAMPLES

    def eval_gate(pair: tuple[str, str]) -> bool:
        a, b = pair
        return eval_counts[a] >= MINIMUM_EVAL_EXAMPLES and eval_counts[b] >= MINIMUM_EVAL_EXAMPLES

    def recall_gate(pair: tuple[str, str]) -> bool:
        a, b = pair
        minimum = min(min(recalls[a], recalls[b]) for recalls in recall_by_trajectory.values())
        return minimum >= MINIMUM_CLEAN_RECALL

    def lexical_gate(pair: tuple[str, str]) -> bool:
        return _lexical_jaccard(*pair) > 0.0

    def any_confusion_gate(pair: tuple[str, str]) -> bool:
        a, b = pair
        return confusion_counts[(a, b)] > 0 or confusion_counts[(b, a)] > 0

    def bidirectional_confusion_gate(pair: tuple[str, str]) -> bool:
        a, b = pair
        return confusion_counts[(a, b)] > 0 and confusion_counts[(b, a)] > 0

    gates = [
        ("train>=66_both", train_gate),
        ("eval>=20_both", eval_gate),
        ("min_clean_recall>=0.90_both", recall_gate),
        ("lexical_jaccard>0", lexical_gate),
        ("any_direction_confusion", any_confusion_gate),
        ("bidirectional_confusion", bidirectional_confusion_gate),
    ]

    individual: dict[str, list[tuple[str, str]]] = {
        name: [pair for pair in all_pairs if gate(pair)] for name, gate in gates
    }

    frozen_cumulative_order = [
        ("train>=66_both", train_gate),
        ("eval>=20_both", eval_gate),
        ("min_clean_recall>=0.90_both", recall_gate),
        ("lexical_jaccard>0", lexical_gate),
        ("bidirectional_confusion", bidirectional_confusion_gate),
    ]

    cumulative: dict[str, list[tuple[str, str]]] = {}
    survivors = list(all_pairs)
    for name, gate in frozen_cumulative_order:
        survivors = [pair for pair in survivors if gate(pair)]
        cumulative[name] = list(survivors)

    pre_confusion = cumulative["lexical_jaccard>0"]
    pre_confusion_any = [pair for pair in pre_confusion if any_confusion_gate(pair)]
    pre_confusion_bidirectional = [
        pair for pair in pre_confusion if bidirectional_confusion_gate(pair)
    ]

    recall_qualified = cumulative["min_clean_recall>=0.90_both"]
    recall_plus_any_confusion = [pair for pair in recall_qualified if any_confusion_gate(pair)]
    recall_plus_bidirectional = [
        pair for pair in recall_qualified if bidirectional_confusion_gate(pair)
    ]
    recall_plus_lexical = [pair for pair in recall_qualified if lexical_gate(pair)]

    lexical_pairs = individual["lexical_jaccard>0"]
    lexical_plus_any = [pair for pair in lexical_pairs if any_confusion_gate(pair)]
    lexical_plus_bidirectional = [
        pair for pair in lexical_pairs if bidirectional_confusion_gate(pair)
    ]

    def count_labels(pairs: list[tuple[str, str]]) -> int:
        return len({label for pair in pairs for label in pair})

    def print_stage(prefix: str, pairs: list[tuple[str, str]]) -> None:
        print(
            f"{prefix}: pairs={len(pairs)} "
            f"participating_labels={count_labels(pairs)} "
            f"lexicographic_greedy_disjoint={_greedy_disjoint_count(pairs)}"
        )

    print("===== EXP009 NUISANCE RULE V1 AUTOPSY =====")
    print(f"development_partition_sha256={partition_sha256}")
    print(f"candidate_labels_after_target_exclusion={len(candidate_labels)}")
    print(f"all_candidate_pairs={len(all_pairs)}")
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")
    print()

    print("===== INDIVIDUAL GATE COUNTS =====")
    for name, _ in gates:
        print_stage(name, individual[name])
    print()

    print("===== FROZEN CUMULATIVE ORDER =====")
    for name, _ in frozen_cumulative_order:
        print_stage(name, cumulative[name])
    print()

    print("===== DIAGNOSTIC INTERSECTIONS =====")
    print_stage("recall_qualified", recall_qualified)
    print_stage("recall_plus_lexical", recall_plus_lexical)
    print_stage("recall_plus_any_direction_confusion", recall_plus_any_confusion)
    print_stage("recall_plus_bidirectional_confusion", recall_plus_bidirectional)
    print_stage("lexical_plus_any_direction_confusion", lexical_plus_any)
    print_stage("lexical_plus_bidirectional_confusion", lexical_plus_bidirectional)
    print_stage("frozen_pre_confusion_plus_any_direction", pre_confusion_any)
    print_stage("frozen_pre_confusion_plus_bidirectional", pre_confusion_bidirectional)
    print()

    print("===== TOP NEAR-MISS PAIRS AFTER COUNT + RECALL GATES =====")
    near_misses = []
    for pair in recall_qualified:
        a, b = pair
        jaccard = _lexical_jaccard(a, b)
        a_to_b = confusion_counts[(a, b)]
        b_to_a = confusion_counts[(b, a)]
        near_misses.append(
            (
                -(a_to_b + b_to_a),
                -jaccard,
                a,
                b,
                a_to_b,
                b_to_a,
                jaccard,
            )
        )

    for (
        _negative_mutual,
        _negative_jaccard,
        a,
        b,
        a_to_b,
        b_to_a,
        jaccard,
    ) in sorted(near_misses)[:25]:
        print(
            f"{a}<->{b} "
            f"lexical_jaccard={jaccard:.6f} "
            f"a_to_b={a_to_b} "
            f"b_to_a={b_to_a} "
            f"any_confusion={'YES' if (a_to_b or b_to_a) else 'NO'} "
            f"bidirectional={'YES' if (a_to_b and b_to_a) else 'NO'}"
        )
    print()

    final_pairs = cumulative["bidirectional_confusion"]
    print("===== FROZEN RULE RESULT =====")
    print(f"eligible_pair_count={len(final_pairs)}")
    print(
        "frozen_rule_status="
        + ("ELIGIBLE_PAIRS_PRESENT" if final_pairs else "FROZEN_PILOT_RULE_INFEASIBLE")
    )
    if final_pairs:
        print("eligible_pairs=" + ",".join(_pair_name(pair) for pair in final_pairs))
    print("model_training_performed=NO")
    print("official_test_split_loaded=NO")


if __name__ == "__main__":
    main()
