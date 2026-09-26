from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from model_forensics.exp009_candidates import build_opaque_candidate_manifests
from model_forensics.exp009_data import DevelopmentPartition
from model_forensics.exp009_release import (
    Exp009ReleaseSlot,
    build_symmetric_label_swap_candidate,
    changed_slot_ids,
    release_diff_manifest,
    release_sha256,
    restore_release_slots,
)

MATCHED_BENCHMARK_NAMESPACE = "exp009-structurally-matched-benchmark-v1"
MATCHED_PER_DIRECTION = 33
MATCHED_CHANGED_SLOT_COUNT = 66
MATCHED_WORLD_COUNT = 2
MATCHED_CANDIDATES_PER_WORLD = 5


@dataclass(frozen=True)
class MatchedPairScore:
    """Clean-only evidence for one structurally matched candidate pair."""

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


@dataclass(frozen=True)
class MatchedWorldDefinition:
    """Prospectively selected intent-pair set for one matched benchmark world."""

    world_index: int
    pairs: tuple[MatchedPairScore, ...]
    root_position: int

    @property
    def root_pair(self) -> MatchedPairScore:
        return self.pairs[self.root_position]


@dataclass(frozen=True)
class BuiltMatchedWorld:
    """Fully audited release objects and manifests for one world."""

    definition: MatchedWorldDefinition
    baseline: tuple[Exp009ReleaseSlot, ...]
    candidate_releases: tuple[tuple[Exp009ReleaseSlot, ...], ...]
    composite: tuple[Exp009ReleaseSlot, ...]
    restorations: tuple[tuple[Exp009ReleaseSlot, ...], ...]
    diagnostic_manifest: dict[str, object]
    truth_manifest: dict[str, object]
    structural_audit: dict[str, object]


def _intent_tokens(label: str) -> frozenset[str]:
    return frozenset(token for token in label.lower().split("_") if token)


def _lexical_jaccard(label_a: str, label_b: str) -> float:
    a_tokens = _intent_tokens(label_a)
    b_tokens = _intent_tokens(label_b)
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def rank_matched_pairs(
    partition: DevelopmentPartition,
    *,
    per_label_recall_by_trajectory: dict[int, dict[str, float]],
    prediction_rows_by_trajectory: dict[int, tuple[dict[str, str], ...]],
    excluded_labels: frozenset[str] = frozenset(),
    minimum_clean_recall: float = 0.90,
    minimum_eval_examples: int = 20,
    minimum_train_examples: int = MATCHED_CHANGED_SLOT_COUNT,
) -> tuple[MatchedPairScore, ...]:
    """Rank structurally matched pair candidates from clean development evidence."""

    if not per_label_recall_by_trajectory:
        raise ValueError("at least one clean trajectory is required")
    if set(per_label_recall_by_trajectory) != set(prediction_rows_by_trajectory):
        raise ValueError("clean metric and prediction trajectory IDs must match")

    train_counts = Counter(record.label for record in partition.development_train)
    eval_counts = Counter(record.label for record in partition.development_eval)
    labels = sorted(train_counts)

    for trajectory_id, recalls in per_label_recall_by_trajectory.items():
        missing = set(labels) - set(recalls)
        if missing:
            raise ValueError(
                f"trajectory {trajectory_id} is missing per-label recall for "
                f"{sorted(missing)!r}"
            )

    true_counts: Counter[str] = Counter()
    confusion_counts: Counter[tuple[str, str]] = Counter()
    for rows in prediction_rows_by_trajectory.values():
        for row in rows:
            truth = row["true_label"]
            prediction = row["predicted_label"]
            true_counts[truth] += 1
            if truth != prediction:
                confusion_counts[(truth, prediction)] += 1

    eligible: list[MatchedPairScore] = []
    candidate_labels = [label for label in labels if label not in excluded_labels]
    for label_a, label_b in combinations(candidate_labels, 2):
        if train_counts[label_a] < minimum_train_examples:
            continue
        if train_counts[label_b] < minimum_train_examples:
            continue
        if eval_counts[label_a] < minimum_eval_examples:
            continue
        if eval_counts[label_b] < minimum_eval_examples:
            continue

        minimum_recall = min(
            min(recalls[label_a], recalls[label_b])
            for recalls in per_label_recall_by_trajectory.values()
        )
        if minimum_recall < minimum_clean_recall:
            continue

        lexical_jaccard = _lexical_jaccard(label_a, label_b)
        a_to_b = confusion_counts[(label_a, label_b)]
        b_to_a = confusion_counts[(label_b, label_a)]
        mutual_confusion = a_to_b + b_to_a
        if mutual_confusion <= 0 and lexical_jaccard <= 0.0:
            continue

        if true_counts[label_a] <= 0 or true_counts[label_b] <= 0:
            raise ValueError("eligible matched label has no pooled clean predictions")

        mean_symmetric_rate = 0.5 * (
            a_to_b / true_counts[label_a] + b_to_a / true_counts[label_b]
        )
        eligible.append(
            MatchedPairScore(
                label_a=label_a,
                label_b=label_b,
                mean_symmetric_confusion_rate=mean_symmetric_rate,
                mutual_confusion_count=mutual_confusion,
                lexical_jaccard=lexical_jaccard,
                minimum_clean_recall=minimum_recall,
                train_count_a=train_counts[label_a],
                train_count_b=train_counts[label_b],
                eval_count_a=eval_counts[label_a],
                eval_count_b=eval_counts[label_b],
                a_to_b_count=a_to_b,
                b_to_a_count=b_to_a,
            )
        )

    return tuple(
        sorted(
            eligible,
            key=lambda score: (
                -score.mean_symmetric_confusion_rate,
                -score.mutual_confusion_count,
                -score.lexical_jaccard,
                -score.minimum_clean_recall,
                score.label_a,
                score.label_b,
            ),
        )
    )


def _root_position(world_index: int) -> int:
    if world_index < 0:
        raise ValueError("world_index must be non-negative")
    digest = hashlib.sha256(
        f"exp009-matched-root-v1|world={world_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big") % MATCHED_CANDIDATES_PER_WORLD


def select_matched_worlds(
    ranked_pairs: tuple[MatchedPairScore, ...],
    *,
    world_count: int = MATCHED_WORLD_COUNT,
    candidates_per_world: int = MATCHED_CANDIDATES_PER_WORLD,
) -> tuple[MatchedWorldDefinition, ...]:
    """Select globally label-disjoint worlds from the frozen clean-only ranking."""

    if world_count <= 0:
        raise ValueError("world_count must be positive")
    if candidates_per_world != MATCHED_CANDIDATES_PER_WORLD:
        raise ValueError(
            "v1 matched benchmark requires exactly five candidates per world"
        )

    used_labels: set[str] = set()
    worlds: list[MatchedWorldDefinition] = []
    for world_index in range(world_count):
        selected: list[MatchedPairScore] = []
        world_labels: set[str] = set()
        for pair in ranked_pairs:
            labels = {pair.label_a, pair.label_b}
            if labels & used_labels or labels & world_labels:
                continue
            selected.append(pair)
            world_labels.update(labels)
            if len(selected) == candidates_per_world:
                break

        if len(selected) != candidates_per_world:
            raise ValueError(
                f"world {world_index}: fewer than {candidates_per_world} "
                "globally disjoint eligible pairs remain"
            )

        used_labels.update(world_labels)
        worlds.append(
            MatchedWorldDefinition(
                world_index=world_index,
                pairs=tuple(selected),
                root_position=_root_position(world_index),
            )
        )

    return tuple(worlds)


def _text_change_count(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
) -> int:
    baseline_by_id = {slot.slot_id: slot for slot in baseline}
    candidate_by_id = {slot.slot_id: slot for slot in candidate}
    return sum(
        baseline_by_id[slot_id].text != candidate_by_id[slot_id].text
        for slot_id in changed_slot_ids(baseline, candidate)
    )


def _aggregate_label_delta(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
) -> dict[str, int]:
    delta = Counter(slot.label for slot in candidate)
    delta.subtract(Counter(slot.label for slot in baseline))
    return {label: count for label, count in sorted(delta.items()) if count}


def _combine_disjoint_changes(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidates: Iterable[tuple[Exp009ReleaseSlot, ...]],
) -> tuple[Exp009ReleaseSlot, ...]:
    baseline_by_id = {slot.slot_id: slot for slot in baseline}
    replacements: dict[str, Exp009ReleaseSlot] = {}
    for candidate in candidates:
        candidate_by_id = {slot.slot_id: slot for slot in candidate}
        if set(candidate_by_id) != set(baseline_by_id):
            raise ValueError("candidate release is not aligned to baseline slots")
        for slot_id in changed_slot_ids(baseline, candidate):
            if slot_id in replacements:
                raise ValueError(f"candidate changes overlap at stable slot {slot_id}")
            replacements[slot_id] = candidate_by_id[slot_id]

    return tuple(
        replacements.get(slot.slot_id, slot)
        for slot in baseline
    )


def _candidate_structural_row(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
    pair: MatchedPairScore,
) -> dict[str, object]:
    manifest = release_diff_manifest(baseline, candidate)
    transitions = manifest["label_transitions"]
    expected_transitions = {
        f"{pair.label_a}->{pair.label_b}": MATCHED_PER_DIRECTION,
        f"{pair.label_b}->{pair.label_a}": MATCHED_PER_DIRECTION,
    }
    return {
        "changed_slot_count": manifest["changed_slot_count"],
        "text_change_count": _text_change_count(baseline, candidate),
        "label_change_count": manifest["changed_slot_count"],
        "aggregate_label_count_delta": _aggregate_label_delta(baseline, candidate),
        "touched_label_count": 2,
        "transition_counts_match_33_33": transitions == expected_transitions,
        "slot_count": manifest["slot_count"],
        "manifest_field_set": sorted(manifest),
    }


def build_matched_world(
    baseline: tuple[Exp009ReleaseSlot, ...],
    definition: MatchedWorldDefinition,
) -> BuiltMatchedWorld:
    """Build and structurally audit one five-candidate matched benchmark world."""

    if len(definition.pairs) != MATCHED_CANDIDATES_PER_WORLD:
        raise ValueError("matched world must contain exactly five candidate pairs")

    labels = [
        label
        for pair in definition.pairs
        for label in (pair.label_a, pair.label_b)
    ]
    if len(labels) != len(set(labels)):
        raise ValueError("matched world candidate intent labels must be disjoint")

    candidates = tuple(
        build_symmetric_label_swap_candidate(
            baseline,
            label_a=pair.label_a,
            label_b=pair.label_b,
            per_direction=MATCHED_PER_DIRECTION,
        )
        for pair in definition.pairs
    )
    changed_sets = [set(changed_slot_ids(baseline, candidate)) for candidate in candidates]
    for i, left in enumerate(changed_sets):
        for right in changed_sets[i + 1 :]:
            if left & right:
                raise AssertionError("matched candidate changed-slot sets overlap")

    composite = _combine_disjoint_changes(baseline, candidates)
    reverse_composite = _combine_disjoint_changes(baseline, reversed(candidates))
    if reverse_composite != composite:
        raise AssertionError("matched composite depends on candidate application order")

    individual_restorations_exact = all(
        restore_release_slots(
            candidate,
            baseline,
            restore_slot_ids=changed_slot_ids(baseline, candidate),
        )
        == baseline
        for candidate in candidates
    )
    if not individual_restorations_exact:
        raise AssertionError("standalone matched candidate did not restore exactly to baseline")

    expected_composite_changes = (
        MATCHED_CANDIDATES_PER_WORLD * MATCHED_CHANGED_SLOT_COUNT
    )
    if len(changed_slot_ids(baseline, composite)) != expected_composite_changes:
        raise AssertionError("matched composite changed-slot count drift")

    restorations = tuple(
        restore_release_slots(
            composite,
            baseline,
            restore_slot_ids=tuple(sorted(changed_sets[index])),
        )
        for index in range(MATCHED_CANDIDATES_PER_WORLD)
    )

    role_map: dict[str, tuple[Exp009ReleaseSlot, ...]] = {}
    pair_truth: dict[str, dict[str, object]] = {}
    for index, (_pair, candidate) in enumerate(zip(definition.pairs, candidates, strict=True)):
        role = "root" if index == definition.root_position else f"non_root_{index}"
        role_map[role] = candidate

    diagnostic, truth = build_opaque_candidate_manifests(
        baseline,
        role_map,
        require_disjoint=True,
    )
    truth_by_role = {row["internal_role"]: row for row in truth["truth"]}
    for index, pair in enumerate(definition.pairs):
        role = "root" if index == definition.root_position else f"non_root_{index}"
        row = truth_by_role[role]
        pair_truth[str(row["candidate_id"])] = {
            "label_a": pair.label_a,
            "label_b": pair.label_b,
            "pair_rank_within_world": index,
            "internal_role": role,
        }

    structural_rows = tuple(
        _candidate_structural_row(baseline, candidate, pair)
        for candidate, pair in zip(candidates, definition.pairs, strict=True)
    )
    first = structural_rows[0]
    invariant_keys = (
        "changed_slot_count",
        "text_change_count",
        "label_change_count",
        "aggregate_label_count_delta",
        "touched_label_count",
        "transition_counts_match_33_33",
        "slot_count",
        "manifest_field_set",
    )
    for row in structural_rows[1:]:
        for key in invariant_keys:
            if row[key] != first[key]:
                raise AssertionError(f"matched candidate structural invariant drift: {key}")

    expected = {
        "changed_slot_count": MATCHED_CHANGED_SLOT_COUNT,
        "text_change_count": 0,
        "label_change_count": MATCHED_CHANGED_SLOT_COUNT,
        "aggregate_label_count_delta": {},
        "touched_label_count": 2,
        "transition_counts_match_33_33": True,
        "slot_count": len(baseline),
    }
    for key, value in expected.items():
        if first[key] != value:
            raise AssertionError(
                f"matched candidate structural audit failed for {key}: "
                f"{first[key]!r} != {value!r}"
            )

    truth = {
        **truth,
        "benchmark_namespace": MATCHED_BENCHMARK_NAMESPACE,
        "world_index": definition.world_index,
        "root_position": definition.root_position,
        "pair_truth_by_candidate_id": pair_truth,
    }
    audit = {
        "benchmark_namespace": MATCHED_BENCHMARK_NAMESPACE,
        "world_index": definition.world_index,
        "candidate_count": MATCHED_CANDIDATES_PER_WORLD,
        "all_candidate_structures_identical": True,
        "candidate_structure": first,
        "pairwise_changed_slots_disjoint": True,
        "intent_labels_pairwise_disjoint": True,
        "composite_changed_slot_count": expected_composite_changes,
        "composite_release_sha256": release_sha256(composite),
        "composite_order_independent": True,
        "all_individual_candidate_restorations_exact_baseline": True,
        "all_restorations_leave_exactly_four_candidate_changes": all(
            len(changed_slot_ids(baseline, restored))
            == (MATCHED_CANDIDATES_PER_WORLD - 1) * MATCHED_CHANGED_SLOT_COUNT
            for restored in restorations
        ),
        "official_test_split_loaded": False,
        "model_training_performed": False,
    }

    return BuiltMatchedWorld(
        definition=definition,
        baseline=baseline,
        candidate_releases=candidates,
        composite=composite,
        restorations=restorations,
        diagnostic_manifest=diagnostic,
        truth_manifest=truth,
        structural_audit=audit,
    )
