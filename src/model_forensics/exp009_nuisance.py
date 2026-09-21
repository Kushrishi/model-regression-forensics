from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations

from model_forensics.exp009_data import DevelopmentPartition, content_id_for_record
from model_forensics.exp009_release import (
    Exp009ReleaseSlot,
    changed_slot_ids,
    release_diff_manifest,
)

NUISANCE_SELECTION_NAMESPACE = "exp009-pilot-balanced-cross-intent-refresh-v1"


@dataclass(frozen=True)
class NuisancePairScore:
    """Clean-only ranking evidence for one prospective nuisance intent pair."""

    label_a: str
    label_b: str
    mean_bidirectional_confusion_rate: float
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
class NuisanceRefreshSelection:
    """Stable recipient and donor identities for one nuisance artifact."""

    nuisance_index: int
    label_a: str
    label_b: str
    recipient_a_slot_ids: tuple[str, ...]
    recipient_b_slot_ids: tuple[str, ...]
    donor_a_slot_ids: tuple[str, ...]
    donor_b_slot_ids: tuple[str, ...]


def _intent_tokens(label: str) -> frozenset[str]:
    return frozenset(token for token in label.lower().split("_") if token)


def _lexical_jaccard(label_a: str, label_b: str) -> float:
    a_tokens = _intent_tokens(label_a)
    b_tokens = _intent_tokens(label_b)
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def rank_nuisance_pairs(
    partition: DevelopmentPartition,
    *,
    per_label_recall_by_trajectory: dict[int, dict[str, float]],
    prediction_rows_by_trajectory: dict[int, tuple[dict[str, str], ...]],
    excluded_labels: frozenset[str],
    minimum_clean_recall: float = 0.90,
    minimum_eval_examples: int = 20,
    minimum_train_examples: int = 66,
) -> tuple[NuisancePairScore, ...]:
    """Rank eligible nuisance pairs from clean development evidence only."""

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
                f"trajectory {trajectory_id} is missing per-label recall for {sorted(missing)!r}"
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

    eligible: list[NuisancePairScore] = []
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

        min_recall = min(
            min(recalls[label_a], recalls[label_b])
            for recalls in per_label_recall_by_trajectory.values()
        )
        if min_recall < minimum_clean_recall:
            continue

        lexical_jaccard = _lexical_jaccard(label_a, label_b)
        if lexical_jaccard <= 0.0:
            continue

        a_to_b = confusion_counts[(label_a, label_b)]
        b_to_a = confusion_counts[(label_b, label_a)]
        if a_to_b <= 0 or b_to_a <= 0:
            continue

        if true_counts[label_a] <= 0 or true_counts[label_b] <= 0:
            raise ValueError("eligible nuisance label has no pooled clean predictions")

        mean_bidirectional_rate = 0.5 * (
            a_to_b / true_counts[label_a] + b_to_a / true_counts[label_b]
        )
        eligible.append(
            NuisancePairScore(
                label_a=label_a,
                label_b=label_b,
                mean_bidirectional_confusion_rate=mean_bidirectional_rate,
                mutual_confusion_count=a_to_b + b_to_a,
                lexical_jaccard=lexical_jaccard,
                minimum_clean_recall=min_recall,
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
                -score.mean_bidirectional_confusion_rate,
                -score.mutual_confusion_count,
                -score.lexical_jaccard,
                score.label_a,
                score.label_b,
            ),
        )
    )


def rank_nuisance_pairs_v2(
    partition: DevelopmentPartition,
    *,
    per_label_recall_by_trajectory: dict[int, dict[str, float]],
    prediction_rows_by_trajectory: dict[int, tuple[dict[str, str], ...]],
    excluded_labels: frozenset[str],
    minimum_clean_recall: float = 0.90,
    minimum_eval_examples: int = 20,
    minimum_train_examples: int = 66,
) -> tuple[NuisancePairScore, ...]:
    """Rank v2 nuisance pairs using clean evidence with one-way confusion allowed."""

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
                f"trajectory {trajectory_id} is missing per-label recall for {sorted(missing)!r}"
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

    eligible: list[NuisancePairScore] = []
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

        min_recall = min(
            min(recalls[label_a], recalls[label_b])
            for recalls in per_label_recall_by_trajectory.values()
        )
        if min_recall < minimum_clean_recall:
            continue

        lexical_jaccard = _lexical_jaccard(label_a, label_b)
        if lexical_jaccard <= 0.0:
            continue

        a_to_b = confusion_counts[(label_a, label_b)]
        b_to_a = confusion_counts[(label_b, label_a)]
        if a_to_b <= 0 and b_to_a <= 0:
            continue

        if true_counts[label_a] <= 0 or true_counts[label_b] <= 0:
            raise ValueError("eligible nuisance label has no pooled clean predictions")

        mean_symmetric_rate = 0.5 * (a_to_b / true_counts[label_a] + b_to_a / true_counts[label_b])
        eligible.append(
            NuisancePairScore(
                label_a=label_a,
                label_b=label_b,
                mean_bidirectional_confusion_rate=mean_symmetric_rate,
                mutual_confusion_count=a_to_b + b_to_a,
                lexical_jaccard=lexical_jaccard,
                minimum_clean_recall=min_recall,
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
                -score.mean_bidirectional_confusion_rate,
                -score.mutual_confusion_count,
                -score.lexical_jaccard,
                score.label_a,
                score.label_b,
            ),
        )
    )


def select_disjoint_nuisance_pairs(
    ranked_pairs: tuple[NuisancePairScore, ...],
    *,
    count: int = 4,
) -> tuple[NuisancePairScore, ...]:
    """Greedily select the first disjoint pairs from a frozen ranking."""

    if count <= 0:
        raise ValueError("nuisance pair count must be positive")

    selected: list[NuisancePairScore] = []
    used_labels: set[str] = set()
    for pair in ranked_pairs:
        labels = {pair.label_a, pair.label_b}
        if labels & used_labels:
            continue
        selected.append(pair)
        used_labels.update(labels)
        if len(selected) == count:
            return tuple(selected)

    raise ValueError(
        f"fewer than {count} disjoint nuisance pairs satisfy the frozen eligibility rule"
    )


def _slot_selection_key(
    slot: Exp009ReleaseSlot,
    *,
    nuisance_index: int,
    label_a: str,
    label_b: str,
    role: str,
) -> bytes:
    pair = tuple(sorted((label_a, label_b)))
    payload = (
        f"{NUISANCE_SELECTION_NAMESPACE}|nuisance={nuisance_index}|"
        f"pair={pair[0]}|{pair[1]}|role={role}|source-label={slot.label}|"
        f"slot={slot.slot_id}|source={slot.source_content_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _rank_label_slots(
    clean_slots: tuple[Exp009ReleaseSlot, ...],
    *,
    label: str,
    nuisance_index: int,
    label_a: str,
    label_b: str,
    role: str,
) -> list[Exp009ReleaseSlot]:
    return sorted(
        (slot for slot in clean_slots if slot.label == label),
        key=lambda slot: _slot_selection_key(
            slot,
            nuisance_index=nuisance_index,
            label_a=label_a,
            label_b=label_b,
            role=role,
        ),
    )


def select_nuisance_refresh_slots(
    clean_slots: tuple[Exp009ReleaseSlot, ...],
    *,
    nuisance_index: int,
    label_a: str,
    label_b: str,
    per_direction: int = 33,
) -> NuisanceRefreshSelection:
    """Select deterministic recipients and distinct same-label donor pools."""

    if nuisance_index <= 0:
        raise ValueError("nuisance_index must be positive")
    if label_a == label_b:
        raise ValueError("nuisance labels must differ")
    if per_direction <= 0:
        raise ValueError("per_direction must be positive")

    recipients_a_ranked = _rank_label_slots(
        clean_slots,
        label=label_a,
        nuisance_index=nuisance_index,
        label_a=label_a,
        label_b=label_b,
        role="recipient",
    )
    recipients_b_ranked = _rank_label_slots(
        clean_slots,
        label=label_b,
        nuisance_index=nuisance_index,
        label_a=label_a,
        label_b=label_b,
        role="recipient",
    )
    if len(recipients_a_ranked) < 2 * per_direction:
        raise ValueError(f"not enough {label_a!r} slots for recipients plus donors")
    if len(recipients_b_ranked) < 2 * per_direction:
        raise ValueError(f"not enough {label_b!r} slots for recipients plus donors")

    recipients_a = tuple(recipients_a_ranked[:per_direction])
    recipients_b = tuple(recipients_b_ranked[:per_direction])
    recipient_ids = {slot.slot_id for slot in (*recipients_a, *recipients_b)}

    donors_a = tuple(
        slot
        for slot in _rank_label_slots(
            clean_slots,
            label=label_a,
            nuisance_index=nuisance_index,
            label_a=label_a,
            label_b=label_b,
            role="donor",
        )
        if slot.slot_id not in recipient_ids
    )[:per_direction]
    donors_b = tuple(
        slot
        for slot in _rank_label_slots(
            clean_slots,
            label=label_b,
            nuisance_index=nuisance_index,
            label_a=label_a,
            label_b=label_b,
            role="donor",
        )
        if slot.slot_id not in recipient_ids
    )[:per_direction]

    if len(donors_a) != per_direction or len(donors_b) != per_direction:
        raise ValueError("could not select the required distinct nuisance donor slots")

    return NuisanceRefreshSelection(
        nuisance_index=nuisance_index,
        label_a=label_a,
        label_b=label_b,
        recipient_a_slot_ids=tuple(slot.slot_id for slot in recipients_a),
        recipient_b_slot_ids=tuple(slot.slot_id for slot in recipients_b),
        donor_a_slot_ids=tuple(slot.slot_id for slot in donors_a),
        donor_b_slot_ids=tuple(slot.slot_id for slot in donors_b),
    )


def build_balanced_cross_intent_refresh(
    clean_slots: tuple[Exp009ReleaseSlot, ...],
    *,
    nuisance_index: int,
    label_a: str,
    label_b: str,
    per_direction: int = 33,
) -> tuple[tuple[Exp009ReleaseSlot, ...], NuisanceRefreshSelection]:
    """Build one label-correct balanced refresh/reweighting nuisance release."""

    selection = select_nuisance_refresh_slots(
        clean_slots,
        nuisance_index=nuisance_index,
        label_a=label_a,
        label_b=label_b,
        per_direction=per_direction,
    )
    by_id = {slot.slot_id: slot for slot in clean_slots}

    recipient_a = [by_id[slot_id] for slot_id in selection.recipient_a_slot_ids]
    recipient_b = [by_id[slot_id] for slot_id in selection.recipient_b_slot_ids]
    donor_a = [by_id[slot_id] for slot_id in selection.donor_a_slot_ids]
    donor_b = [by_id[slot_id] for slot_id in selection.donor_b_slot_ids]

    replacements: dict[str, Exp009ReleaseSlot] = {}
    for recipient, donor in zip(recipient_a, donor_b, strict=True):
        replacements[recipient.slot_id] = replace(
            recipient,
            text=donor.text,
            label=donor.label,
            model_content_id=content_id_for_record(label=donor.label, text=donor.text),
        )
    for recipient, donor in zip(recipient_b, donor_a, strict=True):
        replacements[recipient.slot_id] = replace(
            recipient,
            text=donor.text,
            label=donor.label,
            model_content_id=content_id_for_record(label=donor.label, text=donor.text),
        )

    candidate = tuple(replacements.get(slot.slot_id, slot) for slot in clean_slots)
    return candidate, selection


def duplicate_model_content_profile(
    slots: tuple[Exp009ReleaseSlot, ...],
) -> dict[str, int]:
    """Count intentionally repeated model-facing examples without exposing text."""

    counts = Counter(slot.model_content_id for slot in slots)
    duplicate_groups = sum(count > 1 for count in counts.values())
    duplicate_occurrences = sum(count - 1 for count in counts.values())
    return {
        "duplicate_model_content_groups": duplicate_groups,
        "duplicate_model_content_occurrences_beyond_first": duplicate_occurrences,
    }


def nuisance_refresh_audit(
    clean_slots: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
    selection: NuisanceRefreshSelection,
) -> dict[str, object]:
    """Return a text-free structural audit for one nuisance update."""

    clean_by_id = {slot.slot_id: slot for slot in clean_slots}
    candidate_by_id = {slot.slot_id: slot for slot in candidate}
    changed = changed_slot_ids(clean_slots, candidate)
    changed_set = set(changed)

    recipient_ids = set(selection.recipient_a_slot_ids) | set(selection.recipient_b_slot_ids)
    donor_ids = set(selection.donor_a_slot_ids) | set(selection.donor_b_slot_ids)
    if changed_set != recipient_ids:
        raise AssertionError("nuisance changed slots do not exactly equal selected recipients")
    if recipient_ids & donor_ids:
        raise AssertionError("nuisance recipient and donor slots overlap")

    text_change_count = sum(
        clean_by_id[slot_id].text != candidate_by_id[slot_id].text for slot_id in changed
    )
    label_delta = Counter(slot.label for slot in candidate)
    label_delta.subtract(Counter(slot.label for slot in clean_slots))
    nonzero_label_delta = {label: delta for label, delta in label_delta.items() if delta}

    manifest = release_diff_manifest(clean_slots, candidate)
    duplicate_profile = duplicate_model_content_profile(candidate)
    return {
        "nuisance_index": selection.nuisance_index,
        "label_a": selection.label_a,
        "label_b": selection.label_b,
        "recipient_count_a": len(selection.recipient_a_slot_ids),
        "recipient_count_b": len(selection.recipient_b_slot_ids),
        "donor_count_a": len(selection.donor_a_slot_ids),
        "donor_count_b": len(selection.donor_b_slot_ids),
        "recipient_donor_disjoint": True,
        "changed_slot_count": manifest["changed_slot_count"],
        "changed_slot_ids_sha256": manifest["changed_slot_ids_sha256"],
        "candidate_release_sha256": manifest["candidate_release_sha256"],
        "label_transitions": manifest["label_transitions"],
        "text_change_count": text_change_count,
        "aggregate_label_count_delta": nonzero_label_delta,
        **duplicate_profile,
    }


def combine_disjoint_release_changes(
    baseline: tuple[Exp009ReleaseSlot, ...],
    *candidate_releases: tuple[Exp009ReleaseSlot, ...],
) -> tuple[Exp009ReleaseSlot, ...]:
    """Compose candidate releases only when their changed stable slots are disjoint."""

    baseline_by_id = {slot.slot_id: slot for slot in baseline}
    replacements: dict[str, Exp009ReleaseSlot] = {}
    for candidate in candidate_releases:
        candidate_by_id = {slot.slot_id: slot for slot in candidate}
        if set(candidate_by_id) != set(baseline_by_id):
            raise ValueError("candidate release is not aligned to the baseline slot set")
        for slot_id in changed_slot_ids(baseline, candidate):
            if slot_id in replacements:
                raise ValueError(f"candidate releases overlap at stable slot {slot_id}")
            replacements[slot_id] = candidate_by_id[slot_id]

    return tuple(replacements.get(slot.slot_id, slot) for slot in baseline)
