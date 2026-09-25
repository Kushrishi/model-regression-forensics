from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChangeAttributionScore:
    """Development change-level summary under the frozen Exp009 contract."""

    change_id: str
    changed_slot_count: int
    mean_support_delta: float
    suspicion_score: float


def class_balanced_slot_support(
    scores_by_slot: Mapping[str, Mapping[str, float]],
    target_label_by_example: Mapping[str, str],
    *,
    included_labels: Sequence[str],
) -> dict[str, float]:
    """Aggregate example scores to one class-balanced support value per slot."""

    labels = tuple(included_labels)
    if not labels:
        raise ValueError("included_labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("included_labels must be unique")

    target_ids = tuple(sorted(target_label_by_example))
    if not target_ids:
        raise ValueError("target_label_by_example must be non-empty")

    by_label: dict[str, tuple[str, ...]] = {}
    for label in labels:
        members = tuple(
            example_id
            for example_id in target_ids
            if target_label_by_example[example_id] == label
        )
        if not members:
            raise ValueError(f"no target examples for included label {label!r}")
        by_label[label] = members

    unexpected = sorted(set(target_label_by_example.values()) - set(labels))
    if unexpected:
        raise ValueError(
            "target_label_by_example contains labels outside included_labels: "
            + ", ".join(repr(label) for label in unexpected)
        )

    output: dict[str, float] = {}
    for slot_id, example_scores in scores_by_slot.items():
        missing = [example_id for example_id in target_ids if example_id not in example_scores]
        if missing:
            raise ValueError(
                f"slot {slot_id!r} is missing {len(missing)} target attribution scores"
            )

        class_means = [
            fmean(float(example_scores[example_id]) for example_id in by_label[label])
            for label in labels
        ]
        output[slot_id] = fmean(class_means)

    if not output:
        raise ValueError("scores_by_slot must be non-empty")
    return output


def version_change_suspicion_scores(
    baseline_support: Mapping[str, float],
    composite_support: Mapping[str, float],
    changed_slots_by_change: Mapping[str, Sequence[str]],
) -> tuple[ChangeAttributionScore, ...]:
    """Apply the frozen release-differential Exp009 change aggregation."""

    if set(baseline_support) != set(composite_support):
        raise ValueError("baseline/composite support must use identical stable slot IDs")

    rows: list[ChangeAttributionScore] = []
    for change_id, changed_slots_raw in changed_slots_by_change.items():
        changed_slots = tuple(changed_slots_raw)
        if not changed_slots:
            raise ValueError(f"change {change_id!r} has no changed slots")
        if len(set(changed_slots)) != len(changed_slots):
            raise ValueError(f"change {change_id!r} contains duplicate stable slots")

        missing = [slot_id for slot_id in changed_slots if slot_id not in baseline_support]
        if missing:
            raise ValueError(
                f"change {change_id!r} references {len(missing)} unknown stable slots"
            )

        deltas = [
            float(composite_support[slot_id]) - float(baseline_support[slot_id])
            for slot_id in changed_slots
        ]
        mean_delta = fmean(deltas)
        rows.append(
            ChangeAttributionScore(
                change_id=change_id,
                changed_slot_count=len(changed_slots),
                mean_support_delta=mean_delta,
                suspicion_score=-mean_delta,
            )
        )

    if not rows:
        raise ValueError("changed_slots_by_change must be non-empty")

    return tuple(sorted(rows, key=lambda row: (-row.suspicion_score, row.change_id)))
