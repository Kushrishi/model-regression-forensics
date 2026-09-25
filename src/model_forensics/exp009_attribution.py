from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class TargetMarginSummary:
    """Class-balanced multiclass margin summary for the Exp009 incident slice."""

    mean_margin: float
    example_count: int
    per_label_counts: tuple[tuple[str, int], ...]
    per_label_mean_margins: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class MarginRegressionSummary:
    """Clean-to-composite descriptive regression under the frozen target scalar."""

    clean: TargetMarginSummary
    composite: TargetMarginSummary
    regression: float


def _validate_target_labels(target_labels: tuple[str, str]) -> tuple[str, str]:
    if len(target_labels) != 2:
        raise ValueError("target_labels must contain exactly two labels")
    target_a, target_b = target_labels
    if target_a == target_b:
        raise ValueError("target_labels must be distinct")
    return target_a, target_b


def _label_index(label_order: Sequence[str]) -> dict[str, int]:
    labels = tuple(label_order)
    if not labels:
        raise ValueError("label_order must be non-empty")
    if len(labels) != len(set(labels)):
        raise ValueError("label_order must contain unique labels")
    return {label: index for index, label in enumerate(labels)}


def _logsumexp(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    if not numeric:
        raise ValueError("logsumexp requires at least one value")
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("logsumexp received a non-finite value")
    maximum = max(numeric)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in numeric))


def target_classification_margin_summary(
    logits: Sequence[Sequence[float]],
    true_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    target_labels: tuple[str, str],
) -> TargetMarginSummary:
    """Compute the frozen class-balanced correct-class-vs-all margin."""

    target_a, target_b = _validate_target_labels(target_labels)
    index = _label_index(label_order)
    if target_a not in index or target_b not in index:
        raise ValueError("both target labels must exist in label_order")

    rows = tuple(logits)
    labels = tuple(true_labels)
    if len(rows) != len(labels):
        raise ValueError("logits and true_labels must have identical length")
    if not rows:
        raise ValueError("at least one evaluation example is required")

    expected_width = len(index)
    margins_by_label: dict[str, list[float]] = {target_a: [], target_b: []}

    for row_number, (row, true_label) in enumerate(zip(rows, labels, strict=True)):
        if true_label not in index:
            raise ValueError(f"unknown true label at row {row_number}: {true_label!r}")
        if len(row) != expected_width:
            raise ValueError(
                f"logit width mismatch at row {row_number}: "
                f"expected={expected_width} observed={len(row)}"
            )

        numeric_row = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in numeric_row):
            raise ValueError(f"non-finite logit at row {row_number}")

        if true_label not in margins_by_label:
            continue

        correct_index = index[true_label]
        incorrect = tuple(
            value
            for class_index, value in enumerate(numeric_row)
            if class_index != correct_index
        )
        margin = numeric_row[correct_index] - _logsumexp(incorrect)
        margins_by_label[true_label].append(margin)

    if not margins_by_label[target_a] or not margins_by_label[target_b]:
        raise ValueError("target evaluation slice must contain both frozen target labels")

    per_label_means = {
        label: fmean(margins_by_label[label]) for label in (target_a, target_b)
    }
    return TargetMarginSummary(
        mean_margin=fmean(per_label_means.values()),
        example_count=sum(len(values) for values in margins_by_label.values()),
        per_label_counts=(
            (target_a, len(margins_by_label[target_a])),
            (target_b, len(margins_by_label[target_b])),
        ),
        per_label_mean_margins=(
            (target_a, per_label_means[target_a]),
            (target_b, per_label_means[target_b]),
        ),
    )


def summarize_margin_regression(
    clean_logits: Sequence[Sequence[float]],
    composite_logits: Sequence[Sequence[float]],
    true_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    target_labels: tuple[str, str],
) -> MarginRegressionSummary:
    """Compute descriptive M(clean) - M(composite) under the frozen target."""

    clean = target_classification_margin_summary(
        clean_logits,
        true_labels,
        label_order=label_order,
        target_labels=target_labels,
    )
    composite = target_classification_margin_summary(
        composite_logits,
        true_labels,
        label_order=label_order,
        target_labels=target_labels,
    )
    if clean.example_count != composite.example_count:
        raise AssertionError("clean/composite target-example count mismatch")
    if clean.per_label_counts != composite.per_label_counts:
        raise AssertionError("clean/composite target-label composition mismatch")

    return MarginRegressionSummary(
        clean=clean,
        composite=composite,
        regression=clean.mean_margin - composite.mean_margin,
    )


def class_balanced_slot_support(
    scores_by_slot: Mapping[str, Mapping[str, float]],
    target_label_by_example: Mapping[str, str],
    *,
    target_labels: tuple[str, str],
) -> dict[str, float]:
    """Average target attribution with equal weight per incident intent."""

    target_a, target_b = _validate_target_labels(target_labels)
    target_ids = tuple(sorted(target_label_by_example))
    if not target_ids:
        raise ValueError("target_label_by_example must be non-empty")

    unexpected = sorted(set(target_label_by_example.values()) - {target_a, target_b})
    if unexpected:
        raise ValueError(
            "target_label_by_example contains labels outside target_labels: "
            + ", ".join(repr(label) for label in unexpected)
        )

    by_label = {
        label: tuple(
            example_id
            for example_id in target_ids
            if target_label_by_example[example_id] == label
        )
        for label in (target_a, target_b)
    }
    if not by_label[target_a] or not by_label[target_b]:
        raise ValueError("target attribution slice must contain both target labels")

    output: dict[str, float] = {}
    for slot_id, example_scores in scores_by_slot.items():
        missing = [
            example_id for example_id in target_ids if example_id not in example_scores
        ]
        if missing:
            raise ValueError(
                f"slot {slot_id!r} is missing {len(missing)} target attribution scores"
            )

        label_means = [
            fmean(float(example_scores[example_id]) for example_id in by_label[label])
            for label in (target_a, target_b)
        ]
        if not all(math.isfinite(value) for value in label_means):
            raise ValueError(f"slot {slot_id!r} has non-finite attribution support")
        output[slot_id] = fmean(label_means)

    if not output:
        raise ValueError("scores_by_slot must be non-empty")
    return output


def detracting_support_to_suspiciousness(
    slot_support: Mapping[str, float],
) -> dict[str, float]:
    """Orient helpful-positive support so detracting slots are suspicious."""

    output: dict[str, float] = {}
    for slot_id, value in slot_support.items():
        support = float(value)
        if not math.isfinite(support):
            raise ValueError(f"non-finite support for slot {slot_id!r}")
        output[slot_id] = -support
    if not output:
        raise ValueError("slot_support must be non-empty")
    return output


def aggregate_candidate_suspiciousness(
    slot_suspiciousness: Mapping[str, float],
    candidate_changed_slots: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Sum suspiciousness over every changed slot in each opaque candidate."""

    if not candidate_changed_slots:
        raise ValueError("at least one candidate change is required")

    normalized_scores: dict[str, float] = {}
    for slot_id, value in slot_suspiciousness.items():
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"non-finite suspiciousness for slot {slot_id!r}")
        normalized_scores[slot_id] = score

    output: dict[str, float] = {}
    for candidate_id, slot_ids_raw in candidate_changed_slots.items():
        slot_ids = tuple(slot_ids_raw)
        if not slot_ids:
            raise ValueError(f"candidate {candidate_id!r} has no changed slots")
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError(f"candidate {candidate_id!r} repeats a changed slot")

        missing = sorted(
            slot_id for slot_id in slot_ids if slot_id not in normalized_scores
        )
        if missing:
            raise ValueError(
                f"candidate {candidate_id!r} is missing suspiciousness for slots: {missing}"
            )

        output[candidate_id] = math.fsum(
            normalized_scores[slot_id] for slot_id in slot_ids
        )

    return output


def rank_candidate_scores(candidate_scores: Mapping[str, float]) -> tuple[str, ...]:
    """Rank candidates from highest to lowest suspiciousness with stable ties."""

    if not candidate_scores:
        raise ValueError("at least one candidate score is required")

    normalized: dict[str, float] = {}
    for candidate_id, value in candidate_scores.items():
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"non-finite candidate score for {candidate_id!r}")
        normalized[candidate_id] = score

    return tuple(
        sorted(normalized, key=lambda candidate_id: (-normalized[candidate_id], candidate_id))
    )
