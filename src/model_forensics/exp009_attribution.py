from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetMarginSummary:
    """Frozen Exp009 target-pair margin summary for one model."""

    mean_margin: float
    example_count: int
    per_label_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class MarginRegressionSummary:
    """Observed clean-to-composite regression under the frozen target scalar."""

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


def target_pair_margin_summary(
    logits: Sequence[Sequence[float]],
    true_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    target_labels: tuple[str, str],
) -> TargetMarginSummary:
    """Compute the frozen equal-weight correct-vs-other-target logit margin.

    Only examples whose true label belongs to target_labels contribute to the
    mean. Protected examples are validated but do not contribute.

    The function intentionally does not filter negative flips or weight examples
    by observed clean-to-composite degradation.
    """

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
    margins: list[float] = []
    counts = {target_a: 0, target_b: 0}

    for row_index, (row, true_label) in enumerate(zip(rows, labels, strict=True)):
        if true_label not in index:
            raise ValueError(f"unknown true label at row {row_index}: {true_label!r}")
        if len(row) != expected_width:
            raise ValueError(
                f"logit width mismatch at row {row_index}: "
                f"expected={expected_width} observed={len(row)}"
            )

        numeric_row = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in numeric_row):
            raise ValueError(f"non-finite logit at row {row_index}")

        if true_label not in counts:
            continue

        other_label = target_b if true_label == target_a else target_a
        margin = numeric_row[index[true_label]] - numeric_row[index[other_label]]
        margins.append(margin)
        counts[true_label] += 1

    if counts[target_a] == 0 or counts[target_b] == 0:
        raise ValueError("target evaluation slice must contain both frozen target labels")

    return TargetMarginSummary(
        mean_margin=math.fsum(margins) / len(margins),
        example_count=len(margins),
        per_label_counts=((target_a, counts[target_a]), (target_b, counts[target_b])),
    )


def summarize_margin_regression(
    clean_logits: Sequence[Sequence[float]],
    composite_logits: Sequence[Sequence[float]],
    true_labels: Sequence[str],
    *,
    label_order: Sequence[str],
    target_labels: tuple[str, str],
) -> MarginRegressionSummary:
    """Compute M(clean) - M(composite) under the frozen target contract."""

    clean = target_pair_margin_summary(
        clean_logits,
        true_labels,
        label_order=label_order,
        target_labels=target_labels,
    )
    composite = target_pair_margin_summary(
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


def aggregate_candidate_suspiciousness(
    slot_suspiciousness: Mapping[str, float],
    candidate_changed_slots: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Sum slot suspiciousness over every debugger-visible candidate change.

    Slot scores must already be oriented so larger values mean greater predicted
    improvement in the target behavior if the current contribution is
    removed/restored. This function never flips method-specific signs.
    """

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

        missing = sorted(slot_id for slot_id in slot_ids if slot_id not in normalized_scores)
        if missing:
            raise ValueError(
                f"candidate {candidate_id!r} is missing suspiciousness for slots: {missing}"
            )

        output[candidate_id] = math.fsum(normalized_scores[slot_id] for slot_id in slot_ids)

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
