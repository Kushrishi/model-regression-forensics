from __future__ import annotations

import math

import pytest

from model_forensics.exp009_attribution import (
    aggregate_candidate_suspiciousness,
    rank_candidate_scores,
    summarize_margin_regression,
    target_pair_margin_summary,
)


LABELS = ("A", "B", "C")
TARGET = ("A", "B")


def test_target_margin_uses_all_target_examples_with_equal_weight() -> None:
    logits = (
        (4.0, 1.0, 0.0),
        (3.0, 2.0, 0.0),
        (2.0, 5.0, 0.0),
        (100.0, 100.0, 7.0),
    )
    labels = ("A", "B", "A", "C")

    result = target_pair_margin_summary(
        logits,
        labels,
        label_order=LABELS,
        target_labels=TARGET,
    )

    assert result.example_count == 3
    assert result.per_label_counts == (("A", 2), ("B", 1))
    assert result.mean_margin == pytest.approx((-1.0) / 3.0)


def test_margin_regression_is_clean_minus_composite_without_flip_filtering() -> None:
    labels = ("A", "B", "C")
    clean = (
        (4.0, 1.0, 0.0),
        (1.0, 4.0, 0.0),
        (0.0, 0.0, 3.0),
    )
    composite = (
        (3.0, 1.0, 0.0),
        (1.0, 5.0, 0.0),
        (0.0, 0.0, 3.0),
    )

    result = summarize_margin_regression(
        clean,
        composite,
        labels,
        label_order=LABELS,
        target_labels=TARGET,
    )

    assert result.clean.mean_margin == pytest.approx(3.0)
    assert result.composite.mean_margin == pytest.approx(3.0)
    assert result.regression == pytest.approx(0.0)


def test_target_margin_rejects_missing_target_label() -> None:
    with pytest.raises(ValueError, match="both frozen target labels"):
        target_pair_margin_summary(
            ((3.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
            ("A", "C"),
            label_order=LABELS,
            target_labels=TARGET,
        )


def test_target_margin_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        target_pair_margin_summary(
            ((3.0, math.inf, 0.0), (1.0, 2.0, 0.0)),
            ("A", "B"),
            label_order=LABELS,
            target_labels=TARGET,
        )


def test_candidate_aggregation_uses_primary_sum() -> None:
    scores = {
        "s1": 2.0,
        "s2": -0.5,
        "s3": 0.75,
        "s4": 0.75,
    }
    candidates = {
        "change_a": ("s1", "s2"),
        "change_b": ("s3", "s4"),
    }

    aggregated = aggregate_candidate_suspiciousness(scores, candidates)

    assert aggregated == {"change_a": 1.5, "change_b": 1.5}
    assert rank_candidate_scores(aggregated) == ("change_a", "change_b")


def test_candidate_aggregation_requires_every_changed_slot_score() -> None:
    with pytest.raises(ValueError, match="missing suspiciousness"):
        aggregate_candidate_suspiciousness(
            {"s1": 1.0},
            {"candidate": ("s1", "s2")},
        )


def test_candidate_aggregation_rejects_duplicate_slots() -> None:
    with pytest.raises(ValueError, match="repeats a changed slot"):
        aggregate_candidate_suspiciousness(
            {"s1": 1.0},
            {"candidate": ("s1", "s1")},
        )


def test_candidate_ranking_descends_and_uses_candidate_id_for_ties() -> None:
    ranking = rank_candidate_scores(
        {
            "z_change": 1.0,
            "a_change": 1.0,
            "middle": 2.0,
        }
    )

    assert ranking == ("middle", "a_change", "z_change")
