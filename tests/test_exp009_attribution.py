from __future__ import annotations

import math

import pytest

from model_forensics.exp009_attribution import (
    aggregate_candidate_suspiciousness,
    class_balanced_slot_support,
    detracting_support_to_suspiciousness,
    rank_candidate_scores,
    summarize_margin_regression,
    target_classification_margin_summary,
)

LABELS = ("A", "B", "C")
TARGET = ("A", "B")


def test_target_margin_is_correct_class_against_all_other_classes() -> None:
    logits = ((4.0, 1.0, 0.0), (1.0, 4.0, 0.0))
    labels = ("A", "B")

    result = target_classification_margin_summary(
        logits, labels, label_order=LABELS, target_labels=TARGET
    )

    expected = 4.0 - math.log(math.exp(1.0) + math.exp(0.0))
    assert result.mean_margin == pytest.approx(expected)


def test_target_margin_class_balances_the_two_incident_intents() -> None:
    logits = (
        (5.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (100.0, 100.0, 7.0),
    )
    labels = ("A", "A", "A", "B", "C")

    result = target_classification_margin_summary(
        logits, labels, label_order=LABELS, target_labels=TARGET
    )

    margin_a = 5.0 - math.log(2.0)
    margin_b = 2.0 - math.log(2.0)
    assert result.example_count == 4
    assert result.per_label_counts == (("A", 3), ("B", 1))
    assert result.mean_margin == pytest.approx((margin_a + margin_b) / 2.0)


def test_margin_regression_is_clean_minus_composite() -> None:
    labels = ("A", "B", "C")
    clean = ((4.0, 1.0, 0.0), (1.0, 4.0, 0.0), (0.0, 0.0, 3.0))
    composite = ((2.0, 1.0, 3.0), (1.0, 2.0, 3.0), (0.0, 0.0, 3.0))

    result = summarize_margin_regression(
        clean, composite, labels, label_order=LABELS, target_labels=TARGET
    )

    assert result.regression > 0.0


def test_target_margin_rejects_missing_target_label() -> None:
    with pytest.raises(ValueError, match="both frozen target labels"):
        target_classification_margin_summary(
            ((3.0, 1.0, 0.0), (0.0, 0.0, 2.0)),
            ("A", "C"),
            label_order=LABELS,
            target_labels=TARGET,
        )


def test_target_margin_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        target_classification_margin_summary(
            ((3.0, math.inf, 0.0), (1.0, 2.0, 0.0)),
            ("A", "B"),
            label_order=LABELS,
            target_labels=TARGET,
        )


def test_class_balanced_slot_support_matches_macro_weighting() -> None:
    labels = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
    scores = {"slot": {"a1": 1.0, "a2": 1.0, "a3": 1.0, "b1": 3.0}}

    support = class_balanced_slot_support(scores, labels, target_labels=TARGET)

    assert support["slot"] == pytest.approx(2.0)


def test_detracting_support_is_positive_suspiciousness() -> None:
    result = detracting_support_to_suspiciousness({"harmful": -0.8, "helpful": 0.4})
    assert result == {"harmful": 0.8, "helpful": -0.4}


def test_candidate_aggregation_uses_primary_sum() -> None:
    scores = {"s1": 2.0, "s2": -0.5, "s3": 0.75, "s4": 0.75}
    candidates = {
        "change_a": ("s1", "s2"),
        "change_b": ("s3", "s4"),
    }

    aggregated = aggregate_candidate_suspiciousness(scores, candidates)

    assert aggregated == {"change_a": 1.5, "change_b": 1.5}
    assert rank_candidate_scores(aggregated) == ("change_a", "change_b")


def test_candidate_aggregation_requires_every_changed_slot_score() -> None:
    with pytest.raises(ValueError, match="missing suspiciousness"):
        aggregate_candidate_suspiciousness({"s1": 1.0}, {"candidate": ("s1", "s2")})


def test_candidate_ranking_descends_and_uses_candidate_id_for_ties() -> None:
    ranking = rank_candidate_scores({"z_change": 1.0, "a_change": 1.0, "middle": 2.0})
    assert ranking == ("middle", "a_change", "z_change")
