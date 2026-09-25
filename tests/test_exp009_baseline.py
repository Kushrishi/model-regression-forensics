from __future__ import annotations

import pytest

from model_forensics.exp009_baseline import (
    class_balanced_slot_support,
    version_change_suspicion_scores,
)


def test_class_balancing_gives_each_incident_intent_equal_weight() -> None:
    labels = {
        "a1": "A",
        "a2": "A",
        "a3": "A",
        "b1": "B",
    }
    scores = {
        "slot-1": {
            "a1": 1.0,
            "a2": 1.0,
            "a3": 1.0,
            "b1": 3.0,
        }
    }

    support = class_balanced_slot_support(
        scores,
        labels,
        included_labels=("A", "B"),
    )

    assert support["slot-1"] == pytest.approx(2.0)


def test_change_suspicion_is_negative_mean_release_support_delta() -> None:
    baseline = {
        "s1": 0.8,
        "s2": 0.7,
        "s3": 0.2,
        "s4": 0.2,
    }
    composite = {
        "s1": 0.2,
        "s2": 0.3,
        "s3": 0.1,
        "s4": 0.2,
    }

    rows = version_change_suspicion_scores(
        baseline,
        composite,
        {
            "candidate-root-opaque": ("s1", "s2"),
            "candidate-other-opaque": ("s3", "s4"),
        },
    )

    assert [row.change_id for row in rows] == [
        "candidate-root-opaque",
        "candidate-other-opaque",
    ]
    assert rows[0].mean_support_delta == pytest.approx(-0.5)
    assert rows[0].suspicion_score == pytest.approx(0.5)
    assert rows[1].mean_support_delta == pytest.approx(-0.05)
    assert rows[1].suspicion_score == pytest.approx(0.05)


def test_aggregation_does_not_require_root_or_nuisance_roles() -> None:
    baseline = {"slot-a": 0.1, "slot-b": 0.1}
    composite = {"slot-a": -0.1, "slot-b": 0.2}

    rows = version_change_suspicion_scores(
        baseline,
        composite,
        {"candidate-17": ("slot-a",), "candidate-42": ("slot-b",)},
    )

    assert {row.change_id for row in rows} == {"candidate-17", "candidate-42"}


def test_target_scores_reject_missing_example_attribution() -> None:
    with pytest.raises(ValueError, match="missing"):
        class_balanced_slot_support(
            {"slot": {"a": 1.0}},
            {"a": "A", "b": "B"},
            included_labels=("A", "B"),
        )


def test_change_scores_reject_unknown_stable_slot() -> None:
    with pytest.raises(ValueError, match="unknown stable slots"):
        version_change_suspicion_scores(
            {"s1": 0.0},
            {"s1": 0.0},
            {"candidate": ("missing",)},
        )
