import pytest

from model_forensics.specificity import assess_repairs


def report(*, passed, baseline="v1", candidate="v2", baseline_accuracy=1.0):
    return {
        "evaluation_policy_sha256": "a" * 64,
        "baseline_release_id": baseline,
        "candidate_release_id": candidate,
        "passed": passed,
        "slices": [
            {
                "name": "all",
                "count": 10,
                "baseline_accuracy": baseline_accuracy,
                "candidate_accuracy": 1.0 if passed else 0.5,
                "accuracy_drop": 0.0 if passed else 0.5,
                "maximum_accuracy_drop": 0.0,
                "passed": passed,
            }
        ],
    }


def test_multiple_successful_repairs_are_ambiguous():
    result = assess_repairs(
        report(passed=False),
        {
            "restore-input-order": report(passed=True, candidate="repair-a"),
            "reverse-model-weights": report(passed=True, candidate="repair-b"),
        },
    )
    assert result["status"] == "ambiguous_repairs"
    assert result["historical_cause"] == "not_identified"
    assert result["successful_repairs"] == ["restore-input-order", "reverse-model-weights"]


def test_single_repair_is_not_promoted_to_historical_cause():
    result = assess_repairs(
        report(passed=False),
        {"rollback": report(passed=True, candidate="repair")},
    )
    assert result["status"] == "single_supported_repair"
    assert result["historical_cause"] == "not_identified"


def test_no_repair_evidence_is_insufficient():
    result = assess_repairs(report(passed=False), {})
    assert result["status"] == "insufficient_evidence"
    assert result["successful_repairs"] == []


def test_rejects_mismatched_policy_or_baseline():
    with pytest.raises(ValueError, match="same baseline and evaluation policy"):
        assess_repairs(
            report(passed=False),
            {"other": report(passed=True, baseline="different")},
        )

    with pytest.raises(ValueError, match="same baseline and evaluation policy"):
        assess_repairs(
            report(passed=False),
            {"other": report(passed=True, baseline_accuracy=0.9)},
        )


def test_requires_actual_regression():
    with pytest.raises(ValueError, match="regressed report must fail"):
        assess_repairs(report(passed=True), {})
