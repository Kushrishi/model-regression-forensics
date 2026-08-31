import pytest

from model_forensics.certification import evaluate_exp005_candidate_gate


def _summary(scores: dict[str, float]) -> dict[str, object]:
    return {
        "experiment_id": "exp005",
        "scores": {
            "label_accuracy": {
                split: {"score": score, "total": 16, "failed_case_ids": []}
                for split, score in scores.items()
            }
        },
    }


def _clean_scores() -> dict[str, float]:
    return {
        "circle_small": 1.0,
        "circle_large": 1.0,
        "square_small": 1.0,
        "square_large": 1.0,
        "triangle_small": 1.0,
        "triangle_large": 1.0,
        "all": 1.0,
    }


def test_exp005_candidate_gate_accepts_localized_target_regression() -> None:
    candidate = _clean_scores()
    candidate["triangle_large"] = 0.75
    candidate["all"] = 0.9583333333333334

    result = evaluate_exp005_candidate_gate(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(candidate),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.baseline_passed
    assert result.target_regression == 0.25
    assert result.target_regression_passed
    assert result.all_protected_passed
    assert result.all_passed


def test_exp005_candidate_gate_rejects_insufficient_target_regression() -> None:
    candidate = _clean_scores()
    candidate["triangle_large"] = 0.875
    candidate["all"] = 0.9791666666666666

    result = evaluate_exp005_candidate_gate(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(candidate),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.target_regression == 0.125
    assert not result.target_regression_passed
    assert not result.all_passed


def test_exp005_candidate_gate_rejects_one_protected_slice_error() -> None:
    candidate = _clean_scores()
    candidate["triangle_large"] = 0.75
    candidate["circle_small"] = 0.9375
    candidate["all"] = 0.9479166666666666

    result = evaluate_exp005_candidate_gate(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(candidate),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.protected_drifts["circle_small"] == 0.0625
    assert not result.protected_splits_passed["circle_small"]
    assert not result.all_protected_passed
    assert not result.all_passed


def test_exp005_candidate_gate_rejects_failed_clean_baseline() -> None:
    baseline = _clean_scores()
    baseline["square_small"] = 0.9375
    candidate = baseline.copy()
    candidate["triangle_large"] = 0.75

    result = evaluate_exp005_candidate_gate(
        baseline_summary=_summary(baseline),
        candidate_summary=_summary(candidate),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert not result.baseline_passed
    assert not result.all_passed


def test_exp005_candidate_gate_requires_all_frozen_splits() -> None:
    baseline = _clean_scores()
    candidate = _clean_scores()
    candidate.pop("triangle_small")

    with pytest.raises(ValueError, match="missing required Experiment 005 evaluation splits"):
        evaluate_exp005_candidate_gate(
            baseline_summary=_summary(baseline),
            candidate_summary=_summary(candidate),
            minimum_baseline_score=0.95,
            minimum_regression_delta=0.15,
            maximum_unrelated_delta=0.05,
        )
