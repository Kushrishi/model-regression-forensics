import pytest

from model_forensics.certification import evaluate_exp008_candidate_gate


def _scores(*, target: float = 1.0) -> dict[str, float]:
    return {
        "circle_small": 1.0,
        "circle_large": 1.0,
        "square_small": 1.0,
        "square_large": 1.0,
        "triangle_small": 1.0,
        "triangle_large": target,
        "all": (80.0 + 16.0 * target) / 96.0,
    }


def _summary(
    scores: dict[str, float],
    *,
    experiment_id: str = "exp008",
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "scores": {
            "label_accuracy": {
                split: {
                    "score": score,
                    "total": 96 if split == "all" else 16,
                    "failed_case_ids": [],
                }
                for split, score in scores.items()
            }
        },
    }


def test_exp008_candidate_gate_accepts_world0_observed_pattern() -> None:
    result = evaluate_exp008_candidate_gate(
        baseline_summary=_summary(_scores()),
        candidate_summary=_summary(_scores(target=0.0)),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.baseline_passed
    assert result.target_regression == 1.0
    assert result.target_regression_passed
    assert all(drift == 0.0 for drift in result.protected_drifts.values())
    assert result.all_protected_passed
    assert result.all_passed


def test_exp008_candidate_gate_rejects_exp007_summary() -> None:
    with pytest.raises(ValueError, match="baseline summary is not from exp008"):
        evaluate_exp008_candidate_gate(
            baseline_summary=_summary(
                _scores(),
                experiment_id="exp007",
            ),
            candidate_summary=_summary(_scores(target=0.0)),
            minimum_baseline_score=0.95,
            minimum_regression_delta=0.15,
            maximum_unrelated_delta=0.05,
        )
