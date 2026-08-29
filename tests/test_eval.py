import pytest

from model_forensics.eval import (
    EvalCase,
    ProtocolThresholds,
    RunScores,
    assess_protocol,
    exact_match,
)


def test_exact_match_reports_score_and_failures() -> None:
    summary = exact_match(
        [
            EvalCase(case_id="a", expected="YES", observed="YES"),
            EvalCase(case_id="b", expected="NO", observed=" YES "),
        ]
    )

    assert summary.score == 0.5
    assert summary.total == 2
    assert summary.failed_case_ids == ("b",)


def test_exact_match_requires_cases() -> None:
    with pytest.raises(ValueError, match="at least one"):
        exact_match([])


def test_protocol_assessment_requires_regression_recovery_and_stability() -> None:
    assessment = assess_protocol(
        baseline=RunScores(target=0.95, unrelated=0.90),
        candidate=RunScores(target=0.65, unrelated=0.88),
        recovery=RunScores(target=0.89, unrelated=0.89),
        thresholds=ProtocolThresholds(
            minimum_baseline_score=0.80,
            minimum_regression_delta=0.15,
            minimum_recovery_delta=0.10,
            maximum_unrelated_delta=0.05,
        ),
    )

    assert assessment.regression_delta == pytest.approx(0.30)
    assert assessment.recovery_delta == pytest.approx(0.24)
    assert assessment.maximum_observed_unrelated_delta == pytest.approx(0.02)
    assert assessment.passed


def test_protocol_assessment_fails_unstable_negative_control() -> None:
    assessment = assess_protocol(
        baseline=RunScores(target=0.95, unrelated=0.95),
        candidate=RunScores(target=0.60, unrelated=0.80),
        recovery=RunScores(target=0.90, unrelated=0.94),
        thresholds=ProtocolThresholds(
            minimum_baseline_score=0.80,
            minimum_regression_delta=0.15,
            minimum_recovery_delta=0.10,
            maximum_unrelated_delta=0.05,
        ),
    )

    assert not assessment.unrelated_stability_passed
    assert not assessment.passed
