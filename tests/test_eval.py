import pytest

from model_forensics.eval import EvalCase, exact_match


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
