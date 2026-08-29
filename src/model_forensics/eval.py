from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class EvalCase:
    """One deterministic behavioral evaluation case."""

    case_id: str
    expected: str
    observed: str


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate score plus failing case identifiers."""

    score: float
    total: int
    failed_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunScores:
    """Target and unrelated evaluation scores for one model run."""

    target: float
    unrelated: float

    def __post_init__(self) -> None:
        for name, value in (("target", self.target), ("unrelated", self.unrelated)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} score must be between 0 and 1")


@dataclass(frozen=True)
class ProtocolThresholds:
    """Minimum evidence required for Experiment 000 to pass."""

    minimum_baseline_score: float
    minimum_regression_delta: float
    minimum_recovery_delta: float
    maximum_unrelated_delta: float

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_baseline_score", self.minimum_baseline_score),
            ("minimum_regression_delta", self.minimum_regression_delta),
            ("minimum_recovery_delta", self.minimum_recovery_delta),
            ("maximum_unrelated_delta", self.maximum_unrelated_delta),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class ProtocolAssessment:
    """Computed deltas and pass/fail checks for the intervention protocol."""

    baseline_score: float
    regression_delta: float
    recovery_delta: float
    maximum_observed_unrelated_delta: float
    baseline_passed: bool
    regression_passed: bool
    recovery_passed: bool
    unrelated_stability_passed: bool

    @property
    def passed(self) -> bool:
        """Return true only when every protocol criterion passes."""

        return all(
            (
                self.baseline_passed,
                self.regression_passed,
                self.recovery_passed,
                self.unrelated_stability_passed,
            )
        )


def exact_match(cases: list[EvalCase]) -> EvalSummary:
    """Score exact normalized string agreement."""

    if not cases:
        raise ValueError("at least one evaluation case is required")

    matches = [case.expected.strip() == case.observed.strip() for case in cases]
    failed = tuple(
        case.case_id for case, matched in zip(cases, matches, strict=True) if not matched
    )
    return EvalSummary(score=mean(matches), total=len(cases), failed_case_ids=failed)


def assess_protocol(
    *,
    baseline: RunScores,
    candidate: RunScores,
    recovery: RunScores,
    thresholds: ProtocolThresholds,
) -> ProtocolAssessment:
    """Evaluate regression, recovery, and negative-control stability."""

    regression_delta = baseline.target - candidate.target
    recovery_delta = recovery.target - candidate.target
    unrelated_delta = max(
        abs(candidate.unrelated - baseline.unrelated),
        abs(recovery.unrelated - baseline.unrelated),
    )

    return ProtocolAssessment(
        baseline_score=baseline.target,
        regression_delta=regression_delta,
        recovery_delta=recovery_delta,
        maximum_observed_unrelated_delta=unrelated_delta,
        baseline_passed=baseline.target >= thresholds.minimum_baseline_score,
        regression_passed=regression_delta >= thresholds.minimum_regression_delta,
        recovery_passed=recovery_delta >= thresholds.minimum_recovery_delta,
        unrelated_stability_passed=unrelated_delta <= thresholds.maximum_unrelated_delta,
    )
