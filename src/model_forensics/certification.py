from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXP005_TARGET_SPLIT = "triangle_large"
EXP005_PROTECTED_SPLITS = (
    "circle_small",
    "circle_large",
    "square_small",
    "square_large",
    "triangle_small",
)


@dataclass(frozen=True)
class CandidateGateResult:
    """Frozen Experiment 005 clean/candidate behavioral gate result."""

    baseline_passed: bool
    target_regression: float
    target_regression_passed: bool
    protected_drifts: dict[str, float]
    protected_splits_passed: dict[str, bool]
    all_protected_passed: bool
    all_passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""

        return {
            "baseline_passed": self.baseline_passed,
            "target_split": EXP005_TARGET_SPLIT,
            "target_regression": self.target_regression,
            "target_regression_passed": self.target_regression_passed,
            "protected_drifts": self.protected_drifts,
            "protected_splits_passed": self.protected_splits_passed,
            "all_protected_passed": self.all_protected_passed,
            "all_passed": self.all_passed,
        }


def _label_scores(summary: dict[str, Any]) -> dict[str, float]:
    """Extract named label-accuracy scores from one adapter-evaluation summary."""

    try:
        raw_scores = summary["scores"]["label_accuracy"]
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation summary lacks scores.label_accuracy") from exc

    if not isinstance(raw_scores, dict):
        raise ValueError("scores.label_accuracy must be a mapping")

    scores: dict[str, float] = {}
    for split, payload in raw_scores.items():
        if not isinstance(payload, dict) or "score" not in payload:
            raise ValueError(f"label-accuracy split {split!r} lacks a score")
        scores[str(split)] = float(payload["score"])
    return scores


def evaluate_exp005_candidate_gate(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    maximum_unrelated_delta: float,
) -> CandidateGateResult:
    """Evaluate the frozen Experiment 005 localized-regression gate."""

    if baseline_summary.get("experiment_id") != "exp005":
        raise ValueError("baseline summary is not from Experiment 005")
    if candidate_summary.get("experiment_id") != "exp005":
        raise ValueError("candidate summary is not from Experiment 005")

    baseline_scores = _label_scores(baseline_summary)
    candidate_scores = _label_scores(candidate_summary)

    required = {EXP005_TARGET_SPLIT, *EXP005_PROTECTED_SPLITS, "all"}
    missing_baseline = sorted(required - set(baseline_scores))
    missing_candidate = sorted(required - set(candidate_scores))
    if missing_baseline or missing_candidate:
        raise ValueError(
            "missing required Experiment 005 evaluation splits; "
            f"baseline={missing_baseline} candidate={missing_candidate}"
        )

    baseline_passed = all(baseline_scores[split] >= minimum_baseline_score for split in required)

    target_regression = baseline_scores[EXP005_TARGET_SPLIT] - candidate_scores[EXP005_TARGET_SPLIT]
    target_regression_passed = target_regression >= minimum_regression_delta

    protected_drifts = {
        split: abs(candidate_scores[split] - baseline_scores[split])
        for split in EXP005_PROTECTED_SPLITS
    }
    protected_splits_passed = {
        split: drift <= maximum_unrelated_delta for split, drift in protected_drifts.items()
    }
    all_protected_passed = all(protected_splits_passed.values())

    return CandidateGateResult(
        baseline_passed=baseline_passed,
        target_regression=target_regression,
        target_regression_passed=target_regression_passed,
        protected_drifts=protected_drifts,
        protected_splits_passed=protected_splits_passed,
        all_protected_passed=all_protected_passed,
        all_passed=baseline_passed and target_regression_passed and all_protected_passed,
    )
