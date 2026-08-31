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


@dataclass(frozen=True)
class CausalCertificationResult:
    """Private five-way restoration certification result."""

    planted_candidate_id: str
    target_recoveries: dict[str, float]
    protected_drifts: dict[str, dict[str, float]]
    planted_recovery_passed: bool
    planted_protected_passed: bool
    non_planted_recovery_passed: bool
    non_planted_protected_passed: bool
    unique_recovery_passed: bool
    all_passed: bool

    def public_dict(self) -> dict[str, bool]:
        """Expose certification booleans without benchmark-private identity or scores."""

        return {
            "planted_recovery_passed": self.planted_recovery_passed,
            "planted_protected_passed": self.planted_protected_passed,
            "non_planted_recovery_passed": self.non_planted_recovery_passed,
            "non_planted_protected_passed": self.non_planted_protected_passed,
            "unique_recovery_passed": self.unique_recovery_passed,
            "all_passed": self.all_passed,
        }

    def private_dict(self) -> dict[str, object]:
        """Return full benchmark-private certification evidence."""

        return {
            **self.public_dict(),
            "planted_candidate_id": self.planted_candidate_id,
            "target_recoveries": self.target_recoveries,
            "protected_drifts": self.protected_drifts,
        }


@dataclass(frozen=True)
class OrderControlResult:
    """Prospectively frozen alternative training-order robustness result."""

    clean_baseline_passed: bool
    candidate_gate_passed: bool
    planted_recovery: float
    planted_recovery_passed: bool
    planted_protected_drifts: dict[str, float]
    planted_protected_passed: bool
    all_passed: bool

    def public_dict(self) -> dict[str, bool]:
        """Expose order-control booleans without private restoration scores."""

        return {
            "clean_baseline_passed": self.clean_baseline_passed,
            "candidate_gate_passed": self.candidate_gate_passed,
            "planted_recovery_passed": self.planted_recovery_passed,
            "planted_protected_passed": self.planted_protected_passed,
            "all_passed": self.all_passed,
        }

    def private_dict(self) -> dict[str, object]:
        """Return full benchmark-private order-control evidence."""

        return {
            **self.public_dict(),
            "planted_recovery": self.planted_recovery,
            "planted_protected_drifts": self.planted_protected_drifts,
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


def _validate_required_scores(
    scores: dict[str, float],
    *,
    label: str,
) -> None:
    required = {EXP005_TARGET_SPLIT, *EXP005_PROTECTED_SPLITS, "all"}
    missing = sorted(required - set(scores))
    if missing:
        raise ValueError(f"{label} is missing required Experiment 005 evaluation splits: {missing}")


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

    _validate_required_scores(baseline_scores, label="baseline")
    _validate_required_scores(candidate_scores, label="candidate")

    required = {EXP005_TARGET_SPLIT, *EXP005_PROTECTED_SPLITS, "all"}
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


def evaluate_exp005_causal_certification(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    restoration_summaries: dict[str, dict[str, Any]],
    planted_candidate_id: str,
    candidate_ids: tuple[str, ...],
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> CausalCertificationResult:
    """Evaluate the private five-restoration uniqueness and recovery gates."""

    if set(restoration_summaries) != set(candidate_ids):
        raise ValueError("restoration summaries must exactly match frozen candidate IDs")
    if planted_candidate_id not in candidate_ids:
        raise ValueError("planted candidate ID is not among frozen candidate IDs")

    baseline_scores = _label_scores(baseline_summary)
    candidate_scores = _label_scores(candidate_summary)
    _validate_required_scores(baseline_scores, label="baseline")
    _validate_required_scores(candidate_scores, label="candidate")

    target_recoveries: dict[str, float] = {}
    protected_drifts: dict[str, dict[str, float]] = {}

    for candidate_id in candidate_ids:
        restoration_scores = _label_scores(restoration_summaries[candidate_id])
        _validate_required_scores(
            restoration_scores,
            label=f"restoration {candidate_id}",
        )
        target_recoveries[candidate_id] = (
            restoration_scores[EXP005_TARGET_SPLIT] - candidate_scores[EXP005_TARGET_SPLIT]
        )
        protected_drifts[candidate_id] = {
            split: abs(restoration_scores[split] - baseline_scores[split])
            for split in EXP005_PROTECTED_SPLITS
        }

    planted_recovery_passed = target_recoveries[planted_candidate_id] >= minimum_recovery_delta
    planted_protected_passed = all(
        drift <= maximum_unrelated_delta
        for drift in protected_drifts[planted_candidate_id].values()
    )

    non_planted = tuple(
        candidate_id for candidate_id in candidate_ids if candidate_id != planted_candidate_id
    )
    non_planted_recovery_passed = all(
        target_recoveries[candidate_id] <= maximum_unrelated_delta for candidate_id in non_planted
    )
    non_planted_protected_passed = all(
        drift <= maximum_unrelated_delta
        for candidate_id in non_planted
        for drift in protected_drifts[candidate_id].values()
    )

    recovery_qualifiers = [
        candidate_id
        for candidate_id in candidate_ids
        if target_recoveries[candidate_id] >= minimum_recovery_delta
    ]
    unique_recovery_passed = recovery_qualifiers == [planted_candidate_id]

    all_passed = (
        planted_recovery_passed
        and planted_protected_passed
        and non_planted_recovery_passed
        and non_planted_protected_passed
        and unique_recovery_passed
    )

    return CausalCertificationResult(
        planted_candidate_id=planted_candidate_id,
        target_recoveries=target_recoveries,
        protected_drifts=protected_drifts,
        planted_recovery_passed=planted_recovery_passed,
        planted_protected_passed=planted_protected_passed,
        non_planted_recovery_passed=non_planted_recovery_passed,
        non_planted_protected_passed=non_planted_protected_passed,
        unique_recovery_passed=unique_recovery_passed,
        all_passed=all_passed,
    )


def evaluate_exp005_order_control(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    planted_restoration_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> OrderControlResult:
    """Evaluate the frozen alternative-order clean/candidate/planted-restoration gates."""

    candidate_gate = evaluate_exp005_candidate_gate(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )

    baseline_scores = _label_scores(baseline_summary)
    candidate_scores = _label_scores(candidate_summary)
    restoration_scores = _label_scores(planted_restoration_summary)
    _validate_required_scores(restoration_scores, label="order-control restoration")

    planted_recovery = (
        restoration_scores[EXP005_TARGET_SPLIT] - candidate_scores[EXP005_TARGET_SPLIT]
    )
    planted_recovery_passed = planted_recovery >= minimum_recovery_delta
    planted_protected_drifts = {
        split: abs(restoration_scores[split] - baseline_scores[split])
        for split in EXP005_PROTECTED_SPLITS
    }
    planted_protected_passed = all(
        drift <= maximum_unrelated_delta for drift in planted_protected_drifts.values()
    )

    return OrderControlResult(
        clean_baseline_passed=candidate_gate.baseline_passed,
        candidate_gate_passed=candidate_gate.all_passed,
        planted_recovery=planted_recovery,
        planted_recovery_passed=planted_recovery_passed,
        planted_protected_drifts=planted_protected_drifts,
        planted_protected_passed=planted_protected_passed,
        all_passed=(
            candidate_gate.all_passed and planted_recovery_passed and planted_protected_passed
        ),
    )


def exp005_public_certification_payload(
    *,
    construction_gate_passed: bool,
    candidate_gate: CandidateGateResult,
    causal_certification: CausalCertificationResult,
    order_control: OrderControlResult,
) -> dict[str, object]:
    """Return only the protocol-approved public certification boundary."""

    return {
        "experiment_id": "exp005",
        "construction_gate_passed": construction_gate_passed,
        "clean_baseline_gate_passed": candidate_gate.baseline_passed,
        "localized_regression_gate_passed": (
            candidate_gate.target_regression_passed and candidate_gate.all_protected_passed
        ),
        "unique_causal_certification_passed": causal_certification.all_passed,
        "order_robustness_passed": order_control.all_passed,
        "benchmark_certified": (
            construction_gate_passed
            and candidate_gate.all_passed
            and causal_certification.all_passed
            and order_control.all_passed
        ),
    }
