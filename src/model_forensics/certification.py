from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TARGET_SPLIT = "triangle_large"
PROTECTED_SPLITS = (
    "circle_small",
    "circle_large",
    "square_small",
    "square_large",
    "triangle_small",
)

# Backward-compatible Experiment 005 aliases.
EXP005_TARGET_SPLIT = TARGET_SPLIT
EXP005_PROTECTED_SPLITS = PROTECTED_SPLITS


@dataclass(frozen=True)
class CandidateGateResult:
    """Clean/candidate localized-regression gate result."""

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
            "target_split": TARGET_SPLIT,
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
    experiment_id: str,
) -> None:
    required = {TARGET_SPLIT, *PROTECTED_SPLITS, "all"}
    missing = sorted(required - set(scores))
    if missing:
        experiment_label = f"Experiment {experiment_id.removeprefix('exp')}"
        raise ValueError(
            f"{label} is missing required {experiment_label} evaluation splits: {missing}"
        )


def _evaluate_candidate_gate(
    *,
    experiment_id: str,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    maximum_unrelated_delta: float,
) -> CandidateGateResult:
    if baseline_summary.get("experiment_id") != experiment_id:
        raise ValueError(f"baseline summary is not from {experiment_id}")
    if candidate_summary.get("experiment_id") != experiment_id:
        raise ValueError(f"candidate summary is not from {experiment_id}")

    baseline_scores = _label_scores(baseline_summary)
    candidate_scores = _label_scores(candidate_summary)

    _validate_required_scores(
        baseline_scores,
        label="baseline",
        experiment_id=experiment_id,
    )
    _validate_required_scores(
        candidate_scores,
        label="candidate",
        experiment_id=experiment_id,
    )

    required = {TARGET_SPLIT, *PROTECTED_SPLITS, "all"}
    baseline_passed = all(baseline_scores[split] >= minimum_baseline_score for split in required)

    target_regression = baseline_scores[TARGET_SPLIT] - candidate_scores[TARGET_SPLIT]
    target_regression_passed = target_regression >= minimum_regression_delta

    protected_drifts = {
        split: abs(candidate_scores[split] - baseline_scores[split]) for split in PROTECTED_SPLITS
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


def _evaluate_causal_certification(
    *,
    experiment_id: str,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    restoration_summaries: dict[str, dict[str, Any]],
    planted_candidate_id: str,
    candidate_ids: tuple[str, ...],
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> CausalCertificationResult:
    if set(restoration_summaries) != set(candidate_ids):
        raise ValueError("restoration summaries must exactly match frozen candidate IDs")
    if planted_candidate_id not in candidate_ids:
        raise ValueError("planted candidate ID is not among frozen candidate IDs")
    if baseline_summary.get("experiment_id") != experiment_id:
        raise ValueError(f"baseline summary is not from {experiment_id}")
    if candidate_summary.get("experiment_id") != experiment_id:
        raise ValueError(f"candidate summary is not from {experiment_id}")

    baseline_scores = _label_scores(baseline_summary)
    candidate_scores = _label_scores(candidate_summary)
    _validate_required_scores(
        baseline_scores,
        label="baseline",
        experiment_id=experiment_id,
    )
    _validate_required_scores(
        candidate_scores,
        label="candidate",
        experiment_id=experiment_id,
    )

    target_recoveries: dict[str, float] = {}
    protected_drifts: dict[str, dict[str, float]] = {}

    for candidate_id in candidate_ids:
        restoration_summary = restoration_summaries[candidate_id]
        if restoration_summary.get("experiment_id") != experiment_id:
            raise ValueError(f"restoration {candidate_id} summary is not from {experiment_id}")
        restoration_scores = _label_scores(restoration_summary)
        _validate_required_scores(
            restoration_scores,
            label=f"restoration {candidate_id}",
            experiment_id=experiment_id,
        )
        target_recoveries[candidate_id] = (
            restoration_scores[TARGET_SPLIT] - candidate_scores[TARGET_SPLIT]
        )
        protected_drifts[candidate_id] = {
            split: abs(restoration_scores[split] - baseline_scores[split])
            for split in PROTECTED_SPLITS
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


def _evaluate_order_control(
    *,
    experiment_id: str,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    planted_restoration_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> OrderControlResult:
    candidate_gate = _evaluate_candidate_gate(
        experiment_id=experiment_id,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )
    if planted_restoration_summary.get("experiment_id") != experiment_id:
        raise ValueError(f"order-control restoration summary is not from {experiment_id}")

    baseline_scores = _label_scores(baseline_summary)
    candidate_scores = _label_scores(candidate_summary)
    restoration_scores = _label_scores(planted_restoration_summary)
    _validate_required_scores(
        restoration_scores,
        label="order-control restoration",
        experiment_id=experiment_id,
    )

    planted_recovery = restoration_scores[TARGET_SPLIT] - candidate_scores[TARGET_SPLIT]
    planted_recovery_passed = planted_recovery >= minimum_recovery_delta
    planted_protected_drifts = {
        split: abs(restoration_scores[split] - baseline_scores[split]) for split in PROTECTED_SPLITS
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


def _public_certification_payload(
    *,
    experiment_id: str,
    construction_gate_passed: bool,
    candidate_gate: CandidateGateResult,
    causal_certification: CausalCertificationResult,
    order_control: OrderControlResult,
) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
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


def evaluate_exp005_candidate_gate(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    maximum_unrelated_delta: float,
) -> CandidateGateResult:
    """Evaluate the frozen Experiment 005 localized-regression gate."""

    return _evaluate_candidate_gate(
        experiment_id="exp005",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
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
    """Evaluate the private Experiment 005 five-restoration gates."""

    return _evaluate_causal_certification(
        experiment_id="exp005",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        restoration_summaries=restoration_summaries,
        planted_candidate_id=planted_candidate_id,
        candidate_ids=candidate_ids,
        minimum_recovery_delta=minimum_recovery_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
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
    """Evaluate the frozen Experiment 005 alternative-order gates."""

    return _evaluate_order_control(
        experiment_id="exp005",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        planted_restoration_summary=planted_restoration_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        minimum_recovery_delta=minimum_recovery_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def exp005_public_certification_payload(
    *,
    construction_gate_passed: bool,
    candidate_gate: CandidateGateResult,
    causal_certification: CausalCertificationResult,
    order_control: OrderControlResult,
) -> dict[str, object]:
    """Return only the Experiment 005 protocol-approved public boundary."""

    return _public_certification_payload(
        experiment_id="exp005",
        construction_gate_passed=construction_gate_passed,
        candidate_gate=candidate_gate,
        causal_certification=causal_certification,
        order_control=order_control,
    )


def evaluate_exp006_candidate_gate(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    maximum_unrelated_delta: float,
) -> CandidateGateResult:
    """Evaluate the frozen Experiment 006 localized-regression gate."""

    return _evaluate_candidate_gate(
        experiment_id="exp006",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def evaluate_exp006_causal_certification(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    restoration_summaries: dict[str, dict[str, Any]],
    planted_candidate_id: str,
    candidate_ids: tuple[str, ...],
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> CausalCertificationResult:
    """Evaluate the private Experiment 006 five-restoration gates."""

    return _evaluate_causal_certification(
        experiment_id="exp006",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        restoration_summaries=restoration_summaries,
        planted_candidate_id=planted_candidate_id,
        candidate_ids=candidate_ids,
        minimum_recovery_delta=minimum_recovery_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def evaluate_exp006_order_control(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    planted_restoration_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> OrderControlResult:
    """Evaluate the frozen Experiment 006 alternative-order gates."""

    return _evaluate_order_control(
        experiment_id="exp006",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        planted_restoration_summary=planted_restoration_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        minimum_recovery_delta=minimum_recovery_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def exp006_public_certification_payload(
    *,
    construction_gate_passed: bool,
    candidate_gate: CandidateGateResult,
    causal_certification: CausalCertificationResult,
    order_control: OrderControlResult,
) -> dict[str, object]:
    """Return only the Experiment 006 protocol-approved public boundary."""

    return _public_certification_payload(
        experiment_id="exp006",
        construction_gate_passed=construction_gate_passed,
        candidate_gate=candidate_gate,
        causal_certification=causal_certification,
        order_control=order_control,
    )


def evaluate_exp007_candidate_gate(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    maximum_unrelated_delta: float,
) -> CandidateGateResult:
    """Evaluate the frozen Experiment 007 localized-regression gate."""

    return _evaluate_candidate_gate(
        experiment_id="exp007",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def evaluate_exp007_causal_certification(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    restoration_summaries: dict[str, dict[str, Any]],
    planted_candidate_id: str,
    candidate_ids: tuple[str, ...],
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> CausalCertificationResult:
    """Evaluate the private Experiment 007 five-restoration gates."""

    return _evaluate_causal_certification(
        experiment_id="exp007",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        restoration_summaries=restoration_summaries,
        planted_candidate_id=planted_candidate_id,
        candidate_ids=candidate_ids,
        minimum_recovery_delta=minimum_recovery_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def evaluate_exp007_order_control(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    planted_restoration_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    minimum_recovery_delta: float,
    maximum_unrelated_delta: float,
) -> OrderControlResult:
    """Evaluate the frozen Experiment 007 alternative-order gates."""

    return _evaluate_order_control(
        experiment_id="exp007",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        planted_restoration_summary=planted_restoration_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        minimum_recovery_delta=minimum_recovery_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )


def exp007_public_certification_payload(
    *,
    construction_gate_passed: bool,
    candidate_gate: CandidateGateResult,
    causal_certification: CausalCertificationResult,
    order_control: OrderControlResult,
) -> dict[str, object]:
    """Return only the Experiment 007 protocol-approved public boundary."""

    return _public_certification_payload(
        experiment_id="exp007",
        construction_gate_passed=construction_gate_passed,
        candidate_gate=candidate_gate,
        causal_certification=causal_certification,
        order_control=order_control,
    )


def evaluate_exp008_candidate_gate(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    minimum_baseline_score: float,
    minimum_regression_delta: float,
    maximum_unrelated_delta: float,
) -> CandidateGateResult:
    """Evaluate the frozen Experiment 008 localized-regression gate."""

    return _evaluate_candidate_gate(
        experiment_id="exp008",
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        minimum_baseline_score=minimum_baseline_score,
        minimum_regression_delta=minimum_regression_delta,
        maximum_unrelated_delta=maximum_unrelated_delta,
    )
