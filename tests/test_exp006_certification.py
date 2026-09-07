import json

import pytest

from model_forensics.certification import (
    evaluate_exp006_candidate_gate,
    evaluate_exp006_causal_certification,
    evaluate_exp006_order_control,
    exp006_public_certification_payload,
)

CANDIDATES = tuple(f"shard_semantic_{index:02d}" for index in range(1, 6))


def _summary(scores: dict[str, float], experiment_id: str = "exp006") -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
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


def _candidate_scores() -> dict[str, float]:
    scores = _clean_scores()
    scores["triangle_large"] = 0.75
    scores["all"] = 0.9583333333333334
    return scores


def test_exp006_candidate_gate_accepts_localized_target_regression() -> None:
    result = evaluate_exp006_candidate_gate(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.baseline_passed
    assert result.target_regression == 0.25
    assert result.target_regression_passed
    assert result.all_protected_passed
    assert result.all_passed


def test_exp006_candidate_gate_rejects_protected_drift() -> None:
    candidate = _candidate_scores()
    candidate["circle_small"] = 0.9375

    result = evaluate_exp006_candidate_gate(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(candidate),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.protected_drifts["circle_small"] == 0.0625
    assert not result.all_protected_passed
    assert not result.all_passed


def test_exp006_candidate_gate_rejects_wrong_experiment_summary() -> None:
    with pytest.raises(ValueError, match="not from exp006"):
        evaluate_exp006_candidate_gate(
            baseline_summary=_summary(_clean_scores(), experiment_id="exp005"),
            candidate_summary=_summary(_candidate_scores()),
            minimum_baseline_score=0.95,
            minimum_regression_delta=0.15,
            maximum_unrelated_delta=0.05,
        )


def test_exp006_causal_certification_requires_unique_planted_recovery() -> None:
    planted = CANDIDATES[2]
    restorations = {}

    for candidate_id in CANDIDATES:
        scores = _candidate_scores()
        if candidate_id == planted:
            scores["triangle_large"] = 1.0
            scores["all"] = 1.0
        restorations[candidate_id] = _summary(scores)

    result = evaluate_exp006_causal_certification(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        restoration_summaries=restorations,
        planted_candidate_id=planted,
        candidate_ids=CANDIDATES,
        minimum_recovery_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.planted_recovery_passed
    assert result.planted_protected_passed
    assert result.non_planted_recovery_passed
    assert result.non_planted_protected_passed
    assert result.unique_recovery_passed
    assert result.all_passed


def test_exp006_causal_certification_rejects_non_planted_recovery() -> None:
    planted = CANDIDATES[0]
    restorations = {candidate_id: _summary(_candidate_scores()) for candidate_id in CANDIDATES}

    planted_scores = _candidate_scores()
    planted_scores["triangle_large"] = 1.0
    planted_scores["all"] = 1.0
    restorations[planted] = _summary(planted_scores)

    contaminant = _candidate_scores()
    contaminant["triangle_large"] = 0.8125
    restorations[CANDIDATES[1]] = _summary(contaminant)

    result = evaluate_exp006_causal_certification(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        restoration_summaries=restorations,
        planted_candidate_id=planted,
        candidate_ids=CANDIDATES,
        minimum_recovery_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.target_recoveries[CANDIDATES[1]] == 0.0625
    assert not result.non_planted_recovery_passed
    assert not result.all_passed


def test_exp006_order_control_requires_candidate_gate_and_recovery() -> None:
    restoration = _candidate_scores()
    restoration["triangle_large"] = 1.0
    restoration["all"] = 1.0

    result = evaluate_exp006_order_control(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        planted_restoration_summary=_summary(restoration),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        minimum_recovery_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    assert result.clean_baseline_passed
    assert result.candidate_gate_passed
    assert result.planted_recovery_passed
    assert result.planted_protected_passed
    assert result.all_passed


def test_exp006_public_payload_does_not_leak_private_identity() -> None:
    planted = CANDIDATES[4]
    restorations = {}

    for candidate_id in CANDIDATES:
        scores = _candidate_scores()
        if candidate_id == planted:
            scores["triangle_large"] = 1.0
            scores["all"] = 1.0
        restorations[candidate_id] = _summary(scores)

    candidate_gate = evaluate_exp006_candidate_gate(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        maximum_unrelated_delta=0.05,
    )
    causal = evaluate_exp006_causal_certification(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        restoration_summaries=restorations,
        planted_candidate_id=planted,
        candidate_ids=CANDIDATES,
        minimum_recovery_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    order_restoration = _candidate_scores()
    order_restoration["triangle_large"] = 1.0
    order_restoration["all"] = 1.0
    order = evaluate_exp006_order_control(
        baseline_summary=_summary(_clean_scores()),
        candidate_summary=_summary(_candidate_scores()),
        planted_restoration_summary=_summary(order_restoration),
        minimum_baseline_score=0.95,
        minimum_regression_delta=0.15,
        minimum_recovery_delta=0.15,
        maximum_unrelated_delta=0.05,
    )

    payload = exp006_public_certification_payload(
        construction_gate_passed=True,
        candidate_gate=candidate_gate,
        causal_certification=causal,
        order_control=order,
    )
    rendered = json.dumps(payload, sort_keys=True)

    assert payload["experiment_id"] == "exp006"
    assert payload["benchmark_certified"] is True
    assert "shard_semantic" not in rendered
    assert "planted" not in rendered
    assert "world_seed" not in rendered
    assert "attempt" not in rendered
