"""Deterministic ambiguity fixture for repair evidence.

A feature-order regression has two distinct behavior-restoring interventions:
restore the original input order, or keep the reversed inputs and reverse the
linear model weights. Both produce the same predictions, so behavior alone
cannot distinguish the historical explanation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_forensics.release_compare import Comparison, compare_releases  # noqa: E402
from model_forensics.specificity import assess_repairs  # noqa: E402

WEIGHTS = (2.0, -1.0, 0.5)
CASES = (
    ("a", (-2.0, -2.0, 0.0)),
    ("b", (-2.0, 0.0, 2.0)),
    ("c", (-1.0, 2.0, 1.0)),
    ("d", (0.0, -2.0, 2.0)),
    ("e", (1.0, -1.0, -2.0)),
    ("f", (2.0, 0.0, -1.0)),
    ("g", (2.0, 2.0, 0.0)),
    ("h", (1.0, 1.0, 2.0)),
)


def predict(vector, weights):
    score = sum(x * w for x, w in zip(vector, weights, strict=True))
    return "positive" if score >= 0 else "negative"


def predictions(transform, weights):
    return [
        {"case_id": case_id, "observed": predict(transform(vector), weights)}
        for case_id, vector in CASES
    ]


def comparison(candidate_id, candidate_predictions):
    baseline_predictions = predictions(lambda vector: vector, WEIGHTS)
    record = {
        "cases": [
            {"case_id": case_id, "expected": item["observed"]}
            for (case_id, _), item in zip(CASES, baseline_predictions, strict=True)
        ],
        "baseline": {
            "release_id": "baseline-original-order",
            "predictions": baseline_predictions,
        },
        "candidate": {
            "release_id": candidate_id,
            "predictions": candidate_predictions,
        },
        "slices": [
            {
                "name": "all",
                "case_ids": [case_id for case_id, _ in CASES],
                "maximum_accuracy_drop": 0.0,
            }
        ],
    }
    return compare_releases(Comparison.model_validate(record))


def reverse(vector):
    return tuple(reversed(vector))


def run():
    regressed = comparison("regressed-reversed-inputs", predictions(reverse, WEIGHTS))

    restore_input_order = comparison(
        "repair-restore-input-order",
        predictions(lambda vector: vector, WEIGHTS),
    )
    compensate_weights = comparison(
        "repair-reverse-weights",
        predictions(reverse, tuple(reversed(WEIGHTS))),
    )

    assert regressed["passed"] is False
    assert restore_input_order["passed"] is True
    assert compensate_weights["passed"] is True

    assessment = assess_repairs(
        regressed,
        {
            "restore-input-order": restore_input_order,
            "reverse-model-weights": compensate_weights,
        },
    )
    assert assessment["status"] == "ambiguous_repairs"

    print(
        json.dumps(
            {
                "regression_report": regressed,
                "repair_assessment": assessment,
                "fixture_truth": {
                    "historical_change": "input-order-reversal",
                    "truth_used_by_assessment": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
