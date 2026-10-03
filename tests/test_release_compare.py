import copy
import json

import pytest
from pydantic import ValidationError

from model_forensics.release_compare import Comparison, compare_releases


@pytest.fixture
def record():
    return {
        "cases": [{"case_id": "a", "expected": "yes"}, {"case_id": "b", "expected": "no"}],
        "baseline": {
            "release_id": "v1",
            "predictions": [
                {"case_id": "a", "observed": "yes"},
                {"case_id": "b", "observed": "yes"},
            ],
        },
        "candidate": {
            "release_id": "v2",
            "predictions": [
                {"case_id": "b", "observed": "no"},
                {"case_id": "a", "observed": "no"},
            ],
        },
        "slices": [
            {"name": "target", "case_ids": ["a"], "maximum_accuracy_drop": 0.0},
            {"name": "protected", "case_ids": ["b"], "maximum_accuracy_drop": 0.0},
        ],
    }


def report(record):
    return compare_releases(Comparison.model_validate_json(json.dumps(record)))


def test_slice_regression_is_not_hidden_by_unchanged_overall_accuracy(record):
    result = report(record)
    assert result["passed"] is False
    protected, target = result["slices"]
    assert protected["accuracy_drop"] == -1.0
    assert protected["improved_case_ids"] == ["b"]
    assert target["regressed_case_ids"] == ["a"]
    assert result["causal_attribution"] == "not_assessed"
    assert result["statistical_significance"] == "not_assessed"


def test_order_does_not_change_report_or_identity(record):
    reordered = copy.deepcopy(record)
    reordered["cases"].reverse()
    reordered["baseline"]["predictions"].reverse()
    reordered["candidate"]["predictions"].reverse()
    reordered["slices"].reverse()
    assert report(record) == report(reordered)


def test_policy_and_predictions_are_part_of_identity(record):
    first = report(record)
    record["slices"][0]["maximum_accuracy_drop"] = 1.0
    assert report(record)["passed"] is True
    assert first["input_sha256"] != report(record)["input_sha256"]
    record["candidate"]["predictions"][0]["observed"] = "yes"
    assert first["input_sha256"] != report(record)["input_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["cases"].append(r["cases"][0]),
        lambda r: r["candidate"]["predictions"].pop(),
        lambda r: r["candidate"]["predictions"].append(r["candidate"]["predictions"][0]),
        lambda r: r["candidate"].update(release_id="v1"),
        lambda r: r["slices"][0].update(case_ids=["unknown"]),
        lambda r: r["slices"][0].update(case_ids=["a", "a"]),
        lambda r: r["slices"][0].update(case_ids=[]),
        lambda r: r["slices"][0].update(maximum_accuracy_drop=-0.1),
        lambda r: r["slices"].append(r["slices"][0]),
        lambda r: r.update(hidden_truth="root"),
        lambda r: r["cases"][0].update(expected=1),
        lambda r: r["cases"][0].update(case_id=" "),
    ],
)
def test_rejects_ambiguous_or_invalid_inputs(record, mutation):
    mutation(record)
    with pytest.raises(ValidationError):
        report(record)


def test_exact_labels_are_not_silently_normalized(record):
    record["baseline"]["predictions"][0]["observed"] = "yes "
    assert report(record)["slices"][1]["baseline_accuracy"] == 0.0


def test_overlapping_slices_are_reported_separately(record):
    record["slices"].append({"name": "all", "case_ids": ["b", "a"]})
    result = report(record)
    assert result["slices"][0]["accuracy_drop"] == 0.0
    assert result["slices"][0]["regressed_case_ids"] == ["a"]
    assert result["slices"][0]["improved_case_ids"] == ["b"]
