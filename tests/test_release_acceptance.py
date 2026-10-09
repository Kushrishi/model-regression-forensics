import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_forensics.import_predictions import import_predictions
from model_forensics.investigation_report import reopen
from model_forensics.release_compare import Comparison, compare_releases
from model_forensics.repair_compare import RepairComparison, compare_repairs

EXAMPLE = Path(__file__).parents[1] / "examples/release_acceptance"


def test_complete_rollback_can_fail_a_required_capability(tmp_path):
    report = import_predictions(
        EXAMPLE / "policy.json", EXAMPLE / "predictions.csv", tmp_path / "r"
    )
    _, reopened, _ = reopen(tmp_path / "r")
    assert reopened["assessment"] == report["assessment"]
    assert report["assessment"]["successful_repairs"] == ["targeted_repair"]
    rollback = report["repairs"]["complete_rollback"]
    assert not rollback["passed"]
    new = next(row for row in rollback["slices"] if row["name"] == "Required new capability")
    assert new["accuracy_drop"] == 0.0
    assert new["maximum_accuracy_drop_passed"] is True
    assert new["minimum_accuracy_passed"] is False
    assert rollback["schema_version"] == "release-comparison-pilot/0.3"
    html = (tmp_path / "r/index.html").read_text()
    assert "Minimum accuracy" in html and "Minimum met" in html
    assert "100.00%" in html


def spec(floor=0.5):
    return {
        "cases": [{"case_id": "a", "expected": "yes"}, {"case_id": "b", "expected": "yes"}],
        "baseline": {
            "release_id": "baseline",
            "predictions": [
                {"case_id": "a", "observed": "yes"},
                {"case_id": "b", "observed": "yes"},
            ],
        },
        "candidate": {
            "release_id": "candidate",
            "predictions": [
                {"case_id": "a", "observed": "yes"},
                {"case_id": "b", "observed": "no"},
            ],
        },
        "slices": [
            {
                "name": "all",
                "case_ids": ["a", "b"],
                "maximum_accuracy_drop": 0.5,
                "minimum_accuracy": floor,
            }
        ],
    }


def compute(value):
    return compare_releases(Comparison.model_validate_json(json.dumps(value)))


def test_floor_boundary_and_relative_drop_both_apply():
    value = spec()
    assert compute(value)["passed"]
    value["slices"][0]["minimum_accuracy"] = 0.51
    assert not compute(value)["passed"]
    value["slices"][0].update(minimum_accuracy=0.0, maximum_accuracy_drop=0.0)
    assert not compute(value)["passed"]


@pytest.mark.parametrize("floor", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_floor_is_rejected(floor):
    with pytest.raises(ValidationError):
        compute(spec(floor))


def test_floor_is_bound_to_policy_and_cannot_change_for_a_repair():
    candidate = spec(1.0)
    repair = copy.deepcopy(candidate)
    repair["candidate"]["release_id"] = "repair"
    repair["slices"][0]["minimum_accuracy"] = 0.0
    assert (
        compute(candidate)["evaluation_policy_sha256"]
        != compute(repair)["evaluation_policy_sha256"]
    )
    with pytest.raises(ValueError, match="same baseline and evaluation policy"):
        compare_repairs(
            RepairComparison.model_validate_json(
                json.dumps({"regressed": candidate, "repairs": {"changed_floor": repair}})
            )
        )


def test_omitted_floor_preserves_legacy_report_and_digest():
    value = spec(None)
    omitted = copy.deepcopy(value)
    del omitted["slices"][0]["minimum_accuracy"]
    report = compute(value)
    assert report == compute(omitted)
    assert report["schema_version"] == "release-comparison-pilot/0.2"
    assert "minimum_accuracy" not in report["slices"][0]
