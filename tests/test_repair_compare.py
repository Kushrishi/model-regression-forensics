import copy
import json
import subprocess
import sys

import pytest

from model_forensics.release_compare import Comparison, compare_releases
from model_forensics.repair_compare import RepairComparison, compare_repairs
from model_forensics.specificity import assess_repairs


def record(candidate="regressed", repaired=False):
    return {
        "cases": [{"case_id": "a", "expected": "yes"}, {"case_id": "b", "expected": "no"}],
        "baseline": {
            "release_id": "baseline",
            "predictions": [
                {"case_id": "a", "observed": "yes"},
                {"case_id": "b", "observed": "no"},
            ],
        },
        "candidate": {
            "release_id": candidate,
            "predictions": [
                {"case_id": "a", "observed": "yes" if repaired else "no"},
                {"case_id": "b", "observed": "no"},
            ],
        },
        "slices": [{"name": "all", "case_ids": ["a", "b"]}],
    }


def report(value):
    return compare_releases(Comparison.model_validate_json(json.dumps(value)))


@pytest.mark.parametrize("change", ["cases", "labels", "baseline", "membership"])
def test_equal_summary_statistics_do_not_make_policies_interchangeable(change):
    before = record()
    repair = record("repair", True)
    if change == "cases":
        repair = json.loads(json.dumps(repair).replace('"a"', '"different-case"'))
    elif change == "labels":
        repair["cases"][0]["expected"] = "renamed-label"
        repair["baseline"]["predictions"][0]["observed"] = "renamed-label"
        repair["candidate"]["predictions"][0]["observed"] = "renamed-label"
    elif change == "baseline":
        before["baseline"]["predictions"][0]["observed"] = "no"
        before["candidate"]["predictions"][1]["observed"] = "yes"
        repair["baseline"]["predictions"][1]["observed"] = "yes"
    else:
        before["slices"][0]["case_ids"] = ["a"]
        repair["slices"][0]["case_ids"] = ["b"]
    assert report(before)["slices"][0]["count"] == report(repair)["slices"][0]["count"]
    assert (
        report(before)["slices"][0]["baseline_accuracy"]
        == report(repair)["slices"][0]["baseline_accuracy"]
    )
    with pytest.raises(ValueError, match="same baseline and evaluation policy"):
        assess_repairs(report(before), {"repair": report(repair)})


def test_legacy_reports_fail_closed():
    legacy = report(record())
    del legacy["evaluation_policy_sha256"]
    with pytest.raises(ValueError, match="evaluation_policy_sha256"):
        assess_repairs(legacy, {})


def test_full_saved_prediction_workflow_and_cli(tmp_path):
    value = {
        "regressed": record(),
        "repairs": {
            "restore-input-order": record("repair-a", True),
            "reverse-model-weights": record("repair-b", True),
        },
    }
    expected = compare_repairs(RepairComparison.model_validate_json(json.dumps(value)))
    assert expected["assessment"]["status"] == "ambiguous_repairs"
    assert expected["assessment"]["historical_cause"] == "not_identified"
    reordered = copy.deepcopy(value)
    for spec in [reordered["regressed"], *reordered["repairs"].values()]:
        spec["cases"].reverse()
        spec["baseline"]["predictions"].reverse()
        spec["slices"][0]["case_ids"].reverse()
    assert compare_repairs(RepairComparison.model_validate_json(json.dumps(reordered))) == expected
    path = tmp_path / "repairs.json"
    path.write_text(json.dumps(value))
    completed = subprocess.run(
        [sys.executable, "-m", "model_forensics.repair_compare", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == expected
    value["repairs"]["restore-input-order"]["slices"][0]["maximum_accuracy_drop"] = 1.0
    path.write_text(json.dumps(value))
    rejected = subprocess.run(
        [sys.executable, "-m", "model_forensics.repair_compare", str(path)],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert rejected.stdout == ""
