import json
from pathlib import Path

import pytest

from model_forensics.decision_comparison import compare_decision_paths, ordinary_diff

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "example", ["deployment_comparison", "release_acceptance", "initial_comparison"]
)
def test_retained_inputs_agree_with_ordinary_diff(example, tmp_path):
    fixture = ROOT / "examples" / example
    report = compare_decision_paths(
        fixture / "policy.json", fixture / "predictions.csv", tmp_path / "out"
    )
    assert report["all_supplied_decisions_agree"]
    assert report["historical_cause"] == "not_identified"
    assert report["independent_user_benefit"] == "not_measured"
    assert json.loads((tmp_path / "out/comparison.json").read_text()) == report
    if example == "deployment_comparison":
        diff = report["ordinary_diff"]["candidate"]
        assert diff["passed"]
        assert len(diff["changed_case_ids"]) == 8
        assert len(diff["new_error_case_ids"]) == 3
        assert len(diff["corrected_case_ids"]) == 3
        assert report["mrf_repair_assessment"]["status"] == "not_evaluated"
    if example == "release_acceptance":
        assert not report["complete_rollback"]["passed"]
    with pytest.raises(FileExistsError):
        compare_decision_paths(
            fixture / "policy.json", fixture / "predictions.csv", tmp_path / "out"
        )


def test_exact_drop_boundary_and_partial_slice_coverage(tmp_path):
    policy = {
        "cases": [{"case_id": str(i), "expected": "yes"} for i in range(21)],
        "slices": [
            {
                "name": "covered",
                "case_ids": [str(i) for i in range(20)],
                "maximum_accuracy_drop": 0.05,
            }
        ],
        "declared_changes": {},
    }
    content = "release_id,case_id,observed\n" + "".join(
        f"{release},{i},{'no' if release == 'candidate' and i in (0, 20) else 'yes'}\n"
        for release in ("baseline", "candidate")
        for i in range(21)
    )
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    (tmp_path / "predictions.csv").write_text(content)
    report = compare_decision_paths(
        tmp_path / "policy.json", tmp_path / "predictions.csv", tmp_path / "out"
    )
    assert report["all_supplied_decisions_agree"]
    assert report["ordinary_diff"]["candidate"]["passed"]
    assert report["ordinary_diff"]["candidate"]["new_error_case_ids"] == ["0", "20"]


def test_ordinary_diff_rejects_duplicate_prediction():
    policy = {"cases": [{"case_id": "a", "expected": "yes"}], "slices": []}
    with pytest.raises(ValueError, match="duplicate"):
        ordinary_diff(
            policy, b"release_id,case_id,observed\nbaseline,a,yes\nbaseline,a,no\ncandidate,a,yes\n"
        )
