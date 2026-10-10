import csv
import json

import pytest

from model_forensics.import_predictions import import_predictions
from model_forensics.investigation_report import reopen


def inputs(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "cases": [{"case_id": "a", "expected": "yes"}, {"case_id": "b", "expected": "no"}],
                "slices": [{"name": "all", "case_ids": ["a", "b"], "maximum_accuracy_drop": 0.0}],
                "declared_changes": {"candidate": "<script>unsafe</script>"},
            }
        )
    )
    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["release_id", "case_id", "observed"])
        writer.writerows(
            [
                (r, c, label)
                for r, labels in [
                    ("baseline", ["yes", "no"]),
                    ("candidate", ["no", "no"]),
                    ("../repair", ["yes", "no"]),
                ]
                for c, label in zip(["a", "b"], labels, strict=True)
            ]
        )
    return policy, predictions


def test_import_roundtrip_and_unknown_cost(tmp_path):
    policy, predictions = inputs(tmp_path)
    output = tmp_path / "output"
    report = import_predictions(policy, predictions, output)
    plan, reopened, records = reopen(output)
    assert report["assessment"] == reopened["assessment"]
    assert report["assessment"]["successful_repairs"] == ["../repair"]
    assert plan["models_executed"] is False
    assert all(r["wall_seconds"] is None for r in records)
    assert (output / "source-predictions.csv").read_bytes() == predictions.read_bytes()
    page = (output / "index.html").read_text()
    assert "Unknown: these predictions were imported" in page
    assert "<script>unsafe</script>" not in page
    assert not (tmp_path / "repair").exists()
    with pytest.raises(FileExistsError):
        import_predictions(policy, predictions, output)


@pytest.mark.parametrize(
    "bad",
    [
        "baseline,a,yes\nbaseline,a,no\n",
        "baseline,a,yes,extra\n",
        "baseline,a,\n",
        "newrepair,unknown,yes\n",
    ],
)
def test_invalid_csv_leaves_no_output(tmp_path, bad):
    policy, predictions = inputs(tmp_path)
    with predictions.open("a") as stream:
        stream.write(bad)
    with pytest.raises(ValueError):
        import_predictions(policy, predictions, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_nonfinite_declaration_leaves_no_output(tmp_path):
    policy, predictions = inputs(tmp_path)
    value = json.loads(policy.read_text())
    value["declared_changes"] = {"bad": float("nan")}
    policy.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        import_predictions(policy, predictions, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("passing", [False, True])
def test_initial_comparison_reopens_without_claiming_repairs(tmp_path, passing):
    policy, predictions = inputs(tmp_path)
    rows = list(csv.DictReader(predictions.open()))
    with predictions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["release_id", "case_id", "observed"])
        writer.writeheader()
        for row in rows:
            if row["release_id"] not in {"baseline", "candidate"}:
                continue
            if passing and row["release_id"] == "candidate" and row["case_id"] == "a":
                row["observed"] = "yes"
            writer.writerow(row)
    output = tmp_path / "output"
    report = import_predictions(policy, predictions, output)
    assert report["candidate"]["passed"] is passing
    assert report["assessment"]["status"] == "not_evaluated"
    assert report["assessment"]["evaluated_repairs"] == []
    assert "regressed" not in report
    (output / "report.json").write_text('{"assessment": "forged"}')
    plan, reopened, records = reopen(output)
    assert reopened["candidate"] == report["candidate"]
    assert reopened["assessment"] == report["assessment"]
    assert len(records) == 2
    page = (output / "index.html").read_text()
    assert "Repairs have not been evaluated" in page
    assert "Successful repairs: None" not in page
    assert f"Candidate policy result: {'Pass' if passing else 'Fail'}" in page
    del plan["investigation_stage"]
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="invalid execution order"):
        reopen(output)


def test_initial_stage_cannot_hide_supplied_repairs(tmp_path):
    policy, predictions = inputs(tmp_path)
    output = tmp_path / "output"
    import_predictions(policy, predictions, output)
    plan = json.loads((output / "plan.json").read_text())
    plan["investigation_stage"] = "initial_comparison"
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="exactly baseline and candidate"):
        reopen(output)
