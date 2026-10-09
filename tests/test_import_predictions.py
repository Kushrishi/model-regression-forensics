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
