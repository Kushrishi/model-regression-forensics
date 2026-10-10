"""Replay the real saved conversion predictions without numerical dependencies."""

import json
from pathlib import Path

from model_forensics.import_predictions import import_predictions
from model_forensics.investigation_report import render, reopen


def test_saved_deployment_case_counts_and_policy(tmp_path):
    fixture = Path(__file__).resolve().parents[1] / "examples" / "deployment_comparison"
    output = tmp_path / "investigation"
    report = import_predictions(fixture / "policy.json", fixture / "predictions.csv", output)
    plan, reopened, records = reopen(output)
    assert len(plan["cases"]) == 1969
    assert report["candidate"] == reopened["candidate"]
    assert report["candidate"]["passed"] is True
    assert report["assessment"]["status"] == "not_evaluated"
    assert len(records) == 2
    rows = {r["name"]: r for r in report["candidate"]["slices"]}
    assert len(rows) == 78
    assert rows["all"]["count"] == 1969
    assert len(rows["all"]["regressed_case_ids"]) == 3
    assert len(rows["all"]["improved_case_ids"]) == 3
    assert rows["all"]["accuracy_drop"] == 0.0
    before, after = [
        {r["case_id"]: r["observed"] for r in record["release"]["predictions"]}
        for record in records
    ]
    assert sum(before[k] != after[k] for k in before) == 8
    for name in ["unable_to_verify_identity", "verify_my_identity"]:
        assert rows[name]["count"] == 20
        assert rows[name]["accuracy_drop"] == 0.05
        assert rows[name]["passed"] is True
    previews = json.loads((fixture / "case_inputs.json").read_text())
    assert set(previews) == set(before)
    (output / "case_inputs.json").write_bytes((fixture / "case_inputs.json").read_bytes())
    render(output, output / "inspection.html")
    page = (output / "inspection.html").read_text()
    assert "Imported source integrity checked" in page
    assert "Repairs have not been evaluated" in page
