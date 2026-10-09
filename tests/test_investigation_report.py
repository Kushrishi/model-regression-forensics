import json

import pytest

from model_forensics.investigation import run_investigation
from model_forensics.investigation_report import render, reopen
from model_forensics.release_compare import Case, Prediction, Slice


def attempt(path):
    cases = (Case(case_id="<script>alert(1)</script>", expected="yes"),)
    good = (Prediction(case_id=cases[0].case_id, observed="yes"),)
    bad = (Prediction(case_id=cases[0].case_id, observed="no"),)
    run_investigation(
        path,
        cases=cases,
        slices=(Slice(name="all", case_ids=(cases[0].case_id,)),),
        baseline=lambda: good,
        candidate=lambda: bad,
        repairs={"restore": lambda: good},
        changes={"text": "<script>bad</script>"},
    )


def test_recomputes_instead_of_trusting_saved_report(tmp_path):
    attempt(tmp_path / "run")
    (tmp_path / "run/report.json").write_text("{}")
    _, report, _ = reopen(tmp_path / "run")
    assert not report["regressed"]["passed"]
    assert report["repairs"]["restore"]["passed"]
    render(tmp_path / "run", tmp_path / "report.html")
    page = (tmp_path / "report.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    with pytest.raises(FileExistsError):
        render(tmp_path / "run", tmp_path / "report.html")


def test_rejects_identity_swap(tmp_path):
    attempt(tmp_path / "run")
    path = tmp_path / "run/execution-001.json"
    record = json.loads(path.read_text())
    record["release"]["release_id"] = "restore"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        reopen(tmp_path / "run")
