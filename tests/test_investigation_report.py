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
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "<th>Maximum drop</th>" in page
    assert "<td>0.00%</td>" in page
    assert "<th>Minimum accuracy</th>" not in page
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


def test_previews_are_escaped_and_case_outcomes_are_explicit(tmp_path):
    attempt(tmp_path / "run")
    key = "<script>alert(1)</script>"
    (tmp_path / "run/case_inputs.json").write_text(json.dumps({key: {"text": "<img onerror=bad>"}}))
    render(tmp_path / "run", tmp_path / "report.html")
    page = (tmp_path / "report.html").read_text()
    assert "&lt;img onerror=bad&gt;" in page
    assert "<img onerror=bad>" not in page
    assert 'data-state="regressed"' in page
    assert 'id="case-search"' in page


def test_unknown_preview_case_and_invalid_pixels_rejected(tmp_path):
    from model_forensics.case_previews import previews

    path = tmp_path / "case_inputs.json"
    path.write_text(json.dumps({"unknown": {"text": "x"}}))
    with pytest.raises(ValueError, match="unknown"):
        previews(tmp_path, {"a"})
    for pixels in ([256], [float("nan")], [], [True]):
        path.write_text(json.dumps({"a": {"width": 1, "height": 1, "pixels": pixels}}))
        with pytest.raises(ValueError, match="grayscale"):
            previews(tmp_path, {"a"})
