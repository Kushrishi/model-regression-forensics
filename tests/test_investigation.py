import json

import pytest

from model_forensics.investigation import run_investigation
from model_forensics.release_compare import Case, Prediction, Slice


def spec():
    cases = (Case(case_id="a", expected="yes"), Case(case_id="b", expected="no"))
    good = tuple(Prediction(case_id=c.case_id, observed=c.expected) for c in cases)
    bad = tuple(Prediction(case_id=c.case_id, observed="no") for c in cases)
    return dict(
        cases=cases,
        slices=(Slice(name="all", case_ids=("a", "b")),),
        baseline=lambda: good,
        candidate=lambda: bad,
        repairs={"restore": lambda: good, "alternative": lambda: good},
        changes={"candidate": "declared change", "repairs": ["restore", "alternative"]},
    )


def test_executes_each_function_and_retains_ambiguity(tmp_path):
    calls = []
    args = spec()
    for name in ("baseline", "candidate"):
        original = args[name]
        args[name] = lambda fn=original, n=name: (calls.append(n), fn())[1]
    result = run_investigation(tmp_path / "run", **args)
    assert calls == ["baseline", "candidate"]
    assert result["assessment"]["status"] == "ambiguous_repairs"
    assert result["historical_cause"] == "not_identified"
    assert len(result["executions"]) == 4
    assert json.loads((tmp_path / "run/report.json").read_text()) == result


def test_failure_retains_completed_baseline_without_retry(tmp_path):
    args = spec()
    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("backend failure")

    args["candidate"] = fail
    with pytest.raises(RuntimeError):
        run_investigation(tmp_path / "run", **args)
    assert calls == [1]
    assert (tmp_path / "run/execution-000.json").is_file()
    assert not (tmp_path / "run/report.json").exists()
    assert json.loads((tmp_path / "run/failed.json").read_text())["release"] == "candidate"
    with pytest.raises(FileExistsError):
        run_investigation(tmp_path / "run", **args)
    assert calls == [1]


def test_wrong_case_predictions_fail_before_assessment(tmp_path):
    args = spec()
    args["candidate"] = lambda: (Prediction(case_id="unknown", observed="yes"),)
    with pytest.raises(ValueError, match="exactly"):
        run_investigation(tmp_path / "run", **args)
    assert not (tmp_path / "run/report.json").exists()


def test_invalid_plan_does_not_invoke_functions(tmp_path):
    args = spec()
    args["changes"] = {"invalid": float("nan")}
    with pytest.raises(ValueError):
        run_investigation(tmp_path / "run", **args)
    assert not (tmp_path / "run").exists()
