import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/verify_deployment_fixture.py"
spec = importlib.util.spec_from_file_location("deployment_lineage", SCRIPT)
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)


def test_prediction_rows_reject_duplicate_and_malformed_records():
    raw = b"release_id,case_id,observed\nbaseline,a,yes\n"
    assert lineage.prediction_rows(raw) == {("baseline", "a"): "yes"}
    with pytest.raises(ValueError, match="duplicate"):
        lineage.prediction_rows(raw + b"baseline,a,no\n")
    with pytest.raises(ValueError, match="invalid"):
        lineage.prediction_rows(raw + b"candidate,a,yes,extra\n")
    assert lineage.prediction_rows(raw.replace(b"\n", b"\r\n")) == lineage.prediction_rows(raw)


def test_changed_archive_rejected_before_loading_scores(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"not the historical archive")
    with pytest.raises(ValueError, match="independent pin"):
        lineage.verify(archive, tmp_path)


def test_saved_score_lineage_and_corruption(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        case_ids=np.array(["a", "b"]),
        classes=np.array(["no", "yes"]),
        source=np.array([[0.0, 1.0], [1.0, 0.0]]),
        float32=np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32"),
        int8=np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32"),
    )
    rows = (
        b"release_id,case_id,observed\nbaseline,a,yes\nbaseline,b,no\n"
        b"candidate,a,no\ncandidate,b,no\n"
    )
    policy = json.dumps({"cases": ["a", "b"]}).encode()
    inputs = json.dumps({"a": "text A", "b": "text B"}).encode()
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "x") as z:
        for name, raw in [
            ("scores.npz", buffer.getvalue()),
            ("float32-to-int8-predictions.csv", rows.replace(b"\n", b"\r\n")),
            ("float32-to-int8-policy.json", policy),
            ("float32-to-int8/case_inputs.json", inputs),
        ]:
            z.writestr(lineage.PREFIX + name, raw)
    monkeypatch.setattr(lineage, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name, raw in [
        ("predictions.csv", rows),
        ("policy.json", policy),
        ("case_inputs.json", inputs),
    ]:
        (fixture / name).write_bytes(raw)
    report = lineage.verify(archive, fixture)
    assert report["cases"] == 2 and report["prediction_rows"] == 4
    assert report["user_benefit"] == "not_evaluated"
    (fixture / "predictions.csv").write_bytes(rows.replace(b"candidate,a,no", b"candidate,a,yes"))
    with pytest.raises(ValueError, match="raw-score argmax"):
        lineage.verify(archive, fixture)
    (fixture / "predictions.csv").write_bytes(rows)
    (fixture / "policy.json").write_text('{"cases":["a"]}')
    with pytest.raises(ValueError, match="policy.json"):
        lineage.verify(archive, fixture)
    (fixture / "policy.json").write_bytes(policy)
    (fixture / "case_inputs.json").write_text('{"a":"different text"}')
    with pytest.raises(ValueError, match="case_inputs.json"):
        lineage.verify(archive, fixture)


def test_nonfinite_scores_and_duplicate_identities_rejected():
    np = pytest.importorskip("numpy")
    for ids, matrix, error in [
        (["a", "a"], [[0.0, 1.0], [1.0, 0.0]], "duplicate"),
        (["a", "b"], [[0.0, float("nan")], [1.0, 0.0]], "invalid"),
    ]:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            case_ids=np.array(ids),
            classes=np.array(["no", "yes"]),
            source=np.array(matrix),
            float32=np.array(matrix),
            int8=np.array(matrix),
        )
        with pytest.raises(ValueError, match=error):
            lineage.score_predictions(buffer.getvalue())
