import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("einops")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/replay_lfq_defect.py"
spec = importlib.util.spec_from_file_location("lfq_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.fixture
def rows():
    return json.loads((ROOT / "research/LFQ_FIXED_TENSOR_REPLAY_2026_10_09.json").read_text())[
        "rows"
    ]


def test_retained_twelve_checks_pass(rows):
    replay.validate(rows)


@pytest.mark.parametrize("defect", ["empty", "missing", "duplicate", "unexpected"])
def test_requires_complete_declared_check_set(rows, defect):
    if defect == "empty":
        rows = []
    elif defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[0]["version"] = "unknown"
    with pytest.raises(ValueError, match="twelve"):
        replay.validate(rows)


@pytest.mark.parametrize(
    "field",
    [
        "loss_matches_oracle",
        "gradient_matches_oracle",
        "quantized_matches_oracle",
        "gradient_finite",
        "status",
    ],
)
def test_wrong_followup_outcome_is_rejected(rows, field):
    row = next(row for row in rows if row["version"] == "followup_repair")
    row[field] = "exception" if field == "status" else False
    with pytest.raises(ValueError, match="Unexpected replay outcome"):
        replay.validate(rows)


def test_optimized_python_cannot_disable_validation():
    code = f"import runpy; module = runpy.run_path({str(SCRIPT)!r}); module['validate']([])"
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0
    assert "requires exactly the twelve" in result.stderr


def test_wrong_source_digest_rejected_before_import(tmp_path):
    commit, _ = replay.SOURCES["reported"]
    marker = tmp_path / "executed"
    (tmp_path / f"{commit}.py").write_text(f"open({str(marker)!r}, 'w').write('bad')")
    with pytest.raises(ValueError, match="Source digest mismatch"):
        replay.load_source(tmp_path, "reported")
    assert not marker.exists()


def test_report_collision_preserves_original_bytes(tmp_path):
    path = tmp_path / "result.json"
    replay.write_report(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        replay.write_report(path, {"replacement": True})
    assert path.read_bytes() == before


def test_existing_output_rejected_before_source_read(tmp_path, monkeypatch):
    path = tmp_path / "accepted.json"
    path.write_text("retained evidence")
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--source-dir", str(tmp_path / "absent"), "--output", str(path)]
    )
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert path.read_text() == "retained evidence"
