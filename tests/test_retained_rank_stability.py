import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "retained_rank_analysis", ROOT / "scripts/analyze_retained_rank_stability.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_retained_analysis_reproduces_accepted_output():
    expected = json.loads((ROOT / "research/retained-rank-stability.json").read_text())
    assert analysis.analyze() == expected
    assert expected["benchmark_world_count"] == 2
    assert expected["new_training_or_scoring"] is False


def test_modified_saved_scores_are_rejected(tmp_path, monkeypatch):
    (tmp_path / "research").mkdir()
    shutil.copy(ROOT / "research/M4_RESULT.json", tmp_path / "research/M4_RESULT.json")
    shutil.copytree(
        ROOT / "research/M4_RESULT_ARTIFACTS", tmp_path / "research/M4_RESULT_ARTIFACTS"
    )
    path = tmp_path / "research/M4_RESULT_ARTIFACTS/world_00_t0.json"
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        analysis.analyze()
