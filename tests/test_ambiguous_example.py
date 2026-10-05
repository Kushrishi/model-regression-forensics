"""Keep the documented repair example runnable through the public JSON contract."""

import json
import subprocess
import sys
from pathlib import Path


def test_documented_ambiguous_repair_example_runs():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/ambiguous_repairs.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    record = json.loads(result.stdout)
    assert record["regression_report"]["passed"] is False
    assert record["repair_assessment"]["status"] == "ambiguous_repairs"
    assert record["repair_assessment"]["historical_cause"] == "not_identified"
    assert record["fixture_truth"]["truth_used_by_assessment"] is False
