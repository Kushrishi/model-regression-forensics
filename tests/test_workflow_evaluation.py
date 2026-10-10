import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_forensics.workflow_evaluation import audit, prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "fixture", ["release_acceptance", "initial_comparison", "deployment_comparison"]
)
def test_equal_evidence_runnable_baseline_and_unanswered_audit(fixture, tmp_path):
    source = ROOT / "examples" / fixture
    output = tmp_path / "packet"
    prepare(source / "policy.json", source / "predictions.csv", output)
    gold = json.loads((output / "facilitator-gold.json").read_text())
    for name in (
        "source-policy.json",
        "source-predictions.csv",
        "source-hashes.json",
        "answer-template.json",
    ):
        assert (output / "ordinary" / name).read_bytes() == (output / "mrf" / name).read_bytes()
    for condition in ("ordinary", "mrf"):
        packet = output / condition
        assert not (packet / "facilitator-gold.json").exists()
        subprocess.run(
            [sys.executable, str(packet / "verify_sources.py")], check=True, capture_output=True
        )
        (packet / "source-predictions.csv").write_text("modified evidence")
        assert (
            subprocess.run(
                [sys.executable, str(packet / "verify_sources.py")], capture_output=True
            ).returncode
            == 1
        )
    # Baseline remains runnable without importing the MRF package.
    (output / "ordinary/source-predictions.csv").write_bytes(
        (source / "predictions.csv").read_bytes()
    )
    run = subprocess.run(
        [sys.executable, "-I", str(output / "ordinary/ordinary_diff.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(run.stdout)["candidate"]["passed"] == gold["candidate_passes"]
    template = json.loads((output / "ordinary/answer-template.json").read_text())
    assert set(audit(gold, template)["objective_answers"].values()) == {"unanswered"}
    assert audit(gold, template)["comparative_user_benefit"] == "not_established"
    with pytest.raises(FileExistsError):
        prepare(source / "policy.json", source / "predictions.csv", output)


def test_wrong_decisions_and_overclaims_are_scored_without_inventing_effort(tmp_path):
    source = ROOT / "examples/release_acceptance"
    output = tmp_path / "packet"
    prepare(source / "policy.json", source / "predictions.csv", output)
    gold = json.loads((output / "facilitator-gold.json").read_text())
    submission = json.loads((output / "mrf/answer-template.json").read_text())
    submission["answers"] = copy.deepcopy(gold)
    report = audit(gold, submission)
    assert set(report["objective_answers"].values()) == {"correct"}
    assert report["submitted_effort"]["active_seconds"] is None
    for key in (
        "complete_rollback_passes",
        "historical_cause_identified",
        "retraining_evaluated",
        "execution_cost_known",
    ):
        submission["answers"][key] = True
    report = audit(gold, submission)
    assert report["objective_answers"]["complete_rollback_passes"] == "incorrect"
    assert report["objective_answers"]["historical_cause_identified"] == "incorrect"
    submission["answers"]["candidate_passes"] = 0
    with pytest.raises(ValueError, match="type"):
        audit(gold, submission)


def test_multiple_repairs_and_missing_evidence_remain_distinct(tmp_path):
    source = ROOT / "examples/release_acceptance"
    predictions = tmp_path / "predictions.csv"
    data = (source / "predictions.csv").read_text()
    data += "".join(
        line.replace("targeted_repair", "second_distinct_repair") + "\n"
        for line in data.splitlines()
        if line.startswith("targeted_repair,")
    )
    predictions.write_text(data)
    output = tmp_path / "packet"
    prepare(source / "policy.json", predictions, output)
    gold = json.loads((output / "facilitator-gold.json").read_text())
    assert gold["distinct_passing_repairs"] == 2
    assert not gold["historical_cause_identified"]
    assert gold["repair_evidence_supplied"]
    source = ROOT / "examples/initial_comparison"
    prepare(source / "policy.json", source / "predictions.csv", tmp_path / "missing")
    missing = json.loads((tmp_path / "missing/facilitator-gold.json").read_text())
    assert not missing["repair_evidence_supplied"]
    assert not missing["passing_supplied_repairs"]
