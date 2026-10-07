from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from model_forensics.incidents import Ledger, Truth, decision_metrics, digest, summarize


def artifact(name):
    return {"identity": name, "sha256": "a" * 64}


def payload():
    return {
        "schema_version": "incident-ledger/0.1",
        "incident": {
            "schema_version": "release-incident/0.1",
            "incident_id": "fixture",
            "use": "synthetic_software",
            "good_release": artifact("good"),
            "bad_release": artifact("bad"),
            "environment": artifact("environment"),
            "candidates": [
                {
                    "candidate_id": x,
                    "category": "preprocessing",
                    "visible_diff": artifact(x),
                    "description": "complete fixture change",
                }
                for x in ("a", "b")
            ],
            "candidate_completeness": "declared_complete",
            "target_functional": "exact label accuracy",
            "evaluation": artifact("evaluation"),
            "protected_behavior": "none: deterministic arithmetic fixture only",
            "intervention_semantics": "revert each named change from the bad release",
            "training_process": "no training: retained deterministic fixture",
            "randomness_treatment": "deterministic fixture; one execution identity",
            "restoration_meaning": "baseline predictions exactly equal",
            "engineer_visible_evidence": [artifact("complete-diff")],
            "max_intervention_attempts": 3,
        },
        "runs": [
            {
                "run_id": x,
                "change_set": [x],
                "randomness_id": "seed-1",
                "status": "completed",
                "target_value": 1.0,
                "protected_value": None,
                "artifact": artifact(x + "-run"),
                "wall_seconds": 0.01,
                "cpu_seconds": 0.01,
                "failure": None,
            }
            for x in ("a", "b")
        ],
        "decision": {
            "status": "ambiguity_set",
            "hypotheses": [["a"], ["b"]],
            "evidence_run_ids": ["a", "b"],
            "policy": artifact("fixture-policy"),
            "rationale": "Both exact repairs restore retained fixture predictions.",
        },
    }


def load(p):
    return Ledger.model_validate_json(json.dumps(p))


def truth(ledger, sets):
    return Truth.model_validate_json(
        json.dumps(
            {
                "incident_sha256": digest(ledger.incident),
                "responsibility_sets": sets,
                "provenance": artifact("evaluator-only"),
            }
        )
    )


def test_ambiguity_and_no_historical_inference():
    ledger = load(payload())
    report = summarize(ledger, truth(ledger, [["a"], ["b"]]))
    assert report["historical_cause"] == "not_inferred"
    assert report["evaluation"]["all_responsible_sets_contained"]
    assert report["evaluation"]["returned_candidate_count"] == 2
    m = decision_metrics([(ledger, truth(ledger, [["a"], ["b"]]))])
    assert m["commitment_coverage"] == 0
    assert m["risk_among_committed"] is None
    assert m["intervention_attempts"] == 2


def test_one_restorative_alternative_is_still_wrong_specificity():
    p = payload()
    p["decision"].update(status="specific", hypotheses=[["a"]])
    ledger = load(p)
    report = summarize(ledger, truth(ledger, [["a"], ["b"]]))
    assert decision_metrics([(ledger, truth(ledger, [["a"], ["b"]]))])["risk_among_committed"] == 1
    assert report["evaluation"]["any_responsible_set_contained"]
    assert not report["evaluation"]["all_responsible_sets_contained"]


def test_interaction_and_omitted_candidate_truth():
    p = payload()
    p["decision"].update(status="specific", hypotheses=[["a", "b"]])
    ledger = load(p)
    assert not summarize(ledger, truth(ledger, [["b", "a"]]))["evaluation"]["wrong_specific"]
    assert summarize(ledger, truth(ledger, [["omitted"]]))["evaluation"]["wrong_specific"]


def test_repeats_and_failures_count_without_pseudoreplication():
    p = payload()
    run = copy.deepcopy(p["runs"][0])
    run.update(
        run_id="repeat",
        randomness_id="seed-2",
        status="timeout",
        target_value=None,
        failure="bounded timeout",
        wall_seconds=1.0,
        cpu_seconds=1.5,
    )
    p["runs"].append(run)
    ledger = load(p)
    report = summarize(ledger, truth(ledger, [["a"], ["b"]]))
    assert report["attempts"] == 3 and report["completed"] == 2
    assert report["cpu_seconds"] == 1.52
    assert decision_metrics([(ledger, truth(ledger, [["a"], ["b"]]))])["incident_count"] == 1
    with pytest.raises(ValueError, match="unique incident"):
        decision_metrics([(ledger, truth(ledger, [["a"]])), (ledger, truth(ledger, [["a"]]))])


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "duplicate",
        "budget",
        "nonfinite",
        "failure",
        "evidence",
        "truth_leak",
        "bad_hash",
        "duplicate_hypothesis",
    ],
)
def test_fail_closed(fault):
    p = payload()
    if fault == "unknown":
        p["runs"][0]["change_set"] = ["unknown"]
    if fault == "duplicate":
        p["runs"].append(copy.deepcopy(p["runs"][0]))
    if fault == "budget":
        p["incident"]["max_intervention_attempts"] = 1
    if fault == "nonfinite":
        p["runs"][0]["cpu_seconds"] = float("inf")
    if fault == "failure":
        p["runs"][0]["status"] = "failed"
    if fault == "evidence":
        p["decision"]["evidence_run_ids"] = ["unknown"]
    if fault == "truth_leak":
        p["incident"]["root"] = "a"
    if fault == "bad_hash":
        p["incident"]["evaluation"]["sha256"] = "not-a-digest"
    if fault == "duplicate_hypothesis":
        p["decision"]["hypotheses"] = [["a", "b"], ["b", "a"]]
    with pytest.raises(ValidationError):
        load(p)


def test_truth_identity_and_unscored_ledger():
    ledger = load(payload())
    assert summarize(ledger)["evaluation"] is None
    record = truth(ledger, [["a"]]).model_dump(mode="json")
    record["incident_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different incident"):
        summarize(ledger, Truth.model_validate_json(json.dumps(record)))


def test_cli_no_clobber_and_replay(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload()))
    out = tmp_path / "report.json"
    cmd = [sys.executable, "-m", "model_forensics.incidents", str(source), "--out", str(out)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    first = out.read_bytes()
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    assert out.read_bytes() == first
    second = tmp_path / "second.json"
    cmd[-1] = str(second)
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert second.read_bytes() == first
