from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from model_forensics.artifact_verify import read_map, references, verify_payloads
from model_forensics.incidents import Incident, Ledger, Truth, decision_metrics, digest, summarize


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


@pytest.mark.parametrize("location", ["incident", "run", "policy"])
def test_conflicting_artifact_identity_without_payload_verification(location):
    p = payload()
    conflict = {"identity": "good", "sha256": "b" * 64}
    if location == "incident":
        p["incident"]["environment"] = conflict
        with pytest.raises(ValidationError, match="conflicting hashes"):
            Incident.model_validate_json(json.dumps(p["incident"]))
    elif location == "run":
        p["runs"][0]["artifact"] = conflict
    else:
        p["decision"]["policy"] = conflict
    with pytest.raises(ValidationError, match="conflicting hashes"):
        load(p)


def test_reusing_one_consistent_artifact_identity_is_allowed():
    p = payload()
    p["decision"]["policy"] = copy.deepcopy(p["incident"]["environment"])
    ledger = load(p)
    assert summarize(ledger)["attempts"] == 2


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


def payload_files(tmp_path):
    p = payload()
    paths = {}

    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"identity", "sha256"}:
                name = item["identity"]
                data = name.encode()
                item["sha256"] = hashlib.sha256(data).hexdigest()
                paths[name] = name + ".txt"
                (tmp_path / paths[name]).write_bytes(data)
            else:
                for value in item.values():
                    visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(p)
    return load(p), paths


def test_payload_identity_and_cli(tmp_path):
    ledger, paths = payload_files(tmp_path)
    report = verify_payloads(ledger, tmp_path, paths)
    assert report["total_bytes"] == sum(len(x.encode()) for x in paths)
    assert str(tmp_path) not in json.dumps(report)
    source, mapping, out = (tmp_path / x for x in ("ledger.json", "map.json", "report.json"))
    source.write_text(ledger.model_dump_json())
    mapping.write_text(json.dumps(paths))
    command = [
        sys.executable,
        "-m",
        "model_forensics.incidents",
        str(source),
        "--artifact-map",
        str(mapping),
        "--artifact-root",
        str(tmp_path),
        "--out",
        str(out),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert json.loads(out.read_text())["payload_verification"] == report
    first = out.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert out.read_bytes() == first
    (tmp_path / paths["good"]).write_bytes(b"changed")
    command[-1] = str(tmp_path / "failed.json")
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert not (tmp_path / "failed.json").exists()


@pytest.mark.parametrize(
    "path",
    ["/absolute", "../escape", "a/../b", "a//b", "./a", "a/", "a\\b", "https://example.org/a", ""],
)
def test_payload_paths_rejected(tmp_path, path):
    ledger, paths = payload_files(tmp_path)
    paths["good"] = path
    with pytest.raises(ValueError, match="relative POSIX"):
        verify_payloads(ledger, tmp_path, paths)


def test_payload_missing_extra_conflict_and_truth(tmp_path):
    ledger, paths = payload_files(tmp_path)
    with pytest.raises(ValueError, match="exactly cover"):
        verify_payloads(ledger, tmp_path, {**paths, "unused": "unused.txt"})
    with pytest.raises(ValueError, match="exactly cover"):
        verify_payloads(ledger, tmp_path, {k: v for k, v in paths.items() if k != "good"})
    p = ledger.model_dump(mode="json")
    p["decision"]["policy"] = {"identity": "good", "sha256": "0" * 64}
    with pytest.raises(ValueError, match="conflicting"):
        references(load(p))
    evaluator = truth(ledger, [["a"], ["b"]])
    assert "evaluator-only" not in references(ledger)
    assert "evaluator-only" in references(ledger, evaluator)
    with pytest.raises(ValueError, match="exactly cover"):
        verify_payloads(ledger, tmp_path, paths, evaluator)
    truth_data = b"evaluator provenance"
    evaluator = evaluator.model_copy(
        update={
            "provenance": evaluator.provenance.model_copy(
                update={"sha256": hashlib.sha256(truth_data).hexdigest()}
            )
        }
    )
    (tmp_path / "truth-evidence.json").write_bytes(truth_data)
    with_truth = verify_payloads(
        ledger, tmp_path, {**paths, "evaluator-only": "truth-evidence.json"}, evaluator
    )
    assert len(with_truth["artifacts"]) == len(paths) + 1
    (tmp_path / paths["good"]).unlink()
    with pytest.raises(ValueError, match="inaccessible"):
        verify_payloads(ledger, tmp_path, paths)


def test_payload_symlink_and_fifo(tmp_path):
    ledger, paths = payload_files(tmp_path)
    good = tmp_path / paths["good"]
    good.unlink()
    good.symlink_to(tmp_path / paths["bad"])
    with pytest.raises(ValueError, match="symlinked"):
        verify_payloads(ledger, tmp_path, paths)
    good.unlink()
    os.mkfifo(good)
    with pytest.raises(ValueError, match="regular file"):
        verify_payloads(ledger, tmp_path, paths)
    link = tmp_path / "root-link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked"):
        verify_payloads(ledger, link, paths)
    folder = tmp_path / "directory-link"
    folder.symlink_to(tmp_path, target_is_directory=True)
    paths["good"] = "directory-link/bad.txt"
    with pytest.raises(ValueError, match="symlinked"):
        verify_payloads(ledger, tmp_path, paths)


def test_payload_budgets_and_map(tmp_path, monkeypatch):
    import model_forensics.artifact_verify as verifier

    ledger, paths = payload_files(tmp_path)
    monkeypatch.setattr(verifier, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        verify_payloads(ledger, tmp_path, paths)
    monkeypatch.setattr(verifier, "MAX_FILE_BYTES", 1000)
    monkeypatch.setattr(verifier, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        verify_payloads(ledger, tmp_path, paths)
    mapping = tmp_path / "map.json"
    mapping.write_text('{"a":"x", "a":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        read_map(mapping)
    mapping.write_text('["x"]')
    with pytest.raises(ValueError, match="strings"):
        read_map(mapping)
    monkeypatch.setattr(verifier, "MAX_MAP_BYTES", 1)
    with pytest.raises(ValueError, match="input bound"):
        read_map(mapping)
