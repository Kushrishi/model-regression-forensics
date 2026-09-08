from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from model_forensics.exp007 import (
    EXP007_FROZEN_MANIFEST_SHA256,
    EXP007_SHARD_IDS,
)


def _load_candidate_gate_module():
    path = Path("scripts/check_exp007_candidate_gate.py")
    spec = importlib.util.spec_from_file_location(
        "check_exp007_candidate_gate",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load candidate-gate script module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def _config_sha(repo: Path) -> str:
    return hashlib.sha256((repo / "configs/exp007.yaml").read_bytes()).hexdigest()


def _selection(repo: Path, dose: int = 9) -> dict[str, object]:
    return {
        "experiment_id": "exp007",
        "selection_rule": "minimum_target_dose_meeting_gate",
        "observed_target_doses": [9],
        "selected_target_dose": dose,
        "certification_authorized": True,
        "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
        "config_sha256": _config_sha(repo),
        "calibration_results": {},
    }


def _gate(*, passed: bool) -> dict[str, object]:
    return {
        "experiment_id": "exp007",
        "gate": "candidate_localized_regression",
        "phase": "certification",
        "target_dose": 9,
        "world_index": 0,
        "provenance": {
            "manifest_sha256": EXP007_FROZEN_MANIFEST_SHA256,
        },
        "result": {"all_passed": passed},
    }


def test_prepare_exp007_calibration_keeps_root_private(
    tmp_path: Path,
) -> None:
    repo = Path.cwd()
    output = tmp_path / "calibration"

    result = _run(
        repo,
        "scripts/prepare_exp007.py",
        "--phase",
        "calibration",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads((output / "summary.json").read_text())
    diagnostic = json.loads((output / "lineage/diagnostic.json").read_text())
    private = json.loads((output / "private/world.json").read_text())

    assert summary["construction_gates"]["all_passed"] is True
    assert private["planted_candidate_id"] in EXP007_SHARD_IDS
    assert private["planted_candidate_id"] not in json.dumps(summary)
    assert "hidden_root_cause_id" not in json.dumps(diagnostic)

    rendered = (output / "datasets/all_eval.jsonl").read_text()

    for material in ("bronze", "cotton", "quartz", "velvet"):
        assert f"material={material}" in rendered

    for material in ("bamboo", "ceramic", "marble", "wool"):
        assert f"material={material}" not in rendered


def test_prepare_exp007_certification_requires_selection(
    tmp_path: Path,
) -> None:
    result = _run(
        Path.cwd(),
        "scripts/prepare_exp007.py",
        "--phase",
        "certification",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--output",
        str(tmp_path / "prepared"),
    )

    assert result.returncode != 0
    assert "requires a frozen calibration selection" in result.stderr


def test_restoration_requires_passed_certification_gate(
    tmp_path: Path,
) -> None:
    repo = Path.cwd()

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(_selection(repo)) + "\n",
        encoding="utf-8",
    )

    prepared = tmp_path / "prepared"
    prep = _run(
        repo,
        "scripts/prepare_exp007.py",
        "--phase",
        "certification",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--selection",
        str(selection),
        "--output",
        str(prepared),
    )
    assert prep.returncode == 0, prep.stderr

    failed_gate = tmp_path / "failed_gate.json"
    failed_gate.write_text(
        json.dumps(_gate(passed=False)) + "\n",
        encoding="utf-8",
    )

    refused = _run(
        repo,
        "scripts/prepare_exp007_certification.py",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--selection",
        str(selection),
        "--candidate-gate",
        str(failed_gate),
        "--prepared",
        str(prepared),
        "--output",
        str(tmp_path / "refused"),
    )

    assert refused.returncode != 0
    assert "requires passed localized-regression gate" in refused.stderr

    passed_gate = tmp_path / "passed_gate.json"
    passed_gate.write_text(
        json.dumps(_gate(passed=True)) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "certification"
    allowed = _run(
        repo,
        "scripts/prepare_exp007_certification.py",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--selection",
        str(selection),
        "--candidate-gate",
        str(passed_gate),
        "--prepared",
        str(prepared),
        "--output",
        str(output),
    )

    assert allowed.returncode == 0, allowed.stderr

    summary = json.loads((output / "summary.json").read_text())

    assert summary["primary_restoration_count"] == 5
    assert summary["restored_records_per_candidate"] == 36
    assert summary["order_control_a"]["identical_example_order_across_siblings"] is True

    for candidate_id in EXP007_SHARD_IDS:
        assert (
            output / "primary" / "datasets" / f"restoration_{candidate_id}_train.jsonl"
        ).is_file()


def test_selection_rule_stops_after_dose9_qualifies(
    tmp_path: Path,
) -> None:
    repo = Path.cwd()

    dose9 = tmp_path / "dose9.json"
    dose18 = tmp_path / "dose18.json"

    dose9.write_text(
        json.dumps(
            {
                "experiment_id": "exp007",
                "gate": "calibration_target_dose",
                "target_dose": 9,
                "qualified": True,
                "passing_worlds": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dose18.write_text(
        json.dumps(
            {
                "experiment_id": "exp007",
                "gate": "calibration_target_dose",
                "target_dose": 18,
                "qualified": True,
                "passing_worlds": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    good_output = tmp_path / "selection.json"
    good = _run(
        repo,
        "scripts/freeze_exp007_selection.py",
        "--dose9-summary",
        str(dose9),
        "--output",
        str(good_output),
    )

    assert good.returncode == 0, good.stderr
    selected = json.loads(good_output.read_text())
    assert selected["selected_target_dose"] == 9
    assert selected["observed_target_doses"] == [9]

    bad = _run(
        repo,
        "scripts/freeze_exp007_selection.py",
        "--dose9-summary",
        str(dose9),
        "--dose18-summary",
        str(dose18),
        "--output",
        str(tmp_path / "invalid.json"),
    )

    assert bad.returncode != 0
    assert "must remain uninspected" in bad.stderr


def test_exp007_candidate_gate_rejects_wrong_world_provenance(
    tmp_path: Path,
) -> None:
    _validate_provenance = _load_candidate_gate_module()._validate_provenance

    repo = Path.cwd()
    prepared = tmp_path / "prepared"

    prep = _run(
        repo,
        "scripts/prepare_exp007.py",
        "--phase",
        "calibration",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--output",
        str(prepared),
    )
    assert prep.returncode == 0, prep.stderr

    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    baseline_adapter = tmp_path / "baseline_adapter"
    candidate_adapter = tmp_path / "candidate_adapter"

    baseline_train = {
        "experiment_id": "exp007",
        "run_kind": "lora_sft",
        "train_split": "baseline_train",
        "prepared_input": {"file_sha256": file_sha(prepared / "datasets/baseline_train.jsonl")},
        "adapter_path": str(baseline_adapter),
    }
    candidate_train = {
        "experiment_id": "exp007",
        "run_kind": "lora_sft",
        "train_split": "candidate_train",
        "prepared_input": {"file_sha256": file_sha(prepared / "datasets/candidate_train.jsonl")},
        "adapter_path": str(candidate_adapter),
    }

    eval_hashes = {
        f"{name}_file_sha256": file_sha(prepared / "datasets" / f"{name}.jsonl")
        for name in (
            "circle_small_eval",
            "circle_large_eval",
            "square_small_eval",
            "square_large_eval",
            "triangle_small_eval",
            "triangle_large_eval",
            "all_eval",
        )
    }

    baseline_eval = {
        "experiment_id": "exp007",
        "run_kind": "adapter_eval",
        "adapter": str(baseline_adapter),
        "prepared_inputs": eval_hashes,
    }
    candidate_eval = {
        "experiment_id": "exp007",
        "run_kind": "adapter_eval",
        "adapter": str(candidate_adapter),
        "prepared_inputs": eval_hashes,
    }

    provenance = _validate_provenance(
        prepared=prepared,
        phase="calibration",
        target_dose=9,
        world_index=0,
        baseline_train_summary=baseline_train,
        candidate_train_summary=candidate_train,
        baseline_eval_summary=baseline_eval,
        candidate_eval_summary=candidate_eval,
    )
    assert provenance["manifest_sha256"] == EXP007_FROZEN_MANIFEST_SHA256

    wrong_candidate = dict(candidate_train)
    wrong_candidate["prepared_input"] = {
        "file_sha256": "0" * 64,
    }

    try:
        _validate_provenance(
            prepared=prepared,
            phase="calibration",
            target_dose=9,
            world_index=0,
            baseline_train_summary=baseline_train,
            candidate_train_summary=wrong_candidate,
            baseline_eval_summary=baseline_eval,
            candidate_eval_summary=candidate_eval,
        )
    except ValueError as exc:
        assert "not trained on this prepared world" in str(exc)
    else:
        raise AssertionError("wrong training provenance was accepted")

    wrong_eval = dict(candidate_eval)
    wrong_eval["prepared_inputs"] = {
        **eval_hashes,
        "all_eval_file_sha256": "f" * 64,
    }

    try:
        _validate_provenance(
            prepared=prepared,
            phase="calibration",
            target_dose=9,
            world_index=0,
            baseline_train_summary=baseline_train,
            candidate_train_summary=candidate_train,
            baseline_eval_summary=baseline_eval,
            candidate_eval_summary=wrong_eval,
        )
    except ValueError as exc:
        assert "evaluation inputs do not match" in str(exc)
    else:
        raise AssertionError("wrong evaluation provenance was accepted")


def test_exp007_candidate_gate_rejects_wrong_named_world(
    tmp_path: Path,
) -> None:
    _validate_prepared_world = _load_candidate_gate_module()._validate_prepared_world

    repo = Path.cwd()
    prepared = tmp_path / "prepared"

    prep = _run(
        repo,
        "scripts/prepare_exp007.py",
        "--phase",
        "calibration",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--output",
        str(prepared),
    )
    assert prep.returncode == 0, prep.stderr

    try:
        _validate_prepared_world(
            prepared,
            phase="calibration",
            target_dose=9,
            world_index=1,
        )
    except ValueError as exc:
        assert "world index mismatch" in str(exc)
    else:
        raise AssertionError("wrong named world was accepted")


def test_restoration_rejects_gate_without_provenance(
    tmp_path: Path,
) -> None:
    repo = Path.cwd()

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(_selection(repo)) + "\n",
        encoding="utf-8",
    )

    prepared = tmp_path / "prepared"
    prep = _run(
        repo,
        "scripts/prepare_exp007.py",
        "--phase",
        "certification",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--selection",
        str(selection),
        "--output",
        str(prepared),
    )
    assert prep.returncode == 0, prep.stderr

    gate_payload = _gate(passed=True)
    gate_payload.pop("provenance")

    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(gate_payload) + "\n",
        encoding="utf-8",
    )

    result = _run(
        repo,
        "scripts/prepare_exp007_certification.py",
        "--target-dose",
        "9",
        "--world-index",
        "0",
        "--selection",
        str(selection),
        "--candidate-gate",
        str(gate),
        "--prepared",
        str(prepared),
        "--output",
        str(tmp_path / "restoration"),
    )

    assert result.returncode != 0
    assert "candidate gate lacks world provenance" in result.stderr
