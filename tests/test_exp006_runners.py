from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_forensics.task import EXP006_SHARD_IDS


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def _scores(target: float = 1.0) -> dict[str, float]:
    scores = {
        "circle_small": 1.0,
        "circle_large": 1.0,
        "square_small": 1.0,
        "square_large": 1.0,
        "triangle_small": 1.0,
        "triangle_large": target,
        "all": 1.0,
    }
    if target != 1.0:
        scores["all"] = (80 + 16 * target) / 96
    return scores


def _summary(scores: dict[str, float]) -> dict[str, object]:
    return {
        "experiment_id": "exp006",
        "scores": {
            "label_accuracy": {
                split: {"score": score, "total": 16, "failed_case_ids": []}
                for split, score in scores.items()
            }
        },
    }


def test_prepare_exp006_materializes_frozen_world_without_public_root_leak(
    tmp_path: Path,
) -> None:
    repo = Path.cwd()
    output = tmp_path / "prepared"

    result = _run(
        repo,
        "scripts/prepare_exp006.py",
        "--attempt-index",
        "0",
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((output / "summary.json").read_text())
    diagnostic = json.loads((output / "lineage/diagnostic.json").read_text())
    private_world = json.loads((output / "private/world.json").read_text())

    assert summary["construction_gates"]["all_passed"] is True
    assert summary["world_identity_redacted"] is True
    assert summary["counts"]["records_per_change"] == 48
    assert summary["counts"]["label_changes_per_change"] == 12
    assert summary["counts"]["accept_to_reject"] == 38
    assert summary["counts"]["reject_to_accept"] == 22
    assert summary["counts"]["candidate_label_counts"] == {
        "ACCEPT": 176,
        "REJECT": 112,
    }

    rendered_summary = json.dumps(summary, sort_keys=True)
    rendered_diagnostic = json.dumps(diagnostic, sort_keys=True)
    assert private_world["planted_candidate_id"] == "shard_semantic_01"
    assert private_world["planted_candidate_id"] not in rendered_summary
    assert "hidden_root_cause_id" not in rendered_diagnostic


def test_prepare_exp006_certification_freezes_five_restorations_and_order(
    tmp_path: Path,
) -> None:
    repo = Path.cwd()
    prepared = tmp_path / "prepared"
    certification = tmp_path / "certification"

    first = _run(
        repo,
        "scripts/prepare_exp006.py",
        "--attempt-index",
        "0",
        "--output",
        str(prepared),
    )
    assert first.returncode == 0, first.stderr

    result = _run(
        repo,
        "scripts/prepare_exp006_certification.py",
        "--attempt-index",
        "0",
        "--prepared",
        str(prepared),
        "--output",
        str(certification),
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads((certification / "summary.json").read_text())
    assert summary["experiment_id"] == "exp006"
    assert summary["primary_restoration_count"] == 5
    assert summary["restored_records_per_candidate"] == 12
    assert summary["order_control_a"]["namespace"] == "exp006-order-control-a"
    assert summary["order_control_a"]["identical_example_order_across_siblings"] is True

    for candidate_id in EXP006_SHARD_IDS:
        assert (
            certification / "primary/datasets" / f"restoration_{candidate_id}_train.jsonl"
        ).is_file()
        assert (
            certification / "order_control_a/datasets" / f"restoration_{candidate_id}_train.jsonl"
        ).is_file()


def test_candidate_gate_runner_reports_localized_pass(tmp_path: Path) -> None:
    repo = Path.cwd()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "gate.json"

    baseline.write_text(json.dumps(_summary(_scores())) + "\n")
    candidate.write_text(json.dumps(_summary(_scores(0.75))) + "\n")

    result = _run(
        repo,
        "scripts/check_exp006_candidate_gate.py",
        "--baseline-summary",
        str(baseline),
        "--candidate-summary",
        str(candidate),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(output.read_text())
    assert payload["experiment_id"] == "exp006"
    assert payload["result"]["all_passed"] is True


def test_private_certification_refuses_failed_localized_gate(tmp_path: Path) -> None:
    repo = Path.cwd()
    construction = tmp_path / "construction.json"
    world = tmp_path / "world.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"

    construction.write_text(json.dumps({"construction_gates": {"all_passed": True}}) + "\n")
    world.write_text(json.dumps({"planted_candidate_id": "shard_semantic_01"}) + "\n")
    baseline.write_text(json.dumps(_summary(_scores())) + "\n")
    candidate.write_text(json.dumps(_summary(_scores(1.0))) + "\n")

    result = _run(
        repo,
        "scripts/check_exp006_private_certification.py",
        "--construction-summary",
        str(construction),
        "--world-json",
        str(world),
        "--baseline-summary",
        str(baseline),
        "--candidate-summary",
        str(candidate),
        "--primary-runs-root",
        str(tmp_path / "primary"),
        "--primary-run-prefix",
        "attempt00",
        "--order-runs-root",
        str(tmp_path / "order"),
        "--order-run-prefix",
        "attempt00_order",
        "--private-output",
        str(tmp_path / "private.json"),
        "--public-output",
        str(tmp_path / "public.json"),
    )

    assert result.returncode != 0
    assert "requires passed localized-regression gate" in result.stderr


@pytest.mark.parametrize(
    "script",
    (
        "scripts/train_exp006_sft.py",
        "scripts/eval_exp006_adapter.py",
        "scripts/check_exp006_candidate_gate.py",
        "scripts/check_exp006_private_certification.py",
    ),
)
def test_exp006_execution_scripts_expose_help(script: str) -> None:
    repo = Path.cwd()
    result = _run(repo, script, "--help")

    assert result.returncode == 0
    assert "Experiment 006" in result.stdout
