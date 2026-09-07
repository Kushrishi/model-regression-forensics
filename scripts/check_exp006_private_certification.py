from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_forensics.certification import (
    evaluate_exp006_candidate_gate,
    evaluate_exp006_causal_certification,
    evaluate_exp006_order_control,
    exp006_public_certification_payload,
)
from model_forensics.config import load_experiment_config
from model_forensics.task import EXP006_SHARD_IDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Privately evaluate Experiment 006 causal certification "
            "without revealing root identity."
        )
    )
    parser.add_argument("--config", default="configs/exp006.yaml")
    parser.add_argument("--construction-summary", required=True)
    parser.add_argument("--world-json", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--primary-runs-root", required=True)
    parser.add_argument("--primary-run-prefix", required=True)
    parser.add_argument("--order-runs-root", required=True)
    parser.add_argument("--order-run-prefix", required=True)
    parser.add_argument("--private-output", required=True)
    parser.add_argument("--public-output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _run_summary(root: Path, run_id: str) -> dict[str, Any]:
    return _load(root / run_id / "summary.json")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if config.experiment_id != "exp006":
        raise ValueError("private certification requires Experiment 006 config")

    construction = _load(args.construction_summary)
    world = _load(args.world_json)
    baseline = _load(args.baseline_summary)
    candidate = _load(args.candidate_summary)

    construction_gate_passed = bool(
        construction.get("construction_gates", {}).get("all_passed", False)
    )
    if not construction_gate_passed:
        raise ValueError("Experiment 006 private certification requires passed construction gates")

    candidate_gate = evaluate_exp006_candidate_gate(
        baseline_summary=baseline,
        candidate_summary=candidate,
        minimum_baseline_score=config.evaluation.minimum_baseline_score,
        minimum_regression_delta=config.evaluation.minimum_regression_delta,
        maximum_unrelated_delta=config.evaluation.maximum_unrelated_delta,
    )
    if not candidate_gate.all_passed:
        raise ValueError(
            "Experiment 006 private certification requires passed localized-regression gate"
        )

    planted_candidate_id = world.get("planted_candidate_id")
    if planted_candidate_id not in EXP006_SHARD_IDS:
        raise ValueError("private world file lacks a valid planted candidate ID")

    primary_root = Path(args.primary_runs_root)
    restoration_summaries = {
        candidate_id: _run_summary(
            primary_root,
            f"{args.primary_run_prefix}_restore_{candidate_id}",
        )
        for candidate_id in EXP006_SHARD_IDS
    }

    causal = evaluate_exp006_causal_certification(
        baseline_summary=baseline,
        candidate_summary=candidate,
        restoration_summaries=restoration_summaries,
        planted_candidate_id=str(planted_candidate_id),
        candidate_ids=EXP006_SHARD_IDS,
        minimum_recovery_delta=config.evaluation.minimum_recovery_delta,
        maximum_unrelated_delta=config.evaluation.maximum_unrelated_delta,
    )

    order_root = Path(args.order_runs_root)
    order_baseline = _run_summary(
        order_root,
        f"{args.order_run_prefix}_baseline",
    )
    order_candidate = _run_summary(
        order_root,
        f"{args.order_run_prefix}_candidate",
    )
    order_restoration = _run_summary(
        order_root,
        f"{args.order_run_prefix}_restore_{planted_candidate_id}",
    )
    order = evaluate_exp006_order_control(
        baseline_summary=order_baseline,
        candidate_summary=order_candidate,
        planted_restoration_summary=order_restoration,
        minimum_baseline_score=config.evaluation.minimum_baseline_score,
        minimum_regression_delta=config.evaluation.minimum_regression_delta,
        minimum_recovery_delta=config.evaluation.minimum_recovery_delta,
        maximum_unrelated_delta=config.evaluation.maximum_unrelated_delta,
    )

    public_payload = exp006_public_certification_payload(
        construction_gate_passed=construction_gate_passed,
        candidate_gate=candidate_gate,
        causal_certification=causal,
        order_control=order,
    )
    private_payload = {
        "experiment_id": "exp006",
        "candidate_gate": candidate_gate.to_dict(),
        "causal_certification": causal.private_dict(),
        "order_control": order.private_dict(),
        "public_certification": public_payload,
    }

    private_output = Path(args.private_output)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_output.write_text(
        json.dumps(private_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    public_output = Path(args.public_output)
    public_output.parent.mkdir(parents=True, exist_ok=True)
    public_output.write_text(
        json.dumps(public_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Deliberately print only the protocol-approved public boundary.
    print(json.dumps(public_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
