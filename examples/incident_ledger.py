"""Retrospective software representation of the retained ambiguous-repair fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ambiguous_repairs import WEIGHTS, comparison, predictions, reverse

from model_forensics.incidents import Ledger, Truth, digest, summarize
from model_forensics.specificity import assess_repairs


def artifact(identity, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"identity": identity, "sha256": hashlib.sha256(raw).hexdigest()}


def build():
    baseline = predictions(lambda x: x, WEIGHTS)
    bad = predictions(reverse, WEIGHTS)
    regressed = comparison("retained-regressed", bad)
    repairs = {
        "restore-input-order": comparison("input-repair", baseline),
        "reverse-model-weights": comparison(
            "weight-repair", predictions(reverse, tuple(reversed(WEIGHTS)))
        ),
    }
    assessment = assess_repairs(regressed, repairs)
    record = {
        "schema_version": "incident-ledger/0.1",
        "incident": {
            "schema_version": "release-incident/0.1",
            "incident_id": "retained-ambiguous-repairs",
            "use": "retrospective_software",
            "good_release": artifact("retained-baseline-predictions", baseline),
            "bad_release": artifact("retained-regressed-predictions", bad),
            "environment": artifact("fixture-definition", {"weights": WEIGHTS}),
            "candidates": [
                {
                    "candidate_id": name,
                    "category": category,
                    "visible_diff": artifact(name, description),
                    "description": description,
                }
                for name, category, description in (
                    ("restore-input-order", "preprocessing", "Reverse the fixture feature order"),
                    ("reverse-model-weights", "configuration", "Reverse fixed fixture weights"),
                )
            ],
            "candidate_completeness": "declared_complete",
            "target_functional": "Retained fixture exact label accuracy on all eight cases",
            "evaluation": artifact("retained-comparison-policy", regressed["slices"]),
            "protected_behavior": "None: retained deterministic arithmetic fixture only",
            "intervention_semantics": "Apply one named repair to retained regressed predictions",
            "training_process": "No training; evaluate fixed linear weights",
            "randomness_treatment": "No randomness; one deterministic execution per repair",
            "restoration_meaning": "Exactly recover baseline labels on the retained eight cases",
            "engineer_visible_evidence": [
                artifact(
                    "fixture-change-definition",
                    {"weights": WEIGHTS, "input_change": "feature-order reversal"},
                )
            ],
            "max_intervention_attempts": 2,
        },
        "runs": [
            {
                "run_id": name,
                "change_set": [name],
                "randomness_id": "deterministic",
                "status": "completed",
                "target_value": report["slices"][0]["candidate_accuracy"],
                "protected_value": None,
                "artifact": artifact(name + "-comparison", report),
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "failure": None,
            }
            for name, report in sorted(repairs.items())
        ],
        "decision": {
            "status": "ambiguity_set",
            "hypotheses": [[x] for x in assessment["successful_repairs"]],
            "evidence_run_ids": sorted(repairs),
            "policy": artifact("retained-repair-assessment", "specificity.py:assess_repairs"),
            "rationale": "Retained exact repair assessment reports multiple successful repairs",
        },
    }
    ledger = Ledger.model_validate_json(json.dumps(record))
    # Evaluator-only restorative truth, not historical cause; never passed to assess_repairs.
    truth = Truth.model_validate_json(
        json.dumps(
            {
                "incident_sha256": digest(ledger.incident),
                "responsibility_sets": [[name] for name in sorted(repairs)],
                "provenance": artifact("retained-restorative-comparisons", repairs),
            }
        )
    )
    return ledger, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ledger, truth = build()
    args.out.mkdir()  # Refuse reuse, including a prior failed example.
    for name, record in (
        ("ledger", ledger.model_dump(mode="json")),
        ("truth", truth.model_dump(mode="json")),
        ("report", summarize(ledger, truth)),
    ):
        (args.out / f"{name}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
