"""Prepare fair first-use decision packets; audit actual submitted answers.

No participants, observations, effort measurements or benefit estimates are
created. Facilitator gold is kept outside either participant packet.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from model_forensics.decision_comparison import ordinary_diff
from model_forensics.import_predictions import assess_prediction_sources, import_predictions


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(policy_path: Path, predictions_path: Path, output: Path) -> dict:
    policy_bytes, prediction_bytes = policy_path.read_bytes(), predictions_path.read_bytes()
    policy, _, _ = assess_prediction_sources(policy_bytes, prediction_bytes)
    ordinary = ordinary_diff(policy, prediction_bytes)
    repairs = sorted(set(ordinary) - {"baseline", "candidate"})
    hashes = {
        "source-policy.json": digest(policy_bytes),
        "source-predictions.csv": digest(prediction_bytes),
    }
    gold = {
        "candidate_passes": ordinary["candidate"]["passed"],
        "complete_rollback_passes": ordinary["baseline"]["passed"],
        "passing_supplied_repairs": [name for name in repairs if ordinary[name]["passed"]],
        "changed_case_ids": ordinary["candidate"]["changed_case_ids"],
        "new_error_case_ids": ordinary["candidate"]["new_error_case_ids"],
        "historical_cause_identified": False,
        "retraining_evaluated": False,
        "execution_cost_known": False,
        "repair_evidence_supplied": bool(repairs),
        "distinct_passing_repairs": len([name for name in repairs if ordinary[name]["passed"]]),
        "source_hashes": hashes,
    }
    output.mkdir()
    baseline = output / "ordinary"
    baseline.mkdir()
    (baseline / "source-policy.json").write_bytes(policy_bytes)
    (baseline / "source-predictions.csv").write_bytes(prediction_bytes)
    # A complete runnable standard-library baseline, not a deliberately weak diff.
    script = (
        "import csv,io,json,hashlib\nfrom pathlib import Path\n\n"
        + inspect.getsource(ordinary_diff)
        + "\nroot=Path(__file__).parent\n"
        + "policy=json.loads((root/'source-policy.json').read_text())\n"
        + "predictions=(root/'source-predictions.csv').read_bytes()\n"
        + "print(json.dumps(ordinary_diff(policy,predictions),indent=2))\n"
    )
    (baseline / "ordinary_diff.py").write_text(script)
    write(
        baseline / "decision.json",
        {
            "releases": ordinary,
            "policy": policy,
            "complete_rollback": "reuse supplied baseline predictions; no execution",
            "repair_evidence": (
                "supplied predictions only; multiple passing repairs do not identify a cause"
            ),
            "historical_cause": "unknown",
            "retraining": "not_evaluated",
            "execution_cost": "unknown",
        },
    )
    mrf = output / "mrf"
    import_predictions(policy_path, predictions_path, mrf)
    task = {
        "schema": "mrf-first-use-answer/1",
        "answers": dict.fromkeys(gold),
        "participant": {
            "independent_of_implementation_and_fixture": None,
            "prior_exposure": None,
            "experience": None,
        },
        "effort": {"active_seconds": None, "provenance_lookup_seconds": None},
        "assistance": [],
        "interpretation_notes": None,
        "provenance_steps": None,
        "repair_ambiguity_explanation": None,
        "missing_evidence_explanation": None,
    }
    instructions = (
        "Use this packet to decide whether candidate and complete rollback"
        " satisfy the supplied policy, "
        "which supplied repairs pass, and what the evidence cannot establish. "
        "Return answer-template.json separately; null means unanswered, "
        "never failure. "
        "Record an unaided answer before assistance, then retain revisions"
        " separately. "
        "For provenance, identify original input hashes and explain how to"
        " detect a changed source file. "
        "Record actual active effort and steps only if measured; do not "
        "estimate missing timings. "
        "Disclose prior exposure, including public examples. This is a first-use exercise, "
        "not a blinded study.\n"
    )
    verifier = (
        "import hashlib,json\nfrom pathlib import Path\nroot=Path(__file__).parent\n"
        "expected=json.loads((root/'source-hashes.json').read_text())\n"
        "actual={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}\n"
        "print(json.dumps({'matches':actual==expected,'actual':actual},indent=2))\n"
        "raise SystemExit(0 if actual==expected else 1)\n"
    )
    for directory in (baseline, mrf):
        write(directory / "source-hashes.json", hashes)
        write(directory / "answer-template.json", task)
        (directory / "TASK.txt").write_text(instructions)
        (directory / "verify_sources.py").write_text(verifier)
    write(output / "facilitator-gold.json", gold)
    protocol = {
        "schema": "mrf-first-use-protocol/1",
        "source_hashes": hashes,
        "conditions": ["ordinary", "mrf"],
        "participant_observations": "none",
        "assignment": (
            "Facilitator assigns condition/order before session; use disjoint "
            "matched tasks for crossover, counterbalance order; no same-case "
            "second attempt as independent evidence."
        ),
        "eligibility": (
            "Independent of implementation and fixtures; disclose experience "
            "and prior exposure. Owner/author rehearsals are tooling checks "
            "only."
        ),
        "outcomes": (
            "Correct decisions; overclaims; unanswered items; active effort; "
            "provenance steps; ambiguity/missing-evidence explanations; "
            "assistance needed."
        ),
        "interpretation": (
            "Descriptive first-use evidence only. Equal inputs, policy, repair"
            " records and hash verifier. No benefit inferred from agreement or"
            " a single anecdote; qualitative explanations require human "
            "review."
        ),
        "facilitator_boundary": (
            "Distribute only one condition directory; never distribute this "
            "directory or facilitator-gold.json. No outreach authorized by "
            "preparation."
        ),
    }
    write(output / "protocol.json", protocol)
    return protocol


def audit(gold: dict, submission: dict) -> dict:
    if submission.get("schema") != "mrf-first-use-answer/1":
        raise ValueError("unsupported submitted answer schema")
    answers = submission.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(gold):
        raise ValueError("submitted answer keys must match prospective tasks")
    scored = {}
    for key, expected in gold.items():
        actual = answers[key]
        if actual is None:
            scored[key] = "unanswered"
            continue
        if type(actual) is not type(expected):
            raise ValueError("wrong answer type: " + key)
        if isinstance(expected, list):
            if any(type(value) is not str for value in actual) or len(set(actual)) != len(actual):
                raise ValueError("invalid answer set: " + key)
            correct = sorted(actual) == expected
        else:
            correct = actual == expected
        scored[key] = "correct" if correct else "incorrect"
    effort = submission.get("effort", {})
    if not isinstance(effort, dict):
        raise ValueError("effort must be an object with optional measured values")
    for name in ("active_seconds", "provenance_lookup_seconds"):
        value = effort.get(name)
        if value is not None and (type(value) not in (int, float) or not 0 <= value < float("inf")):
            raise ValueError("invalid measured effort")
    return {
        "schema": "mrf-submission-audit/1",
        "objective_answers": scored,
        "submitted_effort": effort,
        "participant_declaration": submission.get("participant"),
        "assistance": submission.get("assistance"),
        "qualitative_review": "pending human review; steps/explanations are not graded by keyword",
        "independence": "participant declaration only; not independently verified",
        "comparative_user_benefit": "not_established",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("policy", "predictions", "output"):
        preparation.add_argument(name, type=Path)
    scoring = commands.add_parser("audit")
    for name in ("gold", "submission", "output"):
        scoring.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.policy, args.predictions, args.output)
    else:
        gold_bytes, submission_bytes = args.gold.read_bytes(), args.submission.read_bytes()
        result = audit(json.loads(gold_bytes), json.loads(submission_bytes))
        result["source_sha256"] = {
            "gold": digest(gold_bytes),
            "submission": digest(submission_bytes),
        }
        write(args.output, result)


if __name__ == "__main__":
    main()
