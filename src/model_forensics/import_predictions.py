"""Build an investigation from saved classification predictions without running models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_forensics.investigation_report import render
from model_forensics.prediction_assessment import assess_predictions
from model_forensics.prediction_sources import parse_prediction_csv
from model_forensics.release_compare import Comparison


def import_predictions(policy_path: Path, predictions_path: Path, output: Path) -> dict:
    """Validate every record before creating a new, self-contained investigation.

    The policy contains cases, slices and declared_changes. The long-form CSV has
    release_id,case_id,observed columns; baseline and candidate are reserved roles.
    All other release IDs denote supplied repairs. No execution costs are inferred.
    """
    policy_bytes = policy_path.read_bytes()
    prediction_bytes = predictions_path.read_bytes()
    policy = json.loads(policy_bytes)
    if not isinstance(policy, dict) or set(policy) != {"cases", "slices", "declared_changes"}:
        raise ValueError("policy requires exactly cases, slices and declared_changes")
    if not isinstance(policy["declared_changes"], dict):
        raise ValueError("declared_changes must be an object")
    releases = parse_prediction_csv(prediction_bytes)
    comparisons = {
        name: Comparison.model_validate_json(
            json.dumps(
                {
                    "cases": policy["cases"],
                    "slices": policy["slices"],
                    "baseline": releases["baseline"].model_dump(mode="json"),
                    "candidate": release.model_dump(mode="json"),
                }
            )
        )
        for name, release in releases.items()
        if name != "baseline"
    }
    report = assess_predictions(comparisons.pop("candidate"), comparisons)
    order = ["baseline", "candidate", *sorted(comparisons)]
    plan = {
        "schema_version": "investigation-plan/0.1",
        "cases": policy["cases"],
        "slices": policy["slices"],
        "execution_order": order,
        "declared_changes": policy["declared_changes"],
        "change_provenance": "caller_supplied",
        "automatic_retry": False,
        "record_origin": "imported_predictions",
        "models_executed": False,
        "source_sha256": {
            "source-policy.json": hashlib.sha256(policy_bytes).hexdigest(),
            "source-predictions.csv": hashlib.sha256(prediction_bytes).hexdigest(),
        },
    }
    if not comparisons:
        plan["investigation_stage"] = "initial_comparison"
    # Reject non-finite values in declarations before leaving any output.
    json.dumps(plan, allow_nan=False)
    output.mkdir()
    (output / "source-policy.json").write_bytes(policy_bytes)
    (output / "source-predictions.csv").write_bytes(prediction_bytes)

    def write(name: str, value: dict) -> None:
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

    write("plan.json", plan)
    for index, name in enumerate(order):
        write(
            f"execution-{index:03d}.json",
            {
                "release": releases[name].model_dump(mode="json"),
                "wall_seconds": None,
                "current_process_cpu_seconds": None,
                "cpu_scope": "not_measured",
                "record_origin": "imported_predictions",
            },
        )
    report["historical_cause"] = "not_identified"
    write("report.json", report)
    render(output, output / "index.html")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="JSON cases, slices and declared_changes")
    parser.add_argument("predictions", type=Path, help="Long-form prediction CSV")
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    try:
        report = import_predictions(args.policy, args.predictions, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {"report": str(args.output / "index.html"), "assessment": report["assessment"]},
            indent=2,
        )
    )
    # Successful import is distinct from whether any repair satisfies the policy.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
