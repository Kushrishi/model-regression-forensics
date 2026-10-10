"""Compare MRF decisions with an ordinary count-based saved-prediction diff.

The baseline does not call MRF's slice comparator or repair assessor. Agreement
is correctness evidence on the supplied records, not a measure of user benefit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from model_forensics.import_predictions import assess_prediction_sources


def ordinary_diff(policy: dict, prediction_bytes: bytes) -> dict:
    expected = {row["case_id"]: row["expected"] for row in policy["cases"]}
    if len(expected) != len(policy["cases"]):
        raise ValueError("duplicate evaluation cases")
    reader = csv.DictReader(io.StringIO(prediction_bytes.decode("utf-8-sig"), newline=""))
    if sorted(reader.fieldnames or []) != ["case_id", "observed", "release_id"]:
        raise ValueError("invalid prediction columns")
    releases: dict[str, dict[str, str]] = {}
    for row in reader:
        if None in row or any(value is None or not value.strip() for value in row.values()):
            raise ValueError("incomplete prediction row")
        release = releases.setdefault(row["release_id"], {})
        if row["case_id"] in release:
            raise ValueError("duplicate release/case prediction")
        release[row["case_id"]] = row["observed"]
    if not {"baseline", "candidate"} <= releases.keys():
        raise ValueError("baseline and candidate required")
    if any(set(release) != set(expected) for release in releases.values()):
        raise ValueError("identical evaluation case sets required")
    if len({row["name"] for row in policy["slices"]}) != len(policy["slices"]):
        raise ValueError("duplicate slice name")
    decisions = {}
    for name, predictions in sorted(releases.items()):
        slices = []
        for item in sorted(policy["slices"], key=lambda row: row["name"]):
            ids = item["case_ids"]
            if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(expected):
                raise ValueError("invalid slice membership")
            before = sum(releases["baseline"][key] == expected[key] for key in ids)
            after = sum(predictions[key] == expected[key] for key in ids)
            # Subtract integer correct counts before division at exact boundaries.
            drop = (before - after) / len(ids)
            floor = item.get("minimum_accuracy")
            passed = drop <= item.get("maximum_accuracy_drop", 0.0) and (
                floor is None or after / len(ids) >= floor
            )
            slices.append({"name": item["name"], "passed": passed})
        decisions[name] = {
            "passed": all(row["passed"] for row in slices),
            "slices": slices,
            "changed_case_ids": sorted(
                key for key in expected if predictions[key] != releases["baseline"][key]
            ),
            "new_error_case_ids": sorted(
                key
                for key in expected
                if releases["baseline"][key] == expected[key] and predictions[key] != expected[key]
            ),
            "corrected_case_ids": sorted(
                key
                for key in expected
                if releases["baseline"][key] != expected[key] and predictions[key] == expected[key]
            ),
        }
    return decisions


def compare_decision_paths(policy_path: Path, predictions_path: Path, output: Path) -> dict:
    # Run both paths before publication; MRF validates the supplied policy/schema.
    policy_bytes, prediction_bytes = policy_path.read_bytes(), predictions_path.read_bytes()
    policy, _, mrf = assess_prediction_sources(policy_bytes, prediction_bytes)
    baseline = ordinary_diff(policy, prediction_bytes)
    candidate = mrf.get("candidate", mrf.get("regressed"))
    mrf_reports = {"candidate": candidate, **mrf["repairs"]}
    agreement = {}
    # Policy slices need not cover all cases. Compare case evidence on covered IDs.
    covered = {key for row in policy["slices"] for key in row["case_ids"]}
    for name, report in mrf_reports.items():
        ordinary = baseline[name]
        agreement[name] = (
            ordinary["passed"] == report["passed"]
            and ordinary["slices"]
            == [{"name": row["name"], "passed": row["passed"]} for row in report["slices"]]
            and set(ordinary["new_error_case_ids"]) & covered
            == {key for row in report["slices"] for key in row["regressed_case_ids"]}
            and set(ordinary["corrected_case_ids"]) & covered
            == {key for row in report["slices"] for key in row["improved_case_ids"]}
        )
    result = {
        "schema": "release-decision-path-comparison/1",
        "source_sha256": {
            "policy": hashlib.sha256(policy_bytes).hexdigest(),
            "predictions": hashlib.sha256(prediction_bytes).hexdigest(),
        },
        "ordinary_diff": baseline,
        "mrf_slice_decision_agreement": agreement,
        "all_supplied_decisions_agree": all(agreement.values()),
        "complete_rollback": {
            "record_origin": "reuse supplied baseline predictions; no model rollback executed",
            "passed": baseline["baseline"]["passed"],
        },
        "supplied_repairs": sorted(mrf["repairs"]),
        "mrf_repair_assessment": mrf["assessment"],
        "retraining": "not_inferred; only explicitly supplied predictions can be evaluated",
        "model_execution_cost": "unknown for saved-prediction imports",
        "independent_user_benefit": "not_measured",
        "historical_cause": "not_identified",
    }
    output.mkdir()
    (output / "source-policy.json").write_bytes(policy_bytes)
    (output / "source-predictions.csv").write_bytes(prediction_bytes)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        report = compare_decision_paths(args.policy, args.predictions, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["all_supplied_decisions_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
