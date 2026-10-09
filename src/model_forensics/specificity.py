"""Small decision layer for controlled repair evidence.

This module does not infer historical causality. It only reports whether the
declared repair evidence supports zero, one, or multiple behavior-restoring
interventions under the same evaluation policy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


def _policy_signature(report: dict) -> tuple:
    """Return the evaluation policy that must match across repair reports."""
    try:
        rows = report["slices"]
        baseline_id = report["baseline_release_id"]
        policy_digest = report["evaluation_policy_sha256"]
    except KeyError as error:
        raise ValueError(f"missing comparison field: {error.args[0]}") from error

    if not isinstance(rows, list) or not rows:
        raise ValueError("comparison report must contain nonempty slices")
    if not isinstance(policy_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", policy_digest):
        raise ValueError("comparison report requires an evaluation policy SHA-256")

    signature = []
    for row in rows:
        try:
            signature.append(
                (
                    row["name"],
                    row["count"],
                    row["maximum_accuracy_drop"],
                    row["baseline_accuracy"],
                )
            )
        except KeyError as error:
            raise ValueError(f"missing slice field: {error.args[0]}") from error
    return baseline_id, policy_digest, tuple(sorted(signature))


def assess_repairs(regressed_report: dict, repair_reports: Mapping[str, dict]) -> dict:
    """Assess whether declared repair evidence is unique or ambiguous.

    A successful repair shows that an intervention can restore the declared
    behavior under the supplied policy. It does not by itself identify the
    historical cause of the regression.
    """
    if regressed_report.get("passed") is not False:
        raise ValueError("regressed report must fail the declared comparison policy")

    reference = _policy_signature(regressed_report)
    successful: list[str] = []
    evaluated: list[str] = []

    for intervention_id, report in sorted(repair_reports.items()):
        if not intervention_id or not intervention_id.strip():
            raise ValueError("intervention IDs must be nonblank")
        if _policy_signature(report) != reference:
            raise ValueError("all repair reports must use the same baseline and evaluation policy")
        evaluated.append(intervention_id)
        if report.get("passed") is True:
            successful.append(intervention_id)
        elif report.get("passed") is not False:
            raise ValueError("repair report must contain a boolean passed field")

    if not evaluated:
        status = "insufficient_evidence"
        reason = "No repair interventions were evaluated."
    elif len(successful) == 0:
        status = "no_supported_repair"
        reason = "None of the declared interventions restored the required behavior."
    elif len(successful) == 1:
        status = "single_supported_repair"
        reason = (
            "One declared intervention restored the required behavior, but repair alone does not "
            "identify the historical cause."
        )
    else:
        status = "ambiguous_repairs"
        reason = (
            "Multiple distinct declared interventions restored the required behavior, so the "
            "observed repair evidence does not support a unique historical explanation."
        )

    return {
        "schema_version": "repair-specificity/0.1",
        "status": status,
        "evaluated_repairs": evaluated,
        "successful_repairs": successful,
        "historical_cause": "not_identified",
        "reason": reason,
    }
