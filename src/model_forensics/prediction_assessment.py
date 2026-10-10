"""Assess an initial release comparison or supplied repair predictions."""

from model_forensics.release_compare import Comparison, compare_releases
from model_forensics.repair_compare import RepairComparison, compare_repairs


def assess_predictions(candidate: Comparison, repairs: dict[str, Comparison]) -> dict:
    """Keep missing repair evidence distinct from evaluated, unsuccessful repairs."""
    if repairs:
        return compare_repairs(RepairComparison(regressed=candidate, repairs=repairs))
    return {
        "schema_version": "initial-release-comparison/0.1",
        "candidate": compare_releases(candidate),
        "repairs": {},
        "assessment": {
            "status": "not_evaluated",
            "reason": "No repair predictions were supplied. Repairs have not been evaluated.",
            "successful_repairs": [],
            "evaluated_repairs": [],
        },
        "evidence_scope": "supplied predictions and declared policy only",
    }
