"""Assess saved repair predictions under one identical evaluation policy.

No model is loaded or trained. Comparison reports are computed from validated raw
records rather than accepting caller-supplied pass/fail conclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.release_compare import Comparison, Identifier, StrictRecord, compare_releases
from model_forensics.specificity import assess_repairs


class RepairComparison(StrictRecord):
    regressed: Comparison
    repairs: dict[Identifier, Comparison]


def compare_repairs(spec: RepairComparison) -> dict:
    regression = compare_releases(spec.regressed)
    repairs = {name: compare_releases(value) for name, value in sorted(spec.repairs.items())}
    assessment = assess_repairs(regression, repairs)
    return {
        "schema_version": "repair-comparison/0.1",
        "assessment": assessment,
        "regressed": regression,
        "repairs": repairs,
        "evidence_scope": "supplied predictions and declared interventions only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing regressed and repairs records")
    args = parser.parse_args()
    try:
        spec = RepairComparison.model_validate_json(args.input.read_text(encoding="utf-8"))
        result = compare_repairs(spec)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    # Success means the assessment completed, never that a cause was identified.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
