"""Prediction-only release comparison pilot; no causal attribution or training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Case(StrictRecord):
    case_id: Identifier
    expected: Identifier


class Prediction(StrictRecord):
    case_id: Identifier
    observed: Identifier


class Release(StrictRecord):
    release_id: Identifier
    predictions: Annotated[tuple[Prediction, ...], Field(min_length=1)]


class Slice(StrictRecord):
    name: Identifier
    case_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    maximum_accuracy_drop: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)


class Comparison(StrictRecord):
    cases: Annotated[tuple[Case, ...], Field(min_length=1)]
    baseline: Release
    candidate: Release
    slices: Annotated[tuple[Slice, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_identity(self) -> Comparison:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case IDs must be unique")
        if self.baseline.release_id == self.candidate.release_id:
            raise ValueError("release IDs must differ")
        for release in (self.baseline, self.candidate):
            predicted = [item.case_id for item in release.predictions]
            if len(predicted) != len(set(predicted)) or set(predicted) != set(ids):
                raise ValueError("each release must predict exactly the evaluation case IDs")
        names = [item.name for item in self.slices]
        if len(names) != len(set(names)):
            raise ValueError("slice names must be unique")
        for item in self.slices:
            if len(item.case_ids) != len(set(item.case_ids)):
                raise ValueError("slice case IDs must be unique")
            if not set(item.case_ids) <= set(ids):
                raise ValueError("slice contains unknown case IDs")
        return self


def compare_releases(spec: Comparison) -> dict:
    """Report exact-label changes on declared slices, with explicit tolerances.

    Slice overlap is allowed; counts must not be pooled as independent evidence.
    The digest identifies supplied prediction records, not a verified training run.
    Labels are compared exactly, without implicit whitespace normalization.
    """
    expected = {case.case_id: case.expected for case in spec.cases}
    baseline = {item.case_id: item.observed for item in spec.baseline.predictions}
    candidate = {item.case_id: item.observed for item in spec.candidate.predictions}
    canonical = spec.model_dump(mode="json")
    canonical["cases"].sort(key=lambda item: item["case_id"])
    for role in ("baseline", "candidate"):
        canonical[role]["predictions"].sort(key=lambda item: item["case_id"])
    canonical["slices"].sort(key=lambda item: item["name"])
    for item in canonical["slices"]:
        item["case_ids"].sort()
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    # Repairs must share the actual cases, labels, baseline predictions and slice
    # membership. Equal counts and baseline accuracy cannot establish this.
    policy = {key: canonical[key] for key in ("cases", "baseline", "slices")}
    policy_encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    rows = []
    for item in sorted(spec.slices, key=lambda item: item.name):
        ids = sorted(item.case_ids)
        before = {key: baseline[key] == expected[key] for key in ids}
        after = {key: candidate[key] == expected[key] for key in ids}
        drop = (sum(before.values()) - sum(after.values())) / len(ids)
        rows.append(
            {
                "name": item.name,
                "count": len(ids),
                "baseline_accuracy": sum(before.values()) / len(ids),
                "candidate_accuracy": sum(after.values()) / len(ids),
                "accuracy_drop": drop,
                "maximum_accuracy_drop": item.maximum_accuracy_drop,
                "passed": drop <= item.maximum_accuracy_drop,
                "regressed_case_ids": [key for key in ids if before[key] and not after[key]],
                "improved_case_ids": [key for key in ids if not before[key] and after[key]],
            }
        )
    return {
        "schema_version": "release-comparison-pilot/0.2",
        "input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "evaluation_policy_sha256": hashlib.sha256(policy_encoded.encode()).hexdigest(),
        "baseline_release_id": spec.baseline.release_id,
        "candidate_release_id": spec.candidate.release_id,
        "passed": all(item["passed"] for item in rows),
        "slices": rows,
        "causal_attribution": "not_assessed",
        "statistical_significance": "not_assessed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON comparison record")
    args = parser.parse_args()
    try:
        spec = Comparison.model_validate_json(args.input.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = compare_releases(spec)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
