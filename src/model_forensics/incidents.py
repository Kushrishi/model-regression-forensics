"""Non-training incident ledger and retrospective decision metrics.

Records specify an estimand; they do not certify it or infer historical cause.
No numerical diagnosis policy or restoration threshold is supplied here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from model_forensics.release_compare import Identifier, StrictRecord

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Hypothesis = Annotated[tuple[Identifier, ...], Field(min_length=1)]


class Artifact(StrictRecord):
    identity: Identifier
    sha256: Digest


class Candidate(StrictRecord):
    candidate_id: Identifier
    category: Literal[
        "data",
        "labels",
        "preprocessing",
        "sampling",
        "augmentation",
        "loss",
        "configuration",
        "other",
    ]
    visible_diff: Artifact
    description: Identifier


class Incident(StrictRecord):
    schema_version: Literal["release-incident/0.1"]
    incident_id: Identifier
    use: Literal["discovery", "retrospective_software", "synthetic_software"]
    good_release: Artifact
    bad_release: Artifact
    environment: Artifact
    candidates: Annotated[tuple[Candidate, ...], Field(min_length=1)]
    candidate_completeness: Literal["declared_complete", "incomplete", "unknown"]
    target_functional: Identifier
    evaluation: Artifact
    protected_behavior: Identifier  # Use an explicit justification if none applies.
    intervention_semantics: Identifier
    training_process: Identifier
    randomness_treatment: Identifier
    restoration_meaning: Identifier
    engineer_visible_evidence: Annotated[tuple[Artifact, ...], Field(min_length=1)]
    max_intervention_attempts: int = Field(ge=0)

    @model_validator(mode="after")
    def identities(self) -> Incident:
        ids = [x.candidate_id for x in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        if self.good_release == self.bad_release:
            raise ValueError("good and bad release identities must differ")
        return self


class InterventionRun(StrictRecord):
    run_id: Identifier
    change_set: Hypothesis
    randomness_id: Identifier
    status: Literal["completed", "failed", "timeout"]
    target_value: Finite | None
    protected_value: Finite | None
    artifact: Artifact
    wall_seconds: Nonnegative
    cpu_seconds: Nonnegative
    failure: Identifier | None

    @model_validator(mode="after")
    def payload(self) -> InterventionRun:
        if len(set(self.change_set)) != len(self.change_set):
            raise ValueError("a change set cannot repeat a candidate")
        if self.status == "completed":
            if self.target_value is None or self.failure is not None:
                raise ValueError("completed runs require a target measurement and no failure")
        elif (
            self.target_value is not None
            or self.protected_value is not None
            or self.failure is None
        ):
            raise ValueError(
                "failed/timeout runs require failure evidence, not usable measurements"
            )
        return self


class Decision(StrictRecord):
    status: Literal["specific", "ambiguity_set", "insufficient_evidence", "abstain"]
    hypotheses: tuple[Hypothesis, ...]
    evidence_run_ids: tuple[Identifier, ...]
    policy: Artifact
    rationale: Identifier

    @model_validator(mode="after")
    def shape(self) -> Decision:
        n = len(self.hypotheses)
        if (self.status == "specific" and n != 1) or (self.status == "ambiguity_set" and n < 2):
            raise ValueError("specific requires one hypothesis; ambiguity requires at least two")
        if self.status in ("insufficient_evidence", "abstain") and n:
            raise ValueError("noncommitting outputs must not name hypotheses")
        canonical = [frozenset(h) for h in self.hypotheses]
        if any(len(set(h)) != len(h) for h in self.hypotheses) or len(set(canonical)) != n:
            raise ValueError("hypotheses must be unique nonrepeating change sets")
        if len(set(self.evidence_run_ids)) != len(self.evidence_run_ids):
            raise ValueError("decision evidence run IDs must be unique")
        return self


class Ledger(StrictRecord):
    schema_version: Literal["incident-ledger/0.1"]
    incident: Incident
    runs: tuple[InterventionRun, ...]
    decision: Decision

    @model_validator(mode="after")
    def references(self) -> Ledger:
        ids = [r.run_id for r in self.runs]
        keys = [(frozenset(r.change_set), r.randomness_id) for r in self.runs]
        candidates = {c.candidate_id for c in self.incident.candidates}
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError("run IDs and intervention/randomness pairs must be unique")
        if len(self.runs) > self.incident.max_intervention_attempts:
            raise ValueError("intervention attempt budget exceeded (failures also count)")
        sets = [r.change_set for r in self.runs] + list(self.decision.hypotheses)
        if any(not set(s) <= candidates for s in sets):
            raise ValueError("unknown candidate in run or decision")
        if not set(self.decision.evidence_run_ids) <= set(ids):
            raise ValueError("decision refers to unknown intervention evidence")
        return self


class Truth(StrictRecord):
    """Separate evaluator input; never a field in the engineer-visible manifest."""

    incident_sha256: Digest
    responsibility_sets: Annotated[tuple[Hypothesis, ...], Field(min_length=1)]
    provenance: Artifact

    @model_validator(mode="after")
    def sets(self) -> Truth:
        canonical = [frozenset(h) for h in self.responsibility_sets]
        if len(set(canonical)) != len(canonical) or any(
            len(set(h)) != len(h) for h in self.responsibility_sets
        ):
            raise ValueError("truth responsibility sets must be unique without repeated candidates")
        return self


def digest(record: StrictRecord) -> str:
    encoded = json.dumps(
        record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def summarize(ledger: Ledger, truth: Truth | None = None) -> dict:
    """Preserve every repeated outcome and count attempts, without choosing a policy.

    Hashes identify supplied records; payload authenticity/execution is not inferred.
    Truth may contain an omitted candidate. It is evaluator-only and process-specific.
    """
    incident_hash = digest(ledger.incident)
    if truth is not None and truth.incident_sha256 != incident_hash:
        raise ValueError("truth is bound to a different incident specification")
    report = {
        "schema_version": "incident-report/0.1",
        "use": ledger.incident.use,
        "incident_id": ledger.incident.incident_id,
        "incident_sha256": incident_hash,
        "ledger_sha256": digest(ledger),
        "historical_cause": "not_inferred",
        "decision": ledger.decision.model_dump(mode="json"),
        "attempts": len(ledger.runs),
        "completed": sum(r.status == "completed" for r in ledger.runs),
        "cpu_seconds": sum(r.cpu_seconds for r in ledger.runs),
        "sum_worker_wall_seconds": sum(r.wall_seconds for r in ledger.runs),
        "runs": [r.model_dump(mode="json") for r in ledger.runs],
        "evaluation": None,
    }
    if truth is not None:
        actual = {frozenset(h) for h in truth.responsibility_sets}
        returned = {frozenset(h) for h in ledger.decision.hypotheses}
        specific = ledger.decision.status == "specific"
        report["evaluation"] = {
            "truth_sha256": digest(truth),
            "specific_commitment": specific,
            # Naming one of several responsible sets is still unjustified specificity.
            "wrong_specific": specific and (len(actual) != 1 or returned != actual),
            "all_responsible_sets_contained": bool(returned) and actual <= returned,
            "any_responsible_set_contained": bool(returned & actual),
            "returned_hypothesis_count": len(returned),
            "returned_candidate_count": len(set().union(*returned)) if returned else 0,
        }
    return report


def decision_metrics(records: list[tuple[Ledger, Truth]]) -> dict:
    """One report per incident; seeds/runs are never counted as incidents."""
    reports = [summarize(ledger, truth) for ledger, truth in records]
    if not reports or len({r["incident_id"] for r in reports}) != len(reports):
        raise ValueError("require nonempty reports with unique incident IDs")
    if any(
        r.get("schema_version") != "incident-report/0.1" or r.get("evaluation") is None
        for r in reports
    ):
        raise ValueError("metrics require separately truth-evaluated incident reports")
    n = len(reports)
    committed = sum(r["evaluation"]["specific_commitment"] for r in reports)
    wrong = sum(r["evaluation"]["wrong_specific"] for r in reports)
    return {
        "incident_count": n,
        "specific_commitments": committed,
        "wrong_specific_decisions": wrong,
        "commitment_coverage": committed / n,
        "wrong_specific_rate_all_incidents": wrong / n,
        "risk_among_committed": wrong / committed if committed else None,
        "intervention_attempts": sum(r["attempts"] for r in reports),
    }


def read_record(path: Path, kind):
    with path.open("rb") as f:
        raw = f.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("record exceeds the 8-MiB input bound")
    return kind.model_validate_json(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--truth", type=Path, help="separate evaluator-only record")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        ledger = read_record(args.ledger, Ledger)
        truth = read_record(args.truth, Truth) if args.truth else None
        result = summarize(ledger, truth)
        with args.out.open("x", encoding="utf-8") as out:
            out.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"incident record error: {error}\n")


if __name__ == "__main__":
    main()
