from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class EvalCase:
    """One deterministic behavioral evaluation case."""

    case_id: str
    expected: str
    observed: str


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate score plus failing case identifiers."""

    score: float
    total: int
    failed_case_ids: tuple[str, ...]


def exact_match(cases: list[EvalCase]) -> EvalSummary:
    """Score exact normalized string agreement."""
    if not cases:
        raise ValueError("at least one evaluation case is required")

    matches = [case.expected.strip() == case.observed.strip() for case in cases]
    failed = tuple(
        case.case_id for case, matched in zip(cases, matches, strict=True) if not matched
    )
    return EvalSummary(score=mean(matches), total=len(cases), failed_case_ids=failed)
