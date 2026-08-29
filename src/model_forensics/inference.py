from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from model_forensics.eval import EvalCase, EvalSummary, exact_match

VALID_LABELS = frozenset({"ACCEPT", "REJECT"})


@dataclass(frozen=True)
class EvalInput:
    """One prepared evaluation prompt and its expected label."""

    case_id: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class GenerationRecord:
    """Raw model output retained alongside the strict scored output."""

    case_id: str
    prompt: str
    expected: str
    raw_output: str

    @property
    def observed(self) -> str:
        """Return the strict output after trimming outer whitespace only."""

        return self.raw_output.strip()

    @property
    def parsed_label(self) -> str | None:
        """Parse a one-token ACCEPT/REJECT label while ignoring letter case."""

        normalized = self.observed.upper()
        return normalized if normalized in VALID_LABELS else None

    def to_record(self) -> dict[str, str | None]:
        """Return a stable JSON-serializable record."""

        record = asdict(self)
        record["observed"] = self.observed
        record["parsed_label"] = self.parsed_label
        return record


def load_eval_inputs(path: str | Path) -> tuple[EvalInput, ...]:
    """Load prepared evaluation examples from JSON Lines."""

    inputs: list[EvalInput] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                inputs.append(
                    EvalInput(
                        case_id=payload["example_id"],
                        prompt=payload["prompt"],
                        expected=payload["response"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing required evaluation field {exc.args[0]!r} at line {line_number}"
                ) from exc

    if not inputs:
        raise ValueError("at least one prepared evaluation input is required")
    return tuple(inputs)


def score_generation_records(records: tuple[GenerationRecord, ...]) -> EvalSummary:
    """Score raw generations using strict whitespace-only normalization."""

    return exact_match(
        [
            EvalCase(
                case_id=record.case_id,
                expected=record.expected,
                observed=record.observed,
            )
            for record in records
        ]
    )


def score_label_generation_records(records: tuple[GenerationRecord, ...]) -> EvalSummary:
    """Score one-token labels while treating letter case as non-behavioral."""

    return exact_match(
        [
            EvalCase(
                case_id=record.case_id,
                expected=record.expected,
                observed=record.parsed_label or "",
            )
            for record in records
        ]
    )


def write_generation_records(records: tuple[GenerationRecord, ...], path: str | Path) -> None:
    """Write model generations as stable JSON Lines."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_record(), sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash of a file exactly as consumed by a run."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
