from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from model_forensics.lineage import ArtifactChange, DiagnosticManifest

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class CandidateCause:
    """A ranked candidate cause emitted by a diagnostic method."""

    change_id: str
    score: float
    rationale: str


@dataclass(frozen=True)
class RegressionCase:
    """One observed regression where baseline succeeds and candidate fails."""

    case_id: str
    prompt: str
    expected: str
    baseline_label: str
    candidate_label: str | None


def rank_candidates(manifest: DiagnosticManifest) -> list[CandidateCause]:
    """Return a neutral interface baseline over ground-truth-free lineage."""

    if not isinstance(manifest, DiagnosticManifest):
        raise TypeError("diagnostic methods require a DiagnosticManifest")

    return [
        CandidateCause(change_id=change.change_id, score=0.0, rationale="unscored baseline")
        for change in manifest.changes
    ]


def rank_candidates_random(manifest: DiagnosticManifest, *, seed: int = 42) -> list[CandidateCause]:
    """Return a deterministic random-ranking lower baseline."""

    if not isinstance(manifest, DiagnosticManifest):
        raise TypeError("diagnostic methods require a DiagnosticManifest")

    change_ids = [change.change_id for change in manifest.changes]
    random.Random(seed).shuffle(change_ids)
    total = len(change_ids)
    return [
        CandidateCause(
            change_id=change_id,
            score=float(total - rank),
            rationale=f"seeded random baseline (seed={seed})",
        )
        for rank, change_id in enumerate(change_ids, start=1)
    ]


def load_observed_regressions(
    baseline_generations: str | Path,
    candidate_generations: str | Path,
) -> tuple[RegressionCase, ...]:
    """Load cases where the baseline is correct and the candidate is not."""

    baseline = _load_generation_records(baseline_generations)
    candidate = _load_generation_records(candidate_generations)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate generation case IDs must match exactly")

    regressions: list[RegressionCase] = []
    for case_id, baseline_record in baseline.items():
        candidate_record = candidate[case_id]
        if baseline_record["prompt"] != candidate_record["prompt"]:
            raise ValueError(f"prompt mismatch for case {case_id}")
        if baseline_record["expected"] != candidate_record["expected"]:
            raise ValueError(f"expected-label mismatch for case {case_id}")

        expected = str(baseline_record["expected"])
        baseline_label = baseline_record.get("parsed_label")
        candidate_label = candidate_record.get("parsed_label")
        if baseline_label == expected and candidate_label != expected:
            regressions.append(
                RegressionCase(
                    case_id=case_id,
                    prompt=str(baseline_record["prompt"]),
                    expected=expected,
                    baseline_label=str(baseline_label),
                    candidate_label=(str(candidate_label) if candidate_label is not None else None),
                )
            )

    if not regressions:
        raise ValueError("no observed baseline-correct/candidate-wrong regressions found")
    return tuple(regressions)


def rank_candidates_lexical_overlap(
    manifest: DiagnosticManifest,
    *,
    prepared_root: str | Path,
    regressions: tuple[RegressionCase, ...],
) -> list[CandidateCause]:
    """Rank changed data shards by prompt-token overlap with regressed cases.

    This intentionally simple baseline uses only debugger-visible lineage and
    observed regression prompts. It does not inspect benchmark ground truth or
    model internals.
    """

    if not isinstance(manifest, DiagnosticManifest):
        raise TypeError("diagnostic methods require a DiagnosticManifest")
    if not regressions:
        raise ValueError("at least one regression case is required")

    root = Path(prepared_root)
    ranked: list[CandidateCause] = []
    for change in manifest.changes:
        prompts = _load_change_prompts(change, root)
        score = sum(
            max(_jaccard_tokens(regression.prompt, prompt) for prompt in prompts)
            for regression in regressions
        ) / len(regressions)
        ranked.append(
            CandidateCause(
                change_id=change.change_id,
                score=score,
                rationale=(
                    "mean best Jaccard token overlap between observed regression prompts "
                    "and changed-shard prompts"
                ),
            )
        )

    return sorted(ranked, key=lambda candidate: (-candidate.score, candidate.change_id))


def rank_candidates_changed_lexical_overlap(
    manifest: DiagnosticManifest,
    *,
    prepared_root: str | Path,
    regressions: tuple[RegressionCase, ...],
) -> list[CandidateCause]:
    """Rank shards by overlap using only records that changed before to after.

    This baseline is still model-free. It uses debugger-visible before/after
    lineage to discard unchanged filler, then applies the same mean-best
    Jaccard prompt overlap used by the coarse artifact-level baseline.
    """

    if not isinstance(manifest, DiagnosticManifest):
        raise TypeError("diagnostic methods require a DiagnosticManifest")
    if not regressions:
        raise ValueError("at least one regression case is required")

    root = Path(prepared_root)
    ranked: list[CandidateCause] = []
    for change in manifest.changes:
        prompts = _load_changed_prompts(change, root)
        score = sum(
            max(_jaccard_tokens(regression.prompt, prompt) for prompt in prompts)
            for regression in regressions
        ) / len(regressions)
        ranked.append(
            CandidateCause(
                change_id=change.change_id,
                score=score,
                rationale=(
                    "mean best Jaccard token overlap between observed regression prompts "
                    "and prompts of before/after records that changed"
                ),
            )
        )

    return sorted(ranked, key=lambda candidate: (-candidate.score, candidate.change_id))


def dump_ranking(
    ranking: list[CandidateCause],
    path: str | Path,
    *,
    experiment_id: str,
    method: str,
    regression_case_ids: tuple[str, ...] = (),
) -> None:
    """Write a stable diagnostic ranking artifact without benchmark truth."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "method": method,
        "regression_case_ids": list(regression_case_ids),
        "ranking": [
            {
                "rank": rank,
                "change_id": candidate.change_id,
                "score": candidate.score,
                "rationale": candidate.rationale,
            }
            for rank, candidate in enumerate(ranking, start=1)
        ],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def score_ranking(ranking_path: str | Path, hidden_root_cause_id: str) -> dict[str, object]:
    """Score a frozen ranking against benchmark-owned hidden ground truth."""

    payload = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("ranking artifact must contain a non-empty ranking list")

    root_rank = next(
        (
            index
            for index, candidate in enumerate(ranking, start=1)
            if candidate.get("change_id") == hidden_root_cause_id
        ),
        None,
    )
    if root_rank is None:
        raise ValueError("hidden root cause is absent from the submitted ranking")

    candidate_count = len(ranking)
    top_k = min(3, candidate_count)
    root_entry = ranking[root_rank - 1]
    root_score = root_entry.get("score")
    if not isinstance(root_score, (int, float)):
        raise ValueError("ranking entries must contain numeric scores")

    numeric_scores: list[float] = []
    for candidate in ranking:
        score = candidate.get("score")
        if not isinstance(score, (int, float)):
            raise ValueError("ranking entries must contain numeric scores")
        numeric_scores.append(float(score))

    tie_tolerance = 1e-12
    strictly_higher = sum(score > float(root_score) + tie_tolerance for score in numeric_scores)
    tied = sum(abs(score - float(root_score)) <= tie_tolerance for score in numeric_scores)
    best_tied_rank = strictly_higher + 1
    worst_tied_rank = strictly_higher + tied

    average_tied_rank = (best_tied_rank + worst_tied_rank) / 2.0
    uniquely_top_1 = best_tied_rank == 1 and worst_tied_rank == 1
    top_3_guaranteed = worst_tied_rank <= top_k

    return {
        "hidden_root_cause_id": hidden_root_cause_id,
        "candidate_count": candidate_count,
        "root_cause_rank": root_rank,
        "top_1_correct": uniquely_top_1,
        "top_3_recall": top_3_guaranteed,
        "reciprocal_rank": 1.0 / average_tied_rank,
        "tie_aware": {
            "score_tolerance": tie_tolerance,
            "root_cause_tie_size": tied,
            "best_tied_rank": best_tied_rank,
            "worst_tied_rank": worst_tied_rank,
            "average_tied_rank": average_tied_rank,
            "uniquely_top_1": uniquely_top_1,
            "top_3_guaranteed": top_3_guaranteed,
        },
        "chance_reference": {
            "top_1_accuracy": 1.0 / candidate_count,
            "top_3_recall": top_k / candidate_count,
            "expected_reciprocal_rank": sum(1.0 / rank for rank in range(1, candidate_count + 1))
            / candidate_count,
            "permutations": math.factorial(candidate_count),
        },
    }


def _load_generation_records(path: str | Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                case_id = str(record["case_id"])
                record["prompt"]
                record["expected"]
            except KeyError as exc:
                raise ValueError(
                    f"missing generation field {exc.args[0]!r} at line {line_number}"
                ) from exc
            if case_id in records:
                raise ValueError(f"duplicate generation case_id: {case_id}")
            records[case_id] = record
    if not records:
        raise ValueError("generation file must contain at least one record")
    return records


def _load_change_prompts(change: ArtifactChange, prepared_root: Path) -> tuple[str, ...]:
    if change.kind != "dataset_shard":
        raise ValueError(
            f"lexical-overlap baseline supports dataset_shard only, got {change.kind!r}"
        )
    relative_path = change.metadata.get("after_path")
    if not isinstance(relative_path, str):
        raise ValueError(f"change {change.change_id} is missing string metadata.after_path")

    path = prepared_root / relative_path
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"change {change.change_id} has no prompt at {path}:{line_number}")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"change {change.change_id} contains no prompts")
    return tuple(prompts)


def _load_changed_prompts(change: ArtifactChange, prepared_root: Path) -> tuple[str, ...]:
    if change.kind != "dataset_shard":
        raise ValueError(
            f"changed-record lexical baseline supports dataset_shard only, got {change.kind!r}"
        )

    before_records = _load_change_records(change, prepared_root, path_key="before_path")
    after_records = _load_change_records(change, prepared_root, path_key="after_path")
    if before_records.keys() != after_records.keys():
        raise ValueError(f"change {change.change_id} before/after example IDs must match exactly")

    prompts: list[str] = []
    for example_id, before in before_records.items():
        after = after_records[example_id]
        if before == after:
            continue
        prompt = after.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"change {change.change_id} changed record {example_id} has no prompt")
        prompts.append(prompt)

    if not prompts:
        raise ValueError(f"change {change.change_id} contains no changed records")
    return tuple(prompts)


def _load_change_records(
    change: ArtifactChange,
    prepared_root: Path,
    *,
    path_key: str,
) -> dict[str, dict[str, object]]:
    relative_path = change.metadata.get(path_key)
    if not isinstance(relative_path, str):
        raise ValueError(f"change {change.change_id} is missing string metadata.{path_key}")

    path = prepared_root / relative_path
    records: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = record.get("example_id")
            if not isinstance(example_id, str):
                raise ValueError(
                    f"change {change.change_id} has no example_id at {path}:{line_number}"
                )
            if example_id in records:
                raise ValueError(f"change {change.change_id} has duplicate example_id {example_id}")
            records[example_id] = record
    if not records:
        raise ValueError(f"change {change.change_id} contains no records at {path}")
    return records


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard_tokens(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0
