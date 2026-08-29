import json
from pathlib import Path

import pytest

from model_forensics.diagnose import (
    RegressionCase,
    dump_ranking,
    load_observed_regressions,
    rank_candidates,
    rank_candidates_lexical_overlap,
    rank_candidates_random,
    score_ranking,
)
from model_forensics.lineage import ArtifactChange, DiagnosticManifest, LineageManifest


def _manifest(*change_ids: str) -> DiagnosticManifest:
    return DiagnosticManifest(
        experiment_id="exp001",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=[
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="changed shard",
                metadata={"after_path": f"changes/{change_id}/after.jsonl"},
            )
            for change_id in change_ids
        ],
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_rank_candidates_accepts_only_diagnostic_manifest() -> None:
    change = ArtifactChange(
        change_id="shard_a",
        kind="dataset_shard",
        description="changed shard",
    )
    benchmark = LineageManifest(
        experiment_id="exp000",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=[change],
        hidden_root_cause_id="shard_a",
    )

    ranked = rank_candidates(benchmark.redacted())

    assert [candidate.change_id for candidate in ranked] == ["shard_a"]

    with pytest.raises(TypeError, match="DiagnosticManifest"):
        rank_candidates(benchmark)  # type: ignore[arg-type]


def test_diagnostic_manifest_has_no_ground_truth_field() -> None:
    manifest = DiagnosticManifest(
        experiment_id="exp000",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=[],
    )

    assert "hidden_root_cause_id" not in manifest.model_dump()


def test_random_ranking_is_seeded_and_complete() -> None:
    manifest = _manifest("a", "b", "c", "d", "e")

    first = rank_candidates_random(manifest, seed=7)
    second = rank_candidates_random(manifest, seed=7)

    assert first == second
    assert {candidate.change_id for candidate in first} == {"a", "b", "c", "d", "e"}


def test_load_observed_regressions_selects_baseline_correct_candidate_wrong(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        baseline,
        [
            {
                "case_id": "regressed",
                "prompt": "shape=triangle size=large",
                "expected": "ACCEPT",
                "parsed_label": "ACCEPT",
            },
            {
                "case_id": "stable",
                "prompt": "shape=square size=small",
                "expected": "REJECT",
                "parsed_label": "REJECT",
            },
        ],
    )
    _write_jsonl(
        candidate,
        [
            {
                "case_id": "regressed",
                "prompt": "shape=triangle size=large",
                "expected": "ACCEPT",
                "parsed_label": "REJECT",
            },
            {
                "case_id": "stable",
                "prompt": "shape=square size=small",
                "expected": "REJECT",
                "parsed_label": "REJECT",
            },
        ],
    )

    regressions = load_observed_regressions(baseline, candidate)

    assert [case.case_id for case in regressions] == ["regressed"]


def test_lexical_overlap_ranks_matching_changed_shard_first(tmp_path: Path) -> None:
    manifest = _manifest("shard_a", "shard_b")
    _write_jsonl(
        tmp_path / "changes/shard_a/after.jsonl",
        [{"prompt": "shape=circle size=small material=cedar"}],
    )
    _write_jsonl(
        tmp_path / "changes/shard_b/after.jsonl",
        [{"prompt": "shape=triangle size=large material=granite"}],
    )
    regressions = (
        RegressionCase(
            case_id="target",
            prompt="shape=triangle size=large material=wool",
            expected="ACCEPT",
            baseline_label="ACCEPT",
            candidate_label="REJECT",
        ),
    )

    ranked = rank_candidates_lexical_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )

    assert [candidate.change_id for candidate in ranked] == ["shard_b", "shard_a"]
    assert ranked[0].score > ranked[1].score


def test_ranking_scoring_is_separate_from_diagnosis(tmp_path: Path) -> None:
    manifest = _manifest("a", "b", "c")
    ranking = rank_candidates_random(manifest, seed=1)
    path = tmp_path / "ranking.json"
    dump_ranking(ranking, path, experiment_id="exp001", method="random")

    hidden = ranking[1].change_id
    score = score_ranking(path, hidden)

    assert score["root_cause_rank"] == 2
    assert score["top_1_correct"] is False
    assert score["top_3_recall"] is True
    assert score["reciprocal_rank"] == 0.5
    assert score["chance_reference"]["top_1_accuracy"] == pytest.approx(1 / 3)
    assert score["chance_reference"]["permutations"] == 6
    assert "hidden_root_cause_id" not in json.loads(path.read_text(encoding="utf-8"))
