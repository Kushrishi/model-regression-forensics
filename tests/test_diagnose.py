import json
from pathlib import Path

import pytest

from model_forensics.diagnose import (
    RegressionCase,
    dump_ranking,
    load_observed_regressions,
    rank_candidates,
    rank_candidates_changed_lexical_overlap,
    rank_candidates_lexical_overlap,
    rank_candidates_random,
    rank_candidates_selected_role_overlap,
    score_ranking,
    selected_role_descriptor,
)
from model_forensics.lineage import ArtifactChange, DiagnosticManifest, LineageManifest
from model_forensics.task import (
    EXP002_SHARD_IDS,
    EXP003_SHARD_IDS,
    build_exp002_data,
    build_exp003_data,
    select_exp003_shard,
    select_shard,
    write_jsonl,
    write_sft_jsonl,
)


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


def test_exp002_equalizes_existing_artifact_lexical_overlap(tmp_path: Path) -> None:
    data = build_exp002_data(seed=42)
    changes = []
    for shard_id in EXP002_SHARD_IDS:
        after_path = tmp_path / "changes" / shard_id / "after.jsonl"
        write_jsonl(select_shard(data.candidate_train, shard_id), after_path)
        changes.append(
            ArtifactChange(
                change_id=shard_id,
                kind="dataset_shard",
                description="changed shard",
                metadata={"after_path": str(after_path.relative_to(tmp_path))},
            )
        )
    manifest = DiagnosticManifest(
        experiment_id="exp002",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=changes,
    )
    regressions = tuple(
        RegressionCase(
            case_id=example.example_id,
            prompt=example.prompt,
            expected=example.response,
            baseline_label=example.response,
            candidate_label=None,
        )
        for example in data.target_eval
    )

    ranking = rank_candidates_lexical_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )
    scores = [candidate.score for candidate in ranking]

    assert max(scores) - min(scores) <= 1e-12


def test_changed_lexical_overlap_ignores_unchanged_target_filler(tmp_path: Path) -> None:
    changes = []
    for change_id in ("shard_a", "shard_b"):
        changes.append(
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="changed shard",
                metadata={
                    "before_path": f"changes/{change_id}/before.jsonl",
                    "after_path": f"changes/{change_id}/after.jsonl",
                },
            )
        )
    manifest = DiagnosticManifest(
        experiment_id="exp002",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=changes,
    )

    _write_jsonl(
        tmp_path / "changes/shard_a/before.jsonl",
        [
            {
                "example_id": "a_target",
                "prompt": "shape=triangle size=large material=cedar",
                "response": "ACCEPT",
            },
            {
                "example_id": "a_changed",
                "prompt": "shape=circle size=small material=cedar",
                "response": "ACCEPT",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "changes/shard_a/after.jsonl",
        [
            {
                "example_id": "a_target",
                "prompt": "shape=triangle size=large material=cedar",
                "response": "ACCEPT",
            },
            {
                "example_id": "a_changed",
                "prompt": "shape=circle size=small material=cedar",
                "response": "REJECT",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "changes/shard_b/before.jsonl",
        [
            {
                "example_id": "b_target",
                "prompt": "shape=triangle size=large material=granite",
                "response": "ACCEPT",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "changes/shard_b/after.jsonl",
        [
            {
                "example_id": "b_target",
                "prompt": "shape=triangle size=large material=granite",
                "response": "REJECT",
            }
        ],
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

    ranked = rank_candidates_changed_lexical_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )

    assert [candidate.change_id for candidate in ranked] == ["shard_b", "shard_a"]
    assert ranked[0].score > ranked[1].score


def test_score_ranking_reports_score_ties_without_trusting_id_tiebreak(tmp_path: Path) -> None:
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(
        json.dumps(
            {
                "ranking": [
                    {"rank": 1, "change_id": "a", "score": 1.0},
                    {"rank": 2, "change_id": "b", "score": 1.0},
                    {"rank": 3, "change_id": "c", "score": 1.0},
                    {"rank": 4, "change_id": "d", "score": 0.5},
                ]
            }
        ),
        encoding="utf-8",
    )

    score = score_ranking(ranking_path, "b")

    assert score["root_cause_rank"] == 2
    assert score["top_1_correct"] is False
    assert score["top_3_recall"] is True
    assert score["reciprocal_rank"] == 0.5
    assert score["tie_aware"] == {
        "score_tolerance": 1e-12,
        "root_cause_tie_size": 3,
        "best_tied_rank": 1,
        "worst_tied_rank": 3,
        "average_tied_rank": 2.0,
        "uniquely_top_1": False,
        "top_3_guaranteed": True,
    }


def test_exp003_lexical_baselines_tie_on_role_binding_confounders(tmp_path: Path) -> None:
    data = build_exp003_data(seed=42)
    changes = []
    for change_id in EXP003_SHARD_IDS:
        before = select_exp003_shard(data.baseline_train, change_id)
        after = select_exp003_shard(data.candidate_train, change_id)
        before_path = tmp_path / f"changes/{change_id}/before.jsonl"
        after_path = tmp_path / f"changes/{change_id}/after.jsonl"
        write_sft_jsonl(before, before_path)
        write_sft_jsonl(after, after_path)
        changes.append(
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="changed shard",
                metadata={
                    "before_path": str(before_path.relative_to(tmp_path)),
                    "after_path": str(after_path.relative_to(tmp_path)),
                },
            )
        )

    manifest = DiagnosticManifest(
        experiment_id="exp003",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=changes,
    )
    regressions = tuple(
        RegressionCase(
            case_id=example.example_id,
            prompt=example.prompt,
            expected=example.response,
            baseline_label=example.response,
            candidate_label=None,
        )
        for example in data.target_eval
    )

    lexical = rank_candidates_lexical_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )
    changed_lexical = rank_candidates_changed_lexical_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )

    lexical_scores = [candidate.score for candidate in lexical]
    changed_scores = [candidate.score for candidate in changed_lexical]
    assert max(lexical_scores) - min(lexical_scores) <= 1e-12
    assert max(changed_scores) - min(changed_scores) <= 1e-12


def test_selected_role_descriptor_resolves_selected_public_slot() -> None:
    prompt = (
        "Explicit policy: shape=circle -> ACCEPT; shape=triangle -> ACCEPT; "
        "shape=square -> REJECT. "
        "Classify only the selected synthetic object as ACCEPT or REJECT. "
        "material=cedar; color=blue; selected_slot=slot_c. "
        "slot_a:shape=circle,size=small; "
        "slot_b:shape=square,size=large; "
        "slot_c:shape=triangle,size=large; "
        "slot_d:shape=triangle,size=small; "
        "slot_e:shape=square,size=small; "
        "slot_f:shape=circle,size=large. "
        "Reply with exactly one label."
    )

    assert selected_role_descriptor(prompt) == "shape=triangle,size=large"


def test_selected_role_descriptor_rejects_missing_selected_object() -> None:
    prompt = "selected_slot=slot_f. slot_a:shape=circle,size=small; slot_b:shape=square,size=large"

    with pytest.raises(ValueError, match="has no visible object descriptor"):
        selected_role_descriptor(prompt)


def test_selected_role_overlap_ranks_matching_changed_candidate_first(
    tmp_path: Path,
) -> None:
    changes = []

    for change_id in ("candidate_a", "candidate_b"):
        changes.append(
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="changed shard",
                metadata={
                    "before_path": f"changes/{change_id}/before.jsonl",
                    "after_path": f"changes/{change_id}/after.jsonl",
                },
            )
        )

    manifest = DiagnosticManifest(
        experiment_id="synthetic_selected_role_test",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=changes,
    )

    prompt_a = (
        "selected_slot=slot_b. slot_a:shape=triangle,size=large; slot_b:shape=circle,size=small"
    )
    prompt_b = (
        "selected_slot=slot_a. slot_a:shape=triangle,size=large; slot_b:shape=circle,size=small"
    )

    _write_jsonl(
        tmp_path / "changes/candidate_a/before.jsonl",
        [
            {
                "example_id": "a",
                "prompt": prompt_a,
                "response": "ACCEPT",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "changes/candidate_a/after.jsonl",
        [
            {
                "example_id": "a",
                "prompt": prompt_a,
                "response": "REJECT",
            }
        ],
    )

    _write_jsonl(
        tmp_path / "changes/candidate_b/before.jsonl",
        [
            {
                "example_id": "b",
                "prompt": prompt_b,
                "response": "ACCEPT",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "changes/candidate_b/after.jsonl",
        [
            {
                "example_id": "b",
                "prompt": prompt_b,
                "response": "REJECT",
            }
        ],
    )

    regressions = (
        RegressionCase(
            case_id="regression",
            prompt=(
                "selected_slot=slot_f. "
                "slot_a:shape=circle,size=small; "
                "slot_f:shape=triangle,size=large"
            ),
            expected="ACCEPT",
            baseline_label="ACCEPT",
            candidate_label="REJECT",
        ),
    )

    ranking = rank_candidates_selected_role_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )

    assert [candidate.change_id for candidate in ranking] == [
        "candidate_b",
        "candidate_a",
    ]
    assert ranking[0].score == 1.0
    assert ranking[1].score == 0.0


def test_selected_role_overlap_preserves_score_ties(tmp_path: Path) -> None:
    changes = []

    prompt = (
        "selected_slot=slot_a. slot_a:shape=triangle,size=large; slot_b:shape=circle,size=small"
    )

    for change_id in ("candidate_a", "candidate_b"):
        changes.append(
            ArtifactChange(
                change_id=change_id,
                kind="dataset_shard",
                description="changed shard",
                metadata={
                    "before_path": f"changes/{change_id}/before.jsonl",
                    "after_path": f"changes/{change_id}/after.jsonl",
                },
            )
        )

        _write_jsonl(
            tmp_path / f"changes/{change_id}/before.jsonl",
            [
                {
                    "example_id": change_id,
                    "prompt": prompt,
                    "response": "ACCEPT",
                }
            ],
        )
        _write_jsonl(
            tmp_path / f"changes/{change_id}/after.jsonl",
            [
                {
                    "example_id": change_id,
                    "prompt": prompt,
                    "response": "REJECT",
                }
            ],
        )

    manifest = DiagnosticManifest(
        experiment_id="synthetic_selected_role_tie",
        baseline_run_id="baseline",
        candidate_run_id="candidate",
        changes=changes,
    )

    regressions = (
        RegressionCase(
            case_id="regression",
            prompt=prompt,
            expected="ACCEPT",
            baseline_label="ACCEPT",
            candidate_label="REJECT",
        ),
    )

    ranking = rank_candidates_selected_role_overlap(
        manifest,
        prepared_root=tmp_path,
        regressions=regressions,
    )

    assert [candidate.score for candidate in ranking] == [1.0, 1.0]
