from __future__ import annotations

from model_forensics.exp009_data import Banking77Record, DevelopmentPartition, content_id_for_record
from model_forensics.exp009_matched_benchmark import (
    MATCHED_CANDIDATES_PER_WORLD,
    MATCHED_CHANGED_SLOT_COUNT,
    MATCHED_PER_DIRECTION,
    MatchedPairScore,
    MatchedWorldDefinition,
    build_matched_world,
    rank_matched_pairs,
    select_matched_worlds,
)
from model_forensics.exp009_release import build_clean_release_slots, changed_slot_ids


def _record(label: str, index: int) -> Banking77Record:
    text = f"{label} example {index}"
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def _pair(label_a: str, label_b: str, score: float) -> MatchedPairScore:
    return MatchedPairScore(
        label_a=label_a,
        label_b=label_b,
        mean_symmetric_confusion_rate=score,
        mutual_confusion_count=int(score * 1000),
        lexical_jaccard=0.5,
        minimum_clean_recall=0.95,
        train_count_a=80,
        train_count_b=80,
        eval_count_a=20,
        eval_count_b=20,
        a_to_b_count=1,
        b_to_a_count=1,
    )


def test_rank_matched_pairs_accepts_confusion_or_lexical_adjacency() -> None:
    labels = ("alpha", "beta", "card_one", "card_two")
    train = tuple(_record(label, i) for label in labels for i in range(70))
    eval_rows = tuple(_record(label, 1000 + i) for label in labels for i in range(20))
    partition = DevelopmentPartition(
        development_train=train,
        development_eval=eval_rows,
        source_records=len(train) + len(eval_rows),
        unique_records=len(train) + len(eval_rows),
        duplicate_groups=0,
        duplicate_occurrences_beyond_first=0,
    )
    recalls = {0: {label: 0.95 for label in labels}}
    predictions = {
        0: (
            {"true_label": "alpha", "predicted_label": "beta"},
            {"true_label": "beta", "predicted_label": "beta"},
            {"true_label": "card_one", "predicted_label": "card_one"},
            {"true_label": "card_two", "predicted_label": "card_two"},
        )
    }

    ranked = rank_matched_pairs(
        partition,
        per_label_recall_by_trajectory=recalls,
        prediction_rows_by_trajectory=predictions,
    )
    pairs = {(row.label_a, row.label_b) for row in ranked}

    assert ("alpha", "beta") in pairs
    assert ("card_one", "card_two") in pairs


def test_select_matched_worlds_is_globally_label_disjoint() -> None:
    ranked = tuple(
        _pair(f"intent_{2 * i:02d}", f"intent_{2 * i + 1:02d}", 1.0 - i / 100)
        for i in range(15)
    )

    worlds = select_matched_worlds(ranked)

    assert len(worlds) == 3
    observed_labels: set[str] = set()
    for world in worlds:
        assert len(world.pairs) == MATCHED_CANDIDATES_PER_WORLD
        assert 0 <= world.root_position < MATCHED_CANDIDATES_PER_WORLD
        labels = {
            label
            for pair in world.pairs
            for label in (pair.label_a, pair.label_b)
        }
        assert len(labels) == 10
        assert not labels & observed_labels
        observed_labels.update(labels)

    assert len(observed_labels) == 30


def test_build_matched_world_enforces_identical_candidate_structure() -> None:
    pairs = tuple(
        _pair(f"intent_{2 * i:02d}", f"intent_{2 * i + 1:02d}", 1.0 - i / 100)
        for i in range(MATCHED_CANDIDATES_PER_WORLD)
    )
    records = tuple(
        _record(label, index)
        for pair in pairs
        for label in (pair.label_a, pair.label_b)
        for index in range(MATCHED_PER_DIRECTION)
    )
    baseline = build_clean_release_slots(records)
    definition = MatchedWorldDefinition(
        world_index=0,
        pairs=pairs,
        root_position=2,
    )

    world = build_matched_world(baseline, definition)

    assert len(world.candidate_releases) == MATCHED_CANDIDATES_PER_WORLD
    assert len(changed_slot_ids(baseline, world.composite)) == (
        MATCHED_CANDIDATES_PER_WORLD * MATCHED_CHANGED_SLOT_COUNT
    )
    assert world.structural_audit["all_candidate_structures_identical"] is True
    assert world.structural_audit["candidate_structure"]["changed_slot_count"] == 66
    assert world.structural_audit["candidate_structure"]["text_change_count"] == 0
    assert world.structural_audit["candidate_structure"]["aggregate_label_count_delta"] == {}
    assert world.structural_audit["composite_order_independent"] is True
    assert (
        world.structural_audit["all_individual_candidate_restorations_exact_baseline"]
        is True
    )
    assert (
        world.structural_audit["all_restorations_leave_exactly_four_candidate_changes"]
        is True
    )

    diagnostic_ids = {
        row["candidate_id"] for row in world.diagnostic_manifest["candidates"]
    }
    truth_ids = {
        row["candidate_id"] for row in world.truth_manifest["truth"]
    }
    assert diagnostic_ids == truth_ids
    assert world.truth_manifest["root_position"] == 2


def test_matched_world_rejects_reused_intent_labels() -> None:
    pairs = (
        _pair("a", "b", 1.0),
        _pair("a", "c", 0.9),
        _pair("d", "e", 0.8),
        _pair("f", "g", 0.7),
        _pair("h", "i", 0.6),
    )
    records = tuple(
        _record(label, index)
        for label in "abcdefghi"
        for index in range(MATCHED_PER_DIRECTION)
    )
    baseline = build_clean_release_slots(records)

    try:
        build_matched_world(
            baseline,
            MatchedWorldDefinition(world_index=0, pairs=pairs, root_position=0),
        )
    except ValueError as exc:
        assert "intent labels must be disjoint" in str(exc)
    else:
        raise AssertionError("expected reused intent labels to be rejected")
