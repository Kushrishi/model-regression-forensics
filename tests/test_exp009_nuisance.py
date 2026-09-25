from __future__ import annotations

from collections import defaultdict

import pytest

from model_forensics.exp009_data import (
    Banking77Record,
    DevelopmentPartition,
    content_id_for_record,
)
from model_forensics.exp009_nuisance import (
    build_balanced_cross_intent_refresh,
    combine_disjoint_release_changes,
    duplicate_model_content_profile,
    nuisance_refresh_audit,
    rank_nuisance_pairs,
    rank_nuisance_pairs_v2,
    select_disjoint_nuisance_pairs,
)
from model_forensics.exp009_release import (
    build_clean_release_slots,
    changed_slot_ids,
    restore_release_slots,
)


def _record(label: str, index: int) -> Banking77Record:
    text = f"{label} example {index}"
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def _partition(labels: tuple[str, ...], *, train_count: int = 70, eval_count: int = 20):
    train = tuple(_record(label, index) for label in labels for index in range(train_count))
    evaluation = tuple(
        _record(label, 10_000 + index) for label in labels for index in range(eval_count)
    )
    return DevelopmentPartition(
        development_train=train,
        development_eval=evaluation,
        source_records=len(train) + len(evaluation),
        unique_records=len(train) + len(evaluation),
        duplicate_groups=0,
        duplicate_occurrences_beyond_first=0,
    )


def _prediction_rows(
    partition: DevelopmentPartition,
    pair_error_counts: dict[tuple[str, str], int],
) -> tuple[dict[str, str], ...]:
    by_label: dict[str, list[Banking77Record]] = defaultdict(list)
    for record in partition.development_eval:
        by_label[record.label].append(record)

    rows: list[dict[str, str]] = []
    for label in sorted(by_label):
        records = sorted(by_label[label], key=lambda record: record.content_id)
        replacements: dict[int, str] = {}
        offset = 0
        for (truth, prediction), count in sorted(pair_error_counts.items()):
            if truth != label:
                continue
            for index in range(offset, offset + count):
                replacements[index] = prediction
            offset += count

        for index, record in enumerate(records):
            rows.append(
                {
                    "content_id": record.content_id,
                    "true_label": label,
                    "predicted_label": replacements.get(index, label),
                }
            )
    return tuple(rows)


def test_nuisance_pair_selection_is_clean_only_ranked_and_disjoint() -> None:
    labels = (
        "root_a",
        "root_b",
        "alpha_card",
        "beta_card",
        "alpha_cash",
        "beta_cash",
        "alpha_fee",
        "beta_fee",
        "alpha_transfer",
        "beta_transfer",
    )
    partition = _partition(labels)
    recalls = {trajectory: {label: 0.95 for label in labels} for trajectory in range(3)}

    pair_errors = {
        ("alpha_card", "beta_card"): 3,
        ("beta_card", "alpha_card"): 3,
        ("alpha_cash", "beta_cash"): 2,
        ("beta_cash", "alpha_cash"): 2,
        ("alpha_fee", "beta_fee"): 2,
        ("beta_fee", "alpha_fee"): 1,
        ("alpha_transfer", "beta_transfer"): 1,
        ("beta_transfer", "alpha_transfer"): 1,
    }
    predictions = {trajectory: _prediction_rows(partition, pair_errors) for trajectory in range(3)}

    ranked = rank_nuisance_pairs(
        partition,
        per_label_recall_by_trajectory=recalls,
        prediction_rows_by_trajectory=predictions,
        excluded_labels=frozenset({"root_a", "root_b"}),
    )
    selected = select_disjoint_nuisance_pairs(ranked, count=4)

    assert len(selected) == 4
    assert (selected[0].label_a, selected[0].label_b) == ("alpha_card", "beta_card")
    selected_labels = [label for pair in selected for label in (pair.label_a, pair.label_b)]
    assert len(selected_labels) == len(set(selected_labels))
    assert not {"root_a", "root_b"} & set(selected_labels)


def test_nuisance_pair_selection_v2_allows_one_way_confusion() -> None:
    labels = (
        "root_a",
        "root_b",
        "alpha_card",
        "beta_card",
        "alpha_cash",
        "beta_cash",
        "alpha_fee",
        "beta_fee",
        "alpha_transfer",
        "beta_transfer",
    )
    partition = _partition(labels)
    recalls = {trajectory: {label: 0.95 for label in labels} for trajectory in range(3)}

    one_way_pair_errors = {
        ("alpha_card", "beta_card"): 4,
        ("alpha_cash", "beta_cash"): 3,
        ("alpha_fee", "beta_fee"): 2,
        ("alpha_transfer", "beta_transfer"): 1,
    }
    predictions = {
        trajectory: _prediction_rows(partition, one_way_pair_errors) for trajectory in range(3)
    }

    v1_ranked = rank_nuisance_pairs(
        partition,
        per_label_recall_by_trajectory=recalls,
        prediction_rows_by_trajectory=predictions,
        excluded_labels=frozenset({"root_a", "root_b"}),
    )
    assert v1_ranked == ()

    v2_ranked = rank_nuisance_pairs_v2(
        partition,
        per_label_recall_by_trajectory=recalls,
        prediction_rows_by_trajectory=predictions,
        excluded_labels=frozenset({"root_a", "root_b"}),
    )
    selected = select_disjoint_nuisance_pairs(v2_ranked, count=4)

    assert [(pair.label_a, pair.label_b) for pair in selected] == [
        ("alpha_card", "beta_card"),
        ("alpha_cash", "beta_cash"),
        ("alpha_fee", "beta_fee"),
        ("alpha_transfer", "beta_transfer"),
    ]
    assert all((pair.a_to_b_count > 0) != (pair.b_to_a_count > 0) for pair in selected)


def test_balanced_cross_intent_refresh_is_deterministic_and_label_correct() -> None:
    records = tuple(
        _record(label, index) for label in ("alpha_card", "beta_card") for index in range(70)
    )
    baseline = build_clean_release_slots(records)

    candidate, selection = build_balanced_cross_intent_refresh(
        baseline,
        nuisance_index=1,
        label_a="alpha_card",
        label_b="beta_card",
        per_direction=3,
    )
    repeated, repeated_selection = build_balanced_cross_intent_refresh(
        baseline,
        nuisance_index=1,
        label_a="alpha_card",
        label_b="beta_card",
        per_direction=3,
    )

    assert candidate == repeated
    assert selection == repeated_selection
    assert len(changed_slot_ids(baseline, candidate)) == 6

    audit = nuisance_refresh_audit(baseline, candidate, selection)
    assert audit["changed_slot_count"] == 6
    assert audit["text_change_count"] == 6
    assert audit["aggregate_label_count_delta"] == {}
    assert audit["recipient_donor_disjoint"] is True
    assert audit["duplicate_model_content_occurrences_beyond_first"] == 6
    assert audit["label_transitions"] == {
        "alpha_card->beta_card": 3,
        "beta_card->alpha_card": 3,
    }

    changed = changed_slot_ids(baseline, candidate)
    restored = restore_release_slots(candidate, baseline, restore_slot_ids=changed)
    assert restored == baseline


def test_disjoint_nuisance_changes_compose_and_overlap_is_rejected() -> None:
    labels = ("alpha_card", "beta_card", "alpha_cash", "beta_cash")
    records = tuple(_record(label, index) for label in labels for index in range(70))
    baseline = build_clean_release_slots(records)

    candidate_one, _ = build_balanced_cross_intent_refresh(
        baseline,
        nuisance_index=1,
        label_a="alpha_card",
        label_b="beta_card",
        per_direction=3,
    )
    candidate_two, _ = build_balanced_cross_intent_refresh(
        baseline,
        nuisance_index=2,
        label_a="alpha_cash",
        label_b="beta_cash",
        per_direction=3,
    )

    combined = combine_disjoint_release_changes(baseline, candidate_one, candidate_two)
    assert len(changed_slot_ids(baseline, combined)) == 12
    assert (
        duplicate_model_content_profile(combined)[
            "duplicate_model_content_occurrences_beyond_first"
        ]
        == 12
    )

    with pytest.raises(ValueError, match="overlap"):
        combine_disjoint_release_changes(baseline, candidate_one, candidate_one)
