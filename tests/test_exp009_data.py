from __future__ import annotations

import json

import pytest

import model_forensics.exp009_data as exp009_data
from model_forensics.exp009_data import (
    BANKING77_SOURCE_REVISION,
    BANKING77_TRAIN_PATH,
    BANKING77_TRAIN_SHA256,
    BANKING77_TRAIN_URL,
    Banking77Record,
    build_development_partition,
    canonical_record_bytes,
    content_id_for_record,
    deduplicate_canonical_records,
    parse_banking77_train_csv,
    partition_summary,
    sha256_bytes,
)


def _record(label: str, text: str) -> Banking77Record:
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def test_banking77_source_is_pinned_to_train_only() -> None:
    assert BANKING77_SOURCE_REVISION == "57ec275d8078af65b7731c2a98be812d844a6d6b"
    assert BANKING77_TRAIN_PATH == "banking_data/train.csv"
    assert BANKING77_TRAIN_SHA256 == (
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
    )
    assert BANKING77_TRAIN_URL.endswith(f"/{BANKING77_SOURCE_REVISION}/{BANKING77_TRAIN_PATH}")
    assert "test.csv" not in BANKING77_TRAIN_URL


def test_canonical_record_identity_normalizes_outer_space_and_unicode() -> None:
    composed = canonical_record_bytes(label=" intent ", text=" café ")
    decomposed = canonical_record_bytes(label="intent", text="cafe\u0301")
    assert composed == decomposed


def test_parse_train_csv_checks_sha_shape_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"text,category\n hello ,intent_a\nworld,intent_b\n"
    monkeypatch.setattr(exp009_data, "BANKING77_TRAIN_SHA256", sha256_bytes(payload))
    monkeypatch.setattr(exp009_data, "BANKING77_EXPECTED_TRAIN_EXAMPLES", 2)
    monkeypatch.setattr(exp009_data, "BANKING77_EXPECTED_INTENTS", 2)

    records = parse_banking77_train_csv(payload)

    assert [(record.text, record.label) for record in records] == [
        ("hello", "intent_a"),
        ("world", "intent_b"),
    ]


def test_partition_is_stratified_hash_ordered_and_input_order_independent() -> None:
    records = tuple(
        _record(label, f"{label}-example-{index}")
        for label, count in (("intent_a", 9), ("intent_b", 11))
        for index in range(count)
    )

    forward = build_development_partition(records)
    reverse = build_development_partition(tuple(reversed(records)))

    assert [record.content_id for record in forward.development_train] == [
        record.content_id for record in reverse.development_train
    ]
    assert [record.content_id for record in forward.development_eval] == [
        record.content_id for record in reverse.development_eval
    ]

    eval_counts: dict[str, int] = {}
    for record in forward.development_eval:
        eval_counts[record.label] = eval_counts.get(record.label, 0) + 1

    assert eval_counts == {"intent_a": 2, "intent_b": 2}
    assert len(forward.development_train) == 16
    assert len(forward.development_eval) == 4
    assert forward.source_records == 20
    assert forward.unique_records == 20
    assert forward.duplicate_groups == 0
    assert forward.duplicate_occurrences_beyond_first == 0


def test_exact_canonical_duplicates_are_collapsed_before_partitioning() -> None:
    duplicate_a = _record("intent_a", " same example ")
    duplicate_b = _record("intent_a", "same example")
    other_records = tuple(_record("intent_a", f"other-{index}") for index in range(4))

    unique, duplicate_groups, duplicate_occurrences = deduplicate_canonical_records(
        (duplicate_a, *other_records, duplicate_b)
    )
    partition = build_development_partition((duplicate_a, *other_records, duplicate_b))

    assert len(unique) == 5
    assert duplicate_groups == 1
    assert duplicate_occurrences == 1
    assert partition.source_records == 6
    assert partition.unique_records == 5
    assert partition.duplicate_groups == 1
    assert partition.duplicate_occurrences_beyond_first == 1

    partition_ids = {
        record.content_id
        for record in (*partition.development_train, *partition.development_eval)
    }
    assert len(partition_ids) == 5
    assert duplicate_a.content_id in partition_ids


def test_deduplication_is_input_order_independent() -> None:
    records = (
        _record("intent_a", " same example "),
        _record("intent_a", "other"),
        _record("intent_a", "same example"),
    )

    forward = deduplicate_canonical_records(records)
    reverse = deduplicate_canonical_records(tuple(reversed(records)))

    assert forward == reverse


def test_partition_summary_contains_no_training_text() -> None:
    records = tuple(
        _record(label, f"PRIVATE-TEXT-{label}-{index}")
        for label in ("intent_a", "intent_b")
        for index in range(5)
    )
    partition = build_development_partition(records)

    payload = json.dumps(partition_summary(partition), sort_keys=True)

    assert "PRIVATE-TEXT" not in payload
    assert '"source_train_raw": 10' in payload
    assert '"source_train_unique": 10' in payload
    assert '"development_eval": 2' in payload
