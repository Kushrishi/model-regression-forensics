from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from model_forensics.exp009_classifier import (
    EXP009_PILOT_MODEL_NAME,
    EXP009_PILOT_MODEL_REVISION,
    Exp009ClassifierPilotConfig,
    _batch_indices,
    development_partition_manifest_text,
    development_partition_sha256,
    token_length_audit,
)
from model_forensics.exp009_data import Banking77Record, DevelopmentPartition, content_id_for_record


def _record(label: str, text: str) -> Banking77Record:
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def _partition() -> DevelopmentPartition:
    train = tuple(
        sorted(
            (_record("intent_a", "train a"), _record("intent_b", "train b")),
            key=lambda record: record.content_id,
        )
    )
    evaluation = tuple(
        sorted(
            (_record("intent_a", "eval a"), _record("intent_b", "eval b")),
            key=lambda record: record.content_id,
        )
    )
    return DevelopmentPartition(
        development_train=train,
        development_eval=evaluation,
        source_records=4,
        unique_records=4,
        duplicate_groups=0,
        duplicate_occurrences_beyond_first=0,
    )


def test_initial_pilot_model_is_exactly_pinned() -> None:
    config = Exp009ClassifierPilotConfig()

    assert config.model_name == "distilbert/distilbert-base-uncased"
    assert config.model_revision == "12040accade4e8a0f71eabdb258fecc2e7e948be"
    assert EXP009_PILOT_MODEL_NAME == config.model_name
    assert EXP009_PILOT_MODEL_REVISION == config.model_revision


def test_pilot_config_rejects_unpinned_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="pinned DistilBERT"):
        replace(Exp009ClassifierPilotConfig(), model_name="other/model").validate()

    with pytest.raises(ValueError, match="pinned candidate"):
        replace(Exp009ClassifierPilotConfig(), model_revision="main").validate()

    for field, value in (
        ("epochs", 0),
        ("batch_size", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("warmup_ratio", 1.0),
        ("max_length", 0),
        ("max_grad_norm", 0.0),
    ):
        with pytest.raises(ValueError):
            replace(Exp009ClassifierPilotConfig(), **{field: value}).validate()


def test_partition_manifest_is_text_free_deterministic_and_hashed() -> None:
    partition = _partition()
    text = development_partition_manifest_text(partition)
    rows = [json.loads(line) for line in text.splitlines()]

    assert len(rows) == 4
    assert all(set(row) == {"content_id", "label", "split"} for row in rows)
    assert "train a" not in text
    assert "eval b" not in text
    assert development_partition_sha256(partition) == hashlib.sha256(text.encode()).hexdigest()


def test_batch_indices_preserve_order_and_remainder() -> None:
    assert _batch_indices(list(range(10)), 4) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]


class _DummyTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}


def test_token_length_audit_reports_distribution_without_truncation() -> None:
    audit = token_length_audit(
        _DummyTokenizer(),
        ["one", "one two", "one two three", "one two three four"],
    )

    assert audit["examples"] == 4
    assert audit["max"] == 6
    assert audit["p50"] == 5
    assert audit["p95"] == 6
    assert audit["p99"] == 6
