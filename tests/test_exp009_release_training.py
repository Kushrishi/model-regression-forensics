from __future__ import annotations

from dataclasses import replace

import pytest

from model_forensics.exp009_data import (
    Banking77Record,
    DevelopmentPartition,
    content_id_for_record,
)
from model_forensics.exp009_release import build_clean_release_slots
from model_forensics.exp009_release_training import (
    _collect_eval_logits,
    behavior_slice_metrics,
    validate_release_alignment,
)


def _record(label: str, text: str) -> Banking77Record:
    return Banking77Record(
        text=text,
        label=label,
        content_id=content_id_for_record(label=label, text=text),
    )


def _partition() -> DevelopmentPartition:
    labels = tuple(f"intent_{index:02d}" for index in range(77))
    train = tuple(
        sorted(
            (_record(label, f"train {label}") for label in labels),
            key=lambda record: record.content_id,
        )
    )
    evaluation = tuple(
        sorted(
            (_record(label, f"eval {label}") for label in labels),
            key=lambda record: record.content_id,
        )
    )
    return DevelopmentPartition(
        development_train=train,
        development_eval=evaluation,
        source_records=154,
        unique_records=154,
        duplicate_groups=0,
        duplicate_occurrences_beyond_first=0,
    )


def test_release_alignment_canonicalizes_input_order() -> None:
    partition = _partition()
    clean = build_clean_release_slots(partition.development_train)

    aligned = validate_release_alignment(partition, tuple(reversed(clean)))

    assert aligned == clean


def test_release_alignment_requires_exact_slot_membership() -> None:
    partition = _partition()
    clean = build_clean_release_slots(partition.development_train)

    with pytest.raises(ValueError, match="exactly the frozen clean slot IDs"):
        validate_release_alignment(partition, clean[:-1])


def test_release_alignment_rejects_changed_source_identity() -> None:
    partition = _partition()
    clean = build_clean_release_slots(partition.development_train)
    changed = (replace(clean[0], source_content_id="different"), *clean[1:])

    with pytest.raises(ValueError, match="changed source identity"):
        validate_release_alignment(partition, changed)


def test_release_alignment_rejects_unknown_label() -> None:
    partition = _partition()
    clean = build_clean_release_slots(partition.development_train)
    changed = (replace(clean[0], label="unknown_intent"), *clean[1:])

    with pytest.raises(ValueError, match="outside frozen vocabulary"):
        validate_release_alignment(partition, changed)


def test_behavior_slice_metrics_separates_target_and_protected() -> None:
    per_label = {
        "target_a": 0.8,
        "target_b": 0.6,
        "protected_a": 1.0,
        "protected_b": 0.5,
    }

    metrics = behavior_slice_metrics(
        per_label,
        target_labels=("target_a", "target_b"),
    )

    assert metrics["target_macro_recall"] == pytest.approx(0.7)
    assert metrics["protected_macro_recall"] == pytest.approx(0.75)
    assert metrics["protected_worst_intent_recall"] == pytest.approx(0.5)
    assert metrics["protected_worst_intents"] == ["protected_b"]


def test_behavior_slice_metrics_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError, match="must differ"):
        behavior_slice_metrics({"a": 1.0, "b": 1.0}, target_labels=("a", "a"))

    with pytest.raises(ValueError, match="must exist"):
        behavior_slice_metrics({"a": 1.0, "b": 1.0}, target_labels=("a", "missing"))


def test_collect_eval_logits_preserves_example_order() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class ToyModel(nn.Module):
        def forward(self, input_ids, attention_mask):
            del attention_mask
            logits = torch.stack(
                (
                    input_ids[:, 0].to(torch.float32),
                    input_ids[:, 1].to(torch.float32),
                ),
                dim=1,
            )
            return type("Output", (), {"logits": logits})()

    encoded = {
        "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
        "attention_mask": torch.ones((3, 2), dtype=torch.long),
    }

    rows = _collect_eval_logits(
        ToyModel(),
        encoded_eval=encoded,
        example_count=3,
        batch_size=2,
        device="cpu",
        torch=torch,
    )

    assert rows == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
