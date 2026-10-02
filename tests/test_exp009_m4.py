from __future__ import annotations

import pytest

from model_forensics.exp009_m4 import (
    M4CheckpointRecord,
    ascii_tokens,
    deterministic_random_candidate_scores,
    lexical_overlap_candidate_scores,
    mean_trajectory_candidate_scores,
    target_label_overlap_candidate_scores,
    token_set_jaccard,
    tracin_slot_suspiciousness,
    validate_m4_checkpoint_records,
)
from model_forensics.exp009_release import Exp009ReleaseSlot


def _slot(slot_id: str, *, text: str, label: str) -> Exp009ReleaseSlot:
    return Exp009ReleaseSlot(
        slot_id=slot_id,
        source_content_id=f"source-{slot_id}",
        model_content_id=f"model-{slot_id}",
        text=text,
        label=label,
    )


def test_b0_random_reference_matches_frozen_sha256_rule() -> None:
    scores = deterministic_random_candidate_scores(
        "world_1",
        ["candidate_a", "candidate_b"],
    )
    assert scores == {
        "candidate_a": 2531598257586675334,
        "candidate_b": 14078825073553150102,
    }


def test_b1_counts_current_composite_target_labels_over_changed_slots() -> None:
    release = (
        _slot("s1", text="alpha", label="target_a"),
        _slot("s2", text="beta", label="protected"),
        _slot("s3", text="gamma", label="target_b"),
    )
    candidates = {"candidate_a": ("s1", "s2"), "candidate_b": ("s3",)}

    scores = target_label_overlap_candidate_scores(
        release,
        candidates,
        target_labels=("target_a", "target_b"),
    )

    assert scores == {"candidate_a": 1.0, "candidate_b": 1.0}


def test_b1_rejects_candidate_slot_missing_from_composite_release() -> None:
    release = (_slot("s1", text="alpha", label="target_a"),)
    with pytest.raises(ValueError, match="unknown composite slots"):
        target_label_overlap_candidate_scores(
            release,
            {"candidate_a": ("missing",)},
            target_labels=("target_a", "target_b"),
        )


def test_b2_uses_frozen_ascii_tokens_and_maximum_target_jaccard() -> None:
    release = (
        _slot("s1", text="Card payment reversed!", label="x"),
        _slot("s2", text="Cash withdrawal fee", label="y"),
    )
    candidates = {"candidate_a": ("s1",), "candidate_b": ("s2",)}

    scores = lexical_overlap_candidate_scores(
        release,
        candidates,
        target_slice_texts=("card payment failed", "cash withdrawal"),
    )

    assert scores["candidate_a"] == pytest.approx(0.5)
    assert scores["candidate_b"] == pytest.approx(2.0 / 3.0)


def test_b2_empty_token_sets_have_zero_similarity() -> None:
    assert ascii_tokens("!!!") == frozenset()
    assert token_set_jaccard(frozenset(), frozenset()) == 0.0


def test_trajectory_aggregation_is_exact_arithmetic_mean_over_0_1_2() -> None:
    scores = mean_trajectory_candidate_scores(
        {
            0: {"candidate_a": 1.0, "candidate_b": 6.0},
            1: {"candidate_a": 2.0, "candidate_b": 3.0},
            2: {"candidate_a": 3.0, "candidate_b": 0.0},
        }
    )
    assert scores == {"candidate_a": 2.0, "candidate_b": 3.0}


def test_trajectory_aggregation_rejects_missing_frozen_trajectory() -> None:
    with pytest.raises(ValueError, match="trajectory set mismatch"):
        mean_trajectory_candidate_scores(
            {
                0: {"candidate_a": 1.0},
                1: {"candidate_a": 2.0},
            }
        )


def test_tracin_sign_conversion_is_negative_native_influence() -> None:
    assert tracin_slot_suspiciousness(2.5) == -2.5
    assert tracin_slot_suspiciousness(-1.25) == 1.25


def test_checkpoint_contract_requires_epochs_1_to_7_and_producing_learning_rate() -> None:
    rows = tuple(
        M4CheckpointRecord(
            epoch=epoch,
            model_state_sha256=f"{epoch:x}" * 64,
            optimizer_step_count=epoch * 100,
            producing_learning_rate=2e-5 / epoch,
        )
        for epoch in range(1, 8)
    )
    assert validate_m4_checkpoint_records(rows) == rows

    wrong_final_lr = (*rows[:-1], M4CheckpointRecord(7, "7" * 64, 700, 0.0))
    with pytest.raises(ValueError, match="finite and positive"):
        validate_m4_checkpoint_records(wrong_final_lr)
