from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from model_forensics.exp009_attribution import aggregate_candidate_suspiciousness
from model_forensics.exp009_release import Exp009ReleaseSlot

M4_RANDOM_NAMESPACE = "mrf-m4-random-v1"
M4_TRAJECTORIES = (0, 1, 2)
M4_CHECKPOINT_EPOCHS = tuple(range(1, 8))
ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class M4CheckpointRecord:
    """Frozen metadata required for one M4 TracIn checkpoint."""

    epoch: int
    model_state_sha256: str
    optimizer_step_count: int
    producing_learning_rate: float


def deterministic_random_candidate_scores(
    world_id: str,
    candidate_ids: Sequence[str],
) -> dict[str, int]:
    """Return the frozen B0 deterministic random-reference scores."""

    if not world_id:
        raise ValueError("world_id must be non-empty")
    ids = tuple(candidate_ids)
    if not ids:
        raise ValueError("at least one candidate_id is required")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_ids must be unique")

    output: dict[str, int] = {}
    for candidate_id in ids:
        if not candidate_id:
            raise ValueError("candidate_id must be non-empty")
        payload = f"{M4_RANDOM_NAMESPACE}|{world_id}|{candidate_id}".encode()
        digest = hashlib.sha256(payload).digest()
        output[candidate_id] = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return output


def _release_by_slot(release: Sequence[Exp009ReleaseSlot]) -> dict[str, Exp009ReleaseSlot]:
    slots = tuple(release)
    if not slots:
        raise ValueError("composite release must be non-empty")
    output = {slot.slot_id: slot for slot in slots}
    if len(output) != len(slots):
        raise ValueError("composite release contains duplicate slot IDs")
    return output


def target_label_overlap_candidate_scores(
    composite_release: Sequence[Exp009ReleaseSlot],
    candidate_changed_slots: Mapping[str, Sequence[str]],
    *,
    target_labels: tuple[str, str],
) -> dict[str, float]:
    """Compute the frozen B1 direct target-label-overlap candidate scores."""

    target_a, target_b = target_labels
    if target_a == target_b:
        raise ValueError("target_labels must be distinct")
    release_by_slot = _release_by_slot(composite_release)

    relevant_slots = {
        slot_id for slot_ids in candidate_changed_slots.values() for slot_id in slot_ids
    }
    missing = sorted(relevant_slots - set(release_by_slot))
    if missing:
        raise ValueError(f"candidate manifest references unknown composite slots: {missing}")

    target_set = {target_a, target_b}
    slot_scores = {
        slot_id: float(release_by_slot[slot_id].label in target_set) for slot_id in relevant_slots
    }
    return aggregate_candidate_suspiciousness(slot_scores, candidate_changed_slots)


def ascii_tokens(text: str) -> frozenset[str]:
    """Tokenize text using the prospectively frozen B2 rule."""

    return frozenset(ASCII_TOKEN_RE.findall(text.lower()))


def token_set_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Return set Jaccard similarity, defining empty-versus-empty as zero."""

    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def lexical_overlap_candidate_scores(
    composite_release: Sequence[Exp009ReleaseSlot],
    candidate_changed_slots: Mapping[str, Sequence[str]],
    *,
    target_slice_texts: Sequence[str],
) -> dict[str, float]:
    """Compute the frozen B2 maximum-target-slice Jaccard candidate scores."""

    release_by_slot = _release_by_slot(composite_release)
    target_texts = tuple(target_slice_texts)
    if not target_texts:
        raise ValueError("target_slice_texts must be non-empty")
    target_tokens = tuple(ascii_tokens(text) for text in target_texts)

    relevant_slots = {
        slot_id for slot_ids in candidate_changed_slots.values() for slot_id in slot_ids
    }
    missing = sorted(relevant_slots - set(release_by_slot))
    if missing:
        raise ValueError(f"candidate manifest references unknown composite slots: {missing}")

    slot_scores: dict[str, float] = {}
    for slot_id in relevant_slots:
        slot_tokens = ascii_tokens(release_by_slot[slot_id].text)
        slot_scores[slot_id] = max(
            token_set_jaccard(slot_tokens, target_example) for target_example in target_tokens
        )
    return aggregate_candidate_suspiciousness(slot_scores, candidate_changed_slots)


def mean_trajectory_candidate_scores(
    scores_by_trajectory: Mapping[int, Mapping[str, float]],
    *,
    expected_trajectories: tuple[int, ...] = M4_TRAJECTORIES,
) -> dict[str, float]:
    """Average candidate scores across the frozen M4 development trajectories."""

    expected = tuple(expected_trajectories)
    if tuple(sorted(scores_by_trajectory)) != tuple(sorted(expected)):
        raise ValueError(
            "trajectory set mismatch: "
            f"expected={sorted(expected)} observed={sorted(scores_by_trajectory)}"
        )
    if not expected:
        raise ValueError("expected_trajectories must be non-empty")

    candidate_sets = [set(scores_by_trajectory[trajectory]) for trajectory in expected]
    if not candidate_sets[0]:
        raise ValueError("candidate score mappings must be non-empty")
    if any(candidate_set != candidate_sets[0] for candidate_set in candidate_sets[1:]):
        raise ValueError("candidate sets must match across trajectories")

    output: dict[str, float] = {}
    for candidate_id in sorted(candidate_sets[0]):
        values = [float(scores_by_trajectory[t][candidate_id]) for t in expected]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite trajectory score for {candidate_id!r}")
        output[candidate_id] = math.fsum(values) / len(values)
    return output


def tracin_slot_suspiciousness(native_aggregate_influence: float) -> float:
    """Apply the prospectively frozen B4 sign conversion."""

    influence = float(native_aggregate_influence)
    if not math.isfinite(influence):
        raise ValueError("native TracIn influence must be finite")
    return -influence


def validate_m4_checkpoint_records(
    records: Sequence[M4CheckpointRecord],
) -> tuple[M4CheckpointRecord, ...]:
    """Validate the frozen seven-checkpoint TracIn metadata contract."""

    rows = tuple(records)
    if tuple(row.epoch for row in rows) != M4_CHECKPOINT_EPOCHS:
        raise ValueError(
            "checkpoint epochs must be exactly "
            f"{M4_CHECKPOINT_EPOCHS}; observed={tuple(row.epoch for row in rows)}"
        )

    previous_steps = -1
    for row in rows:
        digest = row.model_state_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"epoch {row.epoch}: model_state_sha256 must be a 64-digit hex digest")
        if row.optimizer_step_count <= previous_steps:
            raise ValueError("optimizer_step_count must increase strictly across checkpoints")
        previous_steps = row.optimizer_step_count

        learning_rate = float(row.producing_learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                f"epoch {row.epoch}: producing_learning_rate must be finite and positive"
            )

    return rows
