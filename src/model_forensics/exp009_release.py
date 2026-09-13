from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace

from model_forensics.exp009_data import Banking77Record, content_id_for_record
from model_forensics.exp009_trajectory import build_stable_training_slots

ROOT_SELECTION_NAMESPACE = "exp009-pilot-symmetric-label-swap-v1"


@dataclass(frozen=True)
class Exp009ReleaseSlot:
    """One versioned model-facing record anchored to a stable training slot."""

    slot_id: str
    source_content_id: str
    model_content_id: str
    text: str
    label: str


def build_clean_release_slots(
    records: tuple[Banking77Record, ...],
) -> tuple[Exp009ReleaseSlot, ...]:
    """Build the clean release while preserving the trajectory slot assignment."""

    slots = build_stable_training_slots(records)
    release = tuple(
        Exp009ReleaseSlot(
            slot_id=slot.slot_id,
            source_content_id=slot.content_id,
            model_content_id=slot.content_id,
            text=slot.text,
            label=slot.label,
        )
        for slot in slots
    )
    _validate_release_slots(release, require_unique_model_content_ids=True)
    return release


def _validate_release_slots(
    slots: tuple[Exp009ReleaseSlot, ...],
    *,
    require_unique_model_content_ids: bool,
) -> None:
    if not slots:
        raise ValueError("at least one Exp009 release slot is required")

    slot_ids = [slot.slot_id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("Exp009 release slot IDs must be unique")

    source_ids = [slot.source_content_id for slot in slots]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Exp009 release source content IDs must be unique")

    if require_unique_model_content_ids:
        model_ids = [slot.model_content_id for slot in slots]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Exp009 release model-facing content IDs must be unique")

    for slot in slots:
        expected = content_id_for_record(label=slot.label, text=slot.text)
        if slot.model_content_id != expected:
            raise ValueError(
                "Exp009 release model_content_id does not match current label/text: "
                f"slot={slot.slot_id}"
            )


def release_sha256(slots: tuple[Exp009ReleaseSlot, ...]) -> str:
    """Hash one release without serializing raw training text."""

    _validate_release_slots(slots, require_unique_model_content_ids=False)
    payload = [
        {
            "slot_id": slot.slot_id,
            "source_content_id": slot.source_content_id,
            "model_content_id": slot.model_content_id,
            "label": slot.label,
        }
        for slot in sorted(slots, key=lambda item: item.slot_id)
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _selection_key(
    slot: Exp009ReleaseSlot,
    *,
    label_a: str,
    label_b: str,
) -> bytes:
    pair = tuple(sorted((label_a, label_b)))
    payload = (
        f"{ROOT_SELECTION_NAMESPACE}|pair={pair[0]}|{pair[1]}|"
        f"source-label={slot.label}|slot={slot.slot_id}|source={slot.source_content_id}"
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def select_symmetric_swap_slot_ids(
    clean_slots: tuple[Exp009ReleaseSlot, ...],
    *,
    label_a: str,
    label_b: str,
    per_direction: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select deterministic nested A and B slot sets for a symmetric label fault."""

    _validate_release_slots(clean_slots, require_unique_model_content_ids=True)
    if label_a == label_b:
        raise ValueError("symmetric swap labels must differ")
    if per_direction <= 0:
        raise ValueError("per_direction must be positive")

    a_slots = [slot for slot in clean_slots if slot.label == label_a]
    b_slots = [slot for slot in clean_slots if slot.label == label_b]
    if len(a_slots) < per_direction or len(b_slots) < per_direction:
        raise ValueError(
            "symmetric swap dose exceeds available target slots: "
            f"A={len(a_slots)} B={len(b_slots)} requested={per_direction}"
        )

    a_ranked = sorted(
        a_slots,
        key=lambda slot: _selection_key(slot, label_a=label_a, label_b=label_b),
    )
    b_ranked = sorted(
        b_slots,
        key=lambda slot: _selection_key(slot, label_a=label_a, label_b=label_b),
    )
    return (
        tuple(slot.slot_id for slot in a_ranked[:per_direction]),
        tuple(slot.slot_id for slot in b_ranked[:per_direction]),
    )


def build_symmetric_label_swap_candidate(
    clean_slots: tuple[Exp009ReleaseSlot, ...],
    *,
    label_a: str,
    label_b: str,
    per_direction: int,
) -> tuple[Exp009ReleaseSlot, ...]:
    """Relabel deterministic target slots while leaving slot/text identity fixed."""

    a_ids, b_ids = select_symmetric_swap_slot_ids(
        clean_slots,
        label_a=label_a,
        label_b=label_b,
        per_direction=per_direction,
    )
    a_selected = set(a_ids)
    b_selected = set(b_ids)

    candidate: list[Exp009ReleaseSlot] = []
    for slot in clean_slots:
        if slot.slot_id in a_selected:
            new_label = label_b
        elif slot.slot_id in b_selected:
            new_label = label_a
        else:
            candidate.append(slot)
            continue

        candidate.append(
            replace(
                slot,
                label=new_label,
                model_content_id=content_id_for_record(label=new_label, text=slot.text),
            )
        )

    result = tuple(candidate)
    _validate_release_slots(result, require_unique_model_content_ids=True)
    return result


def _aligned_release_maps(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
) -> tuple[dict[str, Exp009ReleaseSlot], dict[str, Exp009ReleaseSlot]]:
    _validate_release_slots(baseline, require_unique_model_content_ids=False)
    _validate_release_slots(candidate, require_unique_model_content_ids=False)
    baseline_by_id = {slot.slot_id: slot for slot in baseline}
    candidate_by_id = {slot.slot_id: slot for slot in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate releases must contain identical slot IDs")

    for slot_id in baseline_by_id:
        baseline_slot = baseline_by_id[slot_id]
        candidate_slot = candidate_by_id[slot_id]
        if baseline_slot.source_content_id != candidate_slot.source_content_id:
            raise ValueError(f"stable source identity changed for slot {slot_id}")

    return baseline_by_id, candidate_by_id


def changed_slot_ids(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
) -> tuple[str, ...]:
    """Return slots whose current model-facing record differs from baseline."""

    baseline_by_id, candidate_by_id = _aligned_release_maps(baseline, candidate)
    return tuple(
        slot_id
        for slot_id in sorted(baseline_by_id)
        if baseline_by_id[slot_id] != candidate_by_id[slot_id]
    )


def restore_release_slots(
    candidate: tuple[Exp009ReleaseSlot, ...],
    baseline: tuple[Exp009ReleaseSlot, ...],
    *,
    restore_slot_ids: tuple[str, ...],
) -> tuple[Exp009ReleaseSlot, ...]:
    """Restore selected candidate slots to their exact baseline records."""

    baseline_by_id, candidate_by_id = _aligned_release_maps(baseline, candidate)
    requested = set(restore_slot_ids)
    unknown = requested - set(baseline_by_id)
    if unknown:
        raise ValueError(f"cannot restore unknown slot IDs: {sorted(unknown)!r}")

    restored = tuple(
        baseline_by_id[slot_id] if slot_id in requested else candidate_by_id[slot_id]
        for slot_id in sorted(candidate_by_id)
    )
    _validate_release_slots(restored, require_unique_model_content_ids=False)
    return restored


def release_diff_manifest(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidate: tuple[Exp009ReleaseSlot, ...],
) -> dict[str, object]:
    """Return a text-free audit manifest for one versioned release transition."""

    baseline_by_id, candidate_by_id = _aligned_release_maps(baseline, candidate)
    changed = changed_slot_ids(baseline, candidate)
    transitions = Counter(
        f"{baseline_by_id[slot_id].label}->{candidate_by_id[slot_id].label}"
        for slot_id in changed
    )
    changed_ids_sha256 = hashlib.sha256("\n".join(changed).encode("utf-8")).hexdigest()
    return {
        "baseline_release_sha256": release_sha256(baseline),
        "candidate_release_sha256": release_sha256(candidate),
        "slot_count": len(baseline),
        "changed_slot_count": len(changed),
        "changed_slot_ids_sha256": changed_ids_sha256,
        "label_transitions": {key: transitions[key] for key in sorted(transitions)},
    }
