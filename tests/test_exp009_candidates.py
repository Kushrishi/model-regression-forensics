from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from model_forensics.exp009_candidates import (
    build_opaque_candidate_manifests,
    candidate_slots_from_manifest,
)
from model_forensics.exp009_data import content_id_for_record
from model_forensics.exp009_release import Exp009ReleaseSlot


def _slot(slot_id: str, label: str, text: str) -> Exp009ReleaseSlot:
    content_id = content_id_for_record(label=label, text=text)
    return Exp009ReleaseSlot(
        slot_id=slot_id,
        source_content_id=content_id,
        model_content_id=content_id,
        text=text,
        label=label,
    )


def _replace_label(
    baseline: tuple[Exp009ReleaseSlot, ...],
    *,
    slot_id: str,
    label: str,
) -> tuple[Exp009ReleaseSlot, ...]:
    output = []
    for slot in baseline:
        if slot.slot_id != slot_id:
            output.append(slot)
            continue
        output.append(
            replace(
                slot,
                label=label,
                model_content_id=content_id_for_record(label=label, text=slot.text),
            )
        )
    return tuple(output)


def _fixture():
    baseline = (
        _slot("slot_0000", "A", "alpha"),
        _slot("slot_0001", "B", "beta"),
        _slot("slot_0002", "C", "gamma"),
        _slot("slot_0003", "D", "delta"),
    )
    root = _replace_label(baseline, slot_id="slot_0000", label="B")
    nuisance = _replace_label(baseline, slot_id="slot_0002", label="D")
    return baseline, root, nuisance


def test_diagnostic_manifest_contains_no_semantic_truth_roles() -> None:
    baseline, root, nuisance = _fixture()

    diagnostic, truth = build_opaque_candidate_manifests(
        baseline,
        {"root": root, "nuisance_1": nuisance},
    )

    serialized = json.dumps(diagnostic, sort_keys=True)
    assert '"root"' not in serialized
    assert "nuisance_1" not in serialized

    truth_roles = {row["internal_role"] for row in truth["truth"]}
    assert truth_roles == {"root", "nuisance_1"}


def test_candidate_ids_are_deterministic_independent_of_role_mapping_order() -> None:
    baseline, root, nuisance = _fixture()

    first, _ = build_opaque_candidate_manifests(
        baseline,
        {"root": root, "nuisance_1": nuisance},
    )
    second, _ = build_opaque_candidate_manifests(
        baseline,
        {"nuisance_1": nuisance, "root": root},
    )

    assert first == second


def test_candidate_manifest_round_trips_to_changed_slot_mapping() -> None:
    baseline, root, nuisance = _fixture()
    diagnostic, _ = build_opaque_candidate_manifests(
        baseline,
        {"root": root, "nuisance_1": nuisance},
    )

    candidates = candidate_slots_from_manifest(diagnostic)

    assert len(candidates) == 2
    assert sorted(candidates.values()) == [("slot_0000",), ("slot_0002",)]


def test_candidate_manifest_rejects_tampered_hash() -> None:
    baseline, root, nuisance = _fixture()
    diagnostic, _ = build_opaque_candidate_manifests(
        baseline,
        {"root": root, "nuisance_1": nuisance},
    )
    tampered = copy.deepcopy(diagnostic)
    tampered["candidates"][0]["changed_slot_ids_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash mismatch"):
        candidate_slots_from_manifest(tampered)


def test_candidate_manifest_rejects_semantically_renamed_candidate_id() -> None:
    baseline, root, nuisance = _fixture()
    diagnostic, _ = build_opaque_candidate_manifests(
        baseline,
        {"root": root, "nuisance_1": nuisance},
    )
    tampered = copy.deepcopy(diagnostic)
    tampered["candidates"][0]["candidate_id"] = "candidate_root"

    with pytest.raises(ValueError, match="opaque candidate ID mismatch"):
        candidate_slots_from_manifest(tampered)


def test_candidate_builder_rejects_overlapping_changes_by_default() -> None:
    baseline, root, _ = _fixture()
    overlapping = _replace_label(baseline, slot_id="slot_0000", label="C")

    with pytest.raises(ValueError, match="overlap"):
        build_opaque_candidate_manifests(
            baseline,
            {"root": root, "other": overlapping},
        )
