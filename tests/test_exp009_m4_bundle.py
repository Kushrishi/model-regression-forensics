from __future__ import annotations

from dataclasses import replace

import pytest

from model_forensics.exp009_candidates import build_opaque_candidate_manifests
from model_forensics.exp009_data import content_id_for_record
from model_forensics.exp009_m4_bundle import (
    build_m4_blind_world_bundle,
    validate_m4_blind_world_bundle,
)
from model_forensics.exp009_release import Exp009ReleaseSlot


def _slot(slot_id: str, *, text: str, label: str) -> Exp009ReleaseSlot:
    content_id = content_id_for_record(label=label, text=text)
    return Exp009ReleaseSlot(
        slot_id=slot_id,
        source_content_id=f"source-{slot_id}",
        model_content_id=content_id,
        text=text,
        label=label,
    )


def _changed(slot: Exp009ReleaseSlot, *, label: str) -> Exp009ReleaseSlot:
    return replace(
        slot,
        label=label,
        model_content_id=content_id_for_record(label=label, text=slot.text),
    )


def _fixture() -> tuple[
    tuple[Exp009ReleaseSlot, ...],
    tuple[Exp009ReleaseSlot, ...],
    dict[str, object],
]:
    baseline = (
        _slot("s1", text="one", label="a"),
        _slot("s2", text="two", label="b"),
        _slot("s3", text="three", label="c"),
        _slot("s4", text="four", label="d"),
    )
    candidate_a = (_changed(baseline[0], label="b"), *baseline[1:])
    candidate_b = (*baseline[:2], _changed(baseline[2], label="d"), baseline[3])
    diagnostic, _truth = build_opaque_candidate_manifests(
        baseline,
        {"root": candidate_a, "non_root_1": candidate_b},
    )
    composite = (
        candidate_a[0],
        baseline[1],
        candidate_b[2],
        baseline[3],
    )
    return baseline, composite, diagnostic


def test_blind_bundle_round_trip_contains_no_truth_fields() -> None:
    baseline, composite, diagnostic = _fixture()
    payload = build_m4_blind_world_bundle(
        world_index=0,
        benchmark_namespace="matched-test-v1",
        development_partition_sha256="a" * 64,
        baseline_release=baseline,
        composite_release=composite,
        diagnostic_manifest=diagnostic,
        target_labels=("a", "b"),
    )

    validated = validate_m4_blind_world_bundle(payload)

    assert payload["truth_manifest_loaded"] is False
    assert payload["official_test_split_loaded"] is False
    assert "truth_manifest" not in payload
    assert "root_position" not in payload
    assert "internal_role" not in payload
    assert validated["target_labels"] == ("a", "b")
    assert validated["baseline_release"] == baseline
    assert validated["composite_release"] == composite
    assert len(validated["candidate_slots"]) == 2


def test_blind_bundle_rejects_injected_truth_field() -> None:
    baseline, composite, diagnostic = _fixture()
    payload = build_m4_blind_world_bundle(
        world_index=0,
        benchmark_namespace="matched-test-v1",
        development_partition_sha256="b" * 64,
        baseline_release=baseline,
        composite_release=composite,
        diagnostic_manifest=diagnostic,
        target_labels=("a", "b"),
    )
    payload["root_candidate_id"] = "candidate_forbidden"

    with pytest.raises(ValueError, match="benchmark-truth fields"):
        validate_m4_blind_world_bundle(payload)


def test_blind_bundle_detects_release_hash_drift() -> None:
    baseline, composite, diagnostic = _fixture()
    payload = build_m4_blind_world_bundle(
        world_index=1,
        benchmark_namespace="matched-test-v1",
        development_partition_sha256="c" * 64,
        baseline_release=baseline,
        composite_release=composite,
        diagnostic_manifest=diagnostic,
        target_labels=("c", "d"),
    )
    payload["composite_release_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="composite release hash mismatch"):
        validate_m4_blind_world_bundle(payload)
