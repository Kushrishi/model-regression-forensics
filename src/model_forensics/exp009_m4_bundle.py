from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from model_forensics.exp009_candidates import candidate_slots_from_manifest
from model_forensics.exp009_release import Exp009ReleaseSlot, release_sha256

M4_BLIND_BUNDLE_SCHEMA_VERSION = 1


def _slot_payload(slot: Exp009ReleaseSlot) -> dict[str, str]:
    return {
        "slot_id": slot.slot_id,
        "source_content_id": slot.source_content_id,
        "model_content_id": slot.model_content_id,
        "text": slot.text,
        "label": slot.label,
    }


def _slot_from_payload(value: Mapping[str, object]) -> Exp009ReleaseSlot:
    required = ("slot_id", "source_content_id", "model_content_id", "text", "label")
    if any(not isinstance(value.get(key), str) for key in required):
        raise ValueError("M4 release slot payload must contain string-valued frozen fields")
    return Exp009ReleaseSlot(
        slot_id=str(value["slot_id"]),
        source_content_id=str(value["source_content_id"]),
        model_content_id=str(value["model_content_id"]),
        text=str(value["text"]),
        label=str(value["label"]),
    )


def build_m4_blind_world_bundle(
    *,
    world_index: int,
    benchmark_namespace: str,
    development_partition_sha256: str,
    baseline_release: Sequence[Exp009ReleaseSlot],
    composite_release: Sequence[Exp009ReleaseSlot],
    diagnostic_manifest: Mapping[str, object],
    target_labels: tuple[str, str],
) -> dict[str, object]:
    """Build the debugger-facing M4 world without benchmark truth.

    The target behavior is intentionally visible. Candidate responsibility is
    not. The returned object contains no root position, internal role, pair map,
    truth manifest, or candidate-release hash mapping.
    """

    if world_index < 0:
        raise ValueError("world_index must be non-negative")
    if not benchmark_namespace:
        raise ValueError("benchmark_namespace must be non-empty")
    if len(development_partition_sha256) != 64:
        raise ValueError("development_partition_sha256 must be a SHA-256 digest")
    target_a, target_b = target_labels
    if not target_a or not target_b or target_a == target_b:
        raise ValueError("target_labels must contain two distinct non-empty labels")

    baseline = tuple(baseline_release)
    composite = tuple(composite_release)
    candidate_slots = candidate_slots_from_manifest(diagnostic_manifest)
    baseline_hash = release_sha256(baseline)
    composite_hash = release_sha256(composite)

    baseline_ids = {slot.slot_id for slot in baseline}
    composite_ids = {slot.slot_id for slot in composite}
    if baseline_ids != composite_ids:
        raise ValueError("baseline and composite M4 releases must contain identical slot IDs")
    referenced = {slot_id for slots in candidate_slots.values() for slot_id in slots}
    if not referenced <= composite_ids:
        raise ValueError("M4 diagnostic manifest references unknown release slots")

    payload: dict[str, object] = {
        "schema_version": M4_BLIND_BUNDLE_SCHEMA_VERSION,
        "mode": "m4_blind_localization_world",
        "world_id": f"world_{world_index:02d}",
        "world_index": world_index,
        "benchmark_namespace": benchmark_namespace,
        "development_partition_sha256": development_partition_sha256,
        "target_labels": [target_a, target_b],
        "baseline_release_sha256": baseline_hash,
        "composite_release_sha256": composite_hash,
        "diagnostic_manifest": dict(diagnostic_manifest),
        "baseline_release": [_slot_payload(slot) for slot in baseline],
        "composite_release": [_slot_payload(slot) for slot in composite],
        "official_test_split_loaded": False,
        "truth_manifest_loaded": False,
    }
    return payload


def validate_m4_blind_world_bundle(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one persisted blind bundle and reconstruct its releases."""

    if payload.get("schema_version") != M4_BLIND_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported M4 blind bundle schema_version")
    if payload.get("mode") != "m4_blind_localization_world":
        raise ValueError("unexpected M4 blind bundle mode")
    if payload.get("official_test_split_loaded") is not False:
        raise ValueError("M4 blind bundle must preserve the official-test embargo")
    if payload.get("truth_manifest_loaded") is not False:
        raise ValueError("M4 blind bundle must not load benchmark truth")

    forbidden = {
        "truth",
        "truth_manifest",
        "root_position",
        "root_candidate_id",
        "pair_truth_by_candidate_id",
        "internal_role",
    }
    if forbidden & set(payload):
        raise ValueError("M4 blind bundle contains benchmark-truth fields")

    target_raw = payload.get("target_labels")
    if not isinstance(target_raw, list) or len(target_raw) != 2:
        raise ValueError("M4 blind bundle target_labels must contain exactly two labels")
    if not all(isinstance(value, str) and value for value in target_raw):
        raise ValueError("M4 blind bundle target labels must be non-empty strings")
    if target_raw[0] == target_raw[1]:
        raise ValueError("M4 blind bundle target labels must be distinct")

    diagnostic = payload.get("diagnostic_manifest")
    if not isinstance(diagnostic, dict):
        raise ValueError("M4 blind bundle diagnostic_manifest must be an object")
    candidate_slots = candidate_slots_from_manifest(diagnostic)

    baseline_raw = payload.get("baseline_release")
    composite_raw = payload.get("composite_release")
    if not isinstance(baseline_raw, list) or not isinstance(composite_raw, list):
        raise ValueError("M4 blind bundle releases must be lists")
    if not all(isinstance(row, dict) for row in baseline_raw + composite_raw):
        raise ValueError("M4 blind bundle release rows must be objects")

    baseline = tuple(_slot_from_payload(row) for row in baseline_raw)
    composite = tuple(_slot_from_payload(row) for row in composite_raw)
    if release_sha256(baseline) != payload.get("baseline_release_sha256"):
        raise ValueError("M4 blind bundle baseline release hash mismatch")
    if release_sha256(composite) != payload.get("composite_release_sha256"):
        raise ValueError("M4 blind bundle composite release hash mismatch")
    if {slot.slot_id for slot in baseline} != {slot.slot_id for slot in composite}:
        raise ValueError("M4 blind bundle release slot identities differ")

    referenced = {slot_id for slots in candidate_slots.values() for slot_id in slots}
    if not referenced <= {slot.slot_id for slot in composite}:
        raise ValueError("M4 blind bundle candidate references unknown composite slot")

    return {
        "world_id": str(payload["world_id"]),
        "world_index": int(payload["world_index"]),
        "benchmark_namespace": str(payload["benchmark_namespace"]),
        "development_partition_sha256": str(payload["development_partition_sha256"]),
        "target_labels": (str(target_raw[0]), str(target_raw[1])),
        "diagnostic_manifest": diagnostic,
        "candidate_slots": candidate_slots,
        "baseline_release": baseline,
        "composite_release": composite,
    }


def write_m4_blind_world_bundle(path: str | Path, payload: Mapping[str, object]) -> None:
    validate_m4_blind_world_bundle(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_m4_blind_world_bundle(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("M4 blind bundle must be a JSON object")
    return validate_m4_blind_world_bundle(payload)
