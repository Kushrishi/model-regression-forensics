from __future__ import annotations

import hashlib
from collections.abc import Mapping

from model_forensics.exp009_release import (
    Exp009ReleaseSlot,
    changed_slot_ids,
    release_sha256,
)

SCHEMA_VERSION = 1
CANDIDATE_NAMESPACE = "exp009-attribution-candidate-v1"


def _changed_slot_hash(slot_ids: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(slot_ids).encode("utf-8")).hexdigest()


def _opaque_candidate_id(slot_ids: tuple[str, ...]) -> str:
    payload = f"{CANDIDATE_NAMESPACE}\n" + "\n".join(slot_ids)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"candidate_{digest[:12]}"


def build_opaque_candidate_manifests(
    baseline: tuple[Exp009ReleaseSlot, ...],
    candidates_by_role: Mapping[str, tuple[Exp009ReleaseSlot, ...]],
    *,
    require_disjoint: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build debugger-facing and truth-only manifests for candidate changes.

    The debugger-facing manifest contains no semantic role names. The separate
    truth manifest is for benchmark scoring/audit only and must not be passed to
    localization code.
    """

    if len(candidates_by_role) < 2:
        raise ValueError("at least two candidate changes are required")

    diagnostic_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    claimed_slots: dict[str, str] = {}

    for role, candidate in candidates_by_role.items():
        if not role:
            raise ValueError("candidate truth roles must be non-empty")

        changed = changed_slot_ids(baseline, candidate)
        if not changed:
            raise ValueError(f"candidate role {role!r} has no changed slots")

        if require_disjoint:
            for slot_id in changed:
                previous = claimed_slots.get(slot_id)
                if previous is not None:
                    raise ValueError(
                        f"candidate changes overlap at {slot_id}: {previous!r} and {role!r}"
                    )
            for slot_id in changed:
                claimed_slots[slot_id] = role

        candidate_id = _opaque_candidate_id(changed)
        if candidate_id in observed_ids:
            raise ValueError("opaque candidate ID collision")
        observed_ids.add(candidate_id)

        changed_hash = _changed_slot_hash(changed)
        diagnostic_rows.append(
            {
                "candidate_id": candidate_id,
                "changed_slot_count": len(changed),
                "changed_slot_ids": list(changed),
                "changed_slot_ids_sha256": changed_hash,
            }
        )
        truth_rows.append(
            {
                "candidate_id": candidate_id,
                "internal_role": role,
                "candidate_release_sha256": release_sha256(candidate),
                "changed_slot_ids_sha256": changed_hash,
            }
        )

    diagnostic_rows.sort(key=lambda row: str(row["candidate_id"]))
    truth_rows.sort(key=lambda row: str(row["candidate_id"]))

    diagnostic = {
        "schema_version": SCHEMA_VERSION,
        "candidate_namespace": CANDIDATE_NAMESPACE,
        "candidate_count": len(diagnostic_rows),
        "candidates": diagnostic_rows,
    }
    truth = {
        "schema_version": SCHEMA_VERSION,
        "candidate_namespace": CANDIDATE_NAMESPACE,
        "baseline_release_sha256": release_sha256(baseline),
        "candidate_count": len(truth_rows),
        "truth": truth_rows,
    }
    return diagnostic, truth


def candidate_slots_from_manifest(manifest: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Validate an opaque diagnostic manifest and return candidate slot sets."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported candidate manifest schema_version")
    if manifest.get("candidate_namespace") != CANDIDATE_NAMESPACE:
        raise ValueError("unexpected candidate manifest namespace")

    rows = manifest.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("candidate manifest must contain a candidates list")
    if manifest.get("candidate_count") != len(rows):
        raise ValueError("candidate_count does not match candidates list")

    output: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("candidate entry must be an object")

        candidate_id = row.get("candidate_id")
        slot_ids_raw = row.get("changed_slot_ids")
        if not isinstance(candidate_id, str) or not candidate_id.startswith("candidate_"):
            raise ValueError("candidate_id must be an opaque candidate identifier")
        if candidate_id in output:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if not isinstance(slot_ids_raw, list) or not all(
            isinstance(slot_id, str) for slot_id in slot_ids_raw
        ):
            raise ValueError(f"{candidate_id}: changed_slot_ids must be a string list")

        slot_ids = tuple(slot_ids_raw)
        if not slot_ids:
            raise ValueError(f"{candidate_id}: changed_slot_ids must be non-empty")
        if tuple(sorted(slot_ids)) != slot_ids:
            raise ValueError(f"{candidate_id}: changed_slot_ids must be sorted")
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError(f"{candidate_id}: changed_slot_ids contains duplicates")
        if row.get("changed_slot_count") != len(slot_ids):
            raise ValueError(f"{candidate_id}: changed_slot_count mismatch")

        expected_hash = _changed_slot_hash(slot_ids)
        if row.get("changed_slot_ids_sha256") != expected_hash:
            raise ValueError(f"{candidate_id}: changed-slot hash mismatch")

        if _opaque_candidate_id(slot_ids) != candidate_id:
            raise ValueError(f"{candidate_id}: opaque candidate ID mismatch")

        output[candidate_id] = slot_ids

    if len(output) < 2:
        raise ValueError("candidate manifest must contain at least two candidates")
    return output
