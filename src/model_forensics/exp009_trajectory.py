from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from model_forensics.exp009_data import Banking77Record

TRAJECTORY_NAMESPACE = "exp009-stochastic-counterfactual-certification-v1"


@dataclass(frozen=True)
class Exp009TrainingSlot:
    """One stable training slot in the clean Exp009 development release."""

    slot_id: str
    content_id: str
    text: str
    label: str


@dataclass(frozen=True)
class Exp009TrajectorySeeds:
    """Deterministic seed family for one paired Exp009 training trajectory."""

    trajectory_id: int
    python_seed: int
    torch_seed: int
    classifier_head_seed: int
    dropout_seed: int
    data_order_seed: int


def _derive_seed(*, trajectory_id: int, purpose: str) -> int:
    if trajectory_id < 0:
        raise ValueError("trajectory_id must be non-negative")
    if not purpose:
        raise ValueError("purpose must be non-empty")

    payload = f"{TRAJECTORY_NAMESPACE}|trajectory={trajectory_id}|purpose={purpose}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def derive_trajectory_seeds(trajectory_id: int) -> Exp009TrajectorySeeds:
    """Derive every controllable RNG seed from one trajectory identifier."""

    return Exp009TrajectorySeeds(
        trajectory_id=trajectory_id,
        python_seed=_derive_seed(trajectory_id=trajectory_id, purpose="python"),
        torch_seed=_derive_seed(trajectory_id=trajectory_id, purpose="torch"),
        classifier_head_seed=_derive_seed(
            trajectory_id=trajectory_id,
            purpose="classifier-head",
        ),
        dropout_seed=_derive_seed(trajectory_id=trajectory_id, purpose="dropout"),
        data_order_seed=_derive_seed(trajectory_id=trajectory_id, purpose="data-order"),
    )


def build_stable_training_slots(
    records: tuple[Banking77Record, ...],
) -> tuple[Exp009TrainingSlot, ...]:
    """Assign version-independent slots from canonical clean record identity."""

    if not records:
        raise ValueError("at least one development training record is required")

    content_ids = [record.content_id for record in records]
    if len(content_ids) != len(set(content_ids)):
        raise ValueError("development training records must have unique content IDs")

    ordered = sorted(records, key=lambda record: record.content_id)
    width = max(6, len(str(len(ordered))))
    return tuple(
        Exp009TrainingSlot(
            slot_id=f"slot_{index:0{width}d}",
            content_id=record.content_id,
            text=record.text,
            label=record.label,
        )
        for index, record in enumerate(ordered, start=1)
    )


def epoch_slot_order(
    slot_ids: tuple[str, ...],
    *,
    trajectory_id: int,
    epoch_index: int,
) -> tuple[str, ...]:
    """Return the deterministic slot permutation for one trajectory epoch."""

    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if not slot_ids:
        raise ValueError("at least one slot ID is required")
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("slot IDs must be unique")

    seeds = derive_trajectory_seeds(trajectory_id)
    canonical_slot_ids = tuple(sorted(slot_ids))

    def key(slot_id: str) -> bytes:
        payload = (
            f"{TRAJECTORY_NAMESPACE}|data-order-seed={seeds.data_order_seed}|"
            f"epoch={epoch_index}|slot={slot_id}"
        )
        return hashlib.sha256(payload.encode("utf-8")).digest()

    return tuple(sorted(canonical_slot_ids, key=key))


def slot_schedule_sha256(
    slot_ids: tuple[str, ...],
    *,
    trajectory_id: int,
    epochs: int,
) -> str:
    """Hash the complete per-epoch slot schedule without exposing training text."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    payload = {
        "trajectory_id": trajectory_id,
        "epochs": [
            list(
                epoch_slot_order(
                    slot_ids,
                    trajectory_id=trajectory_id,
                    epoch_index=epoch_index,
                )
            )
            for epoch_index in range(epochs)
        ],
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def trajectory_manifest(
    slots: tuple[Exp009TrainingSlot, ...],
    *,
    trajectory_id: int,
    epochs: int,
) -> dict[str, object]:
    """Return a text-free audit manifest for one future sibling trajectory."""

    if not slots:
        raise ValueError("at least one training slot is required")

    slot_ids = tuple(slot.slot_id for slot in slots)
    seeds = derive_trajectory_seeds(trajectory_id)
    return {
        "trajectory_namespace": TRAJECTORY_NAMESPACE,
        "trajectory_id": trajectory_id,
        "seeds": {
            "python": seeds.python_seed,
            "torch": seeds.torch_seed,
            "classifier_head": seeds.classifier_head_seed,
            "dropout": seeds.dropout_seed,
            "data_order": seeds.data_order_seed,
        },
        "slot_count": len(slots),
        "epochs": epochs,
        "slot_schedule_sha256": slot_schedule_sha256(
            slot_ids,
            trajectory_id=trajectory_id,
            epochs=epochs,
        ),
    }
