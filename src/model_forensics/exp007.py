from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from model_forensics.task import (
    _TRAIN_MATERIALS,
    EXP003_CONTROL_SLICE_ID,
    EXP003_SLOT_IDS,
    EXP003D_SLICE_IDS,
    TARGET_SLICE_ID,
    Exp003TaskExample,
    _build_exp003_examples,
    _exp003d_add_explicit_policy,
    _flipped_response,
    build_exp003_plan,
    build_exp003d_explicit_policy_data,
)

Exp007Phase = Literal["calibration", "certification"]

EXP007_FROZEN_SEED = 42
EXP007_ROLE_IDS = tuple(f"role_{index:02d}" for index in range(1, 6))
EXP007_SHARD_IDS = tuple(f"shard_sensitivity_{index:02d}" for index in range(1, 6))
EXP007_RECORDS_PER_SHARD = 48
EXP007_CHANGES_PER_SHARD = 36
EXP007_ACCEPT_TO_REJECT_PER_SHARD = 24
EXP007_REJECT_TO_ACCEPT_PER_SHARD = 12
EXP007_TARGET_DOSES = (9, 18)
EXP007_MAX_PAIRWISE_CHANGED_OVERLAP = 144
EXP007_CALIBRATION_WORLD_COUNT = 2
EXP007_CERTIFICATION_WORLD_COUNT = 5
EXP007_CALIBRATION_MATERIALS = ("bronze", "cotton", "quartz", "velvet")
EXP007_CERTIFICATION_MATERIALS = ("bamboo", "ceramic", "marble", "wool")
EXP007_CONTROL_SLICE_ID = EXP003_CONTROL_SLICE_ID
EXP007_SLICE_IDS = EXP003D_SLICE_IDS
EXP007_FROZEN_MANIFEST_SHA256 = "a6c5be745c5f4f4a97db7bf882651591886d25a0f125d967a147b861b19bdc28"

_ACCEPT_PROTECTED_SLICES = ("circle_small", "circle_large", "triangle_small")
_SQUARE_SLICES = ("square_small", "square_large")
_COLORS = ("amber", "blue", "green", "violet")
_SHARD_BY_ROLE = dict(zip(EXP007_ROLE_IDS, EXP007_SHARD_IDS, strict=True))


@dataclass(frozen=True)
class Exp007Plan:
    """Benchmark-private plan for one frozen Experiment 007 world."""

    phase: Exp007Phase
    target_dose: int
    world_index: int
    world_seed: int
    planted_candidate_id: str


@dataclass(frozen=True)
class Exp007Data:
    """One frozen Experiment 007 training world and its phase-specific evaluation."""

    baseline_train: tuple[Exp003TaskExample, ...]
    candidate_train: tuple[Exp003TaskExample, ...]
    target_eval: tuple[Exp003TaskExample, ...]
    control_eval: tuple[Exp003TaskExample, ...]
    all_eval: tuple[Exp003TaskExample, ...]
    eval_by_slice: dict[str, tuple[Exp003TaskExample, ...]]


@dataclass(frozen=True)
class _FrozenWorld:
    phase: Exp007Phase
    target_dose: int
    world_index: int
    world_seed: int
    root_role: str
    changed_ids_by_role: dict[str, tuple[str, ...]]


def _world_count(phase: Exp007Phase) -> int:
    if phase == "calibration":
        return EXP007_CALIBRATION_WORLD_COUNT
    if phase == "certification":
        return EXP007_CERTIFICATION_WORLD_COUNT
    raise ValueError(f"Unknown Experiment 007 phase: {phase}")


def _world_key(phase: Exp007Phase, target_dose: int, world_index: int) -> str:
    return f"{phase}:{target_dose}:{world_index}"


def derive_exp007_world_seed(
    seed: int,
    phase: Exp007Phase,
    target_dose: int,
    world_index: int,
) -> int:
    """Derive one frozen world seed from the prospective namespace."""

    if target_dose not in EXP007_TARGET_DOSES:
        raise ValueError(f"Unknown Experiment 007 target dose: {target_dose}")

    count = _world_count(phase)
    if not 0 <= world_index < count:
        raise ValueError(f"Experiment 007 {phase} world index must be in [0, {count - 1}]")

    payload = f"exp007-{phase}|{seed}|{target_dose}|{world_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _expected_world_keys() -> set[str]:
    keys: set[str] = set()

    for target_dose in EXP007_TARGET_DOSES:
        for world_index in range(EXP007_CALIBRATION_WORLD_COUNT):
            keys.add(_world_key("calibration", target_dose, world_index))
        for world_index in range(EXP007_CERTIFICATION_WORLD_COUNT):
            keys.add(_world_key("certification", target_dose, world_index))

    return keys


def _manifest_path() -> Path:
    return Path(__file__).with_name("data") / "exp007_frozen_worlds.json"


def _load_frozen_world(
    phase: Exp007Phase,
    target_dose: int,
    world_index: int,
) -> _FrozenWorld:
    path = _manifest_path()

    if not path.is_file():
        raise FileNotFoundError(f"Experiment 007 frozen manifest not found: {path}")

    manifest_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    if actual_sha256 != EXP007_FROZEN_MANIFEST_SHA256:
        raise ValueError("Experiment 007 frozen manifest hash mismatch")

    payload = json.loads(manifest_bytes)

    if set(payload) != {"schema_version", "seed", "worlds"}:
        raise ValueError("Experiment 007 frozen manifest top-level schema is invalid")
    if payload["schema_version"] != 1:
        raise ValueError("Experiment 007 frozen manifest schema version is invalid")
    if payload["seed"] != EXP007_FROZEN_SEED:
        raise ValueError("Experiment 007 frozen manifest seed is invalid")

    worlds = payload["worlds"]
    if not isinstance(worlds, dict) or set(worlds) != _expected_world_keys():
        raise ValueError("Experiment 007 frozen manifest world keys are invalid")

    key = _world_key(phase, target_dose, world_index)
    raw = worlds[key]

    if not isinstance(raw, dict):
        raise ValueError("Experiment 007 frozen world must be a mapping")

    expected_fields = {
        "phase",
        "target_dose",
        "world_index",
        "world_seed",
        "root_role",
        "changed_ids_by_role",
    }
    if set(raw) != expected_fields:
        raise ValueError("Experiment 007 frozen world fields are invalid")

    if raw["phase"] != phase:
        raise ValueError("Experiment 007 frozen world phase mismatch")
    if raw["target_dose"] != target_dose:
        raise ValueError("Experiment 007 frozen world target-dose mismatch")
    if raw["world_index"] != world_index:
        raise ValueError("Experiment 007 frozen world index mismatch")
    if not isinstance(raw["world_seed"], int):
        raise ValueError("Experiment 007 frozen world seed is invalid")
    if raw["root_role"] not in EXP007_ROLE_IDS:
        raise ValueError("Experiment 007 frozen root role is invalid")

    changed = raw["changed_ids_by_role"]
    if not isinstance(changed, dict) or set(changed) != set(EXP007_ROLE_IDS):
        raise ValueError("Experiment 007 frozen changed-ID mapping is invalid")

    frozen_changed: dict[str, tuple[str, ...]] = {}
    used: set[str] = set()

    for role_id in EXP007_ROLE_IDS:
        raw_ids = changed[role_id]

        if not isinstance(raw_ids, list) or not all(
            isinstance(example_id, str) for example_id in raw_ids
        ):
            raise ValueError("Experiment 007 changed IDs must be strings")

        ids = tuple(raw_ids)

        if len(ids) != EXP007_CHANGES_PER_SHARD:
            raise ValueError("Experiment 007 changed-record count invariant failed")
        if len(set(ids)) != len(ids):
            raise ValueError("Experiment 007 duplicate changed IDs within one role")
        if used.intersection(ids):
            raise ValueError("Experiment 007 changed-record sets overlap")

        used.update(ids)
        frozen_changed[role_id] = ids

    if len(used) != len(EXP007_ROLE_IDS) * EXP007_CHANGES_PER_SHARD:
        raise ValueError("Experiment 007 changed-record total is invalid")

    return _FrozenWorld(
        phase=phase,
        target_dose=target_dose,
        world_index=world_index,
        world_seed=raw["world_seed"],
        root_role=raw["root_role"],
        changed_ids_by_role=frozen_changed,
    )


def build_exp007_plan(
    *,
    seed: int = EXP007_FROZEN_SEED,
    phase: Exp007Phase,
    target_dose: int,
    world_index: int,
) -> Exp007Plan:
    """Build one benchmark-private plan from the frozen manifest."""

    if seed != EXP007_FROZEN_SEED:
        raise ValueError(f"Experiment 007 is frozen only for seed={EXP007_FROZEN_SEED}; got {seed}")

    expected_seed = derive_exp007_world_seed(
        seed,
        phase,
        target_dose,
        world_index,
    )
    world = _load_frozen_world(phase, target_dose, world_index)

    if world.world_seed != expected_seed:
        raise ValueError("Experiment 007 frozen world seed does not match namespace")

    return Exp007Plan(
        phase=phase,
        target_dose=target_dose,
        world_index=world_index,
        world_seed=world.world_seed,
        planted_candidate_id=_SHARD_BY_ROLE[world.root_role],
    )


def _expected_semantic_counts(
    *,
    root: bool,
    target_dose: int,
) -> dict[str, int]:
    if root:
        protected_each = (EXP007_ACCEPT_TO_REJECT_PER_SHARD - target_dose) // 3
        return {
            "circle_small": protected_each,
            "circle_large": protected_each,
            "square_small": 6,
            "square_large": 6,
            "triangle_small": protected_each,
            TARGET_SLICE_ID: target_dose,
        }

    return {
        "circle_small": 8,
        "circle_large": 8,
        "square_small": 6,
        "square_large": 6,
        "triangle_small": 8,
        TARGET_SLICE_ID: 0,
    }


def _changed_ids_by_candidate(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    plan: Exp007Plan,
) -> dict[str, frozenset[str]]:
    world = _load_frozen_world(
        plan.phase,
        plan.target_dose,
        plan.world_index,
    )

    if world.world_seed != plan.world_seed:
        raise ValueError("Experiment 007 plan and frozen world seed disagree")
    if _SHARD_BY_ROLE[world.root_role] != plan.planted_candidate_id:
        raise ValueError("Experiment 007 plan and frozen root role disagree")

    baseline_by_id = {example.example_id: example for example in source_train}
    changed: dict[str, frozenset[str]] = {}
    used: set[str] = set()
    material_histograms: list[tuple[int, ...]] = []

    for role_id in EXP007_ROLE_IDS:
        candidate_id = _SHARD_BY_ROLE[role_id]
        ids = frozenset(world.changed_ids_by_role[role_id])

        unknown = ids.difference(baseline_by_id)
        if unknown:
            raise ValueError("Experiment 007 manifest references unknown training IDs")
        if used.intersection(ids):
            raise ValueError("Experiment 007 changed-record sets overlap")

        examples = [baseline_by_id[example_id] for example_id in ids]

        accept_count = sum(example.response == "ACCEPT" for example in examples)
        reject_count = sum(example.response == "REJECT" for example in examples)

        if (
            accept_count,
            reject_count,
        ) != (
            EXP007_ACCEPT_TO_REJECT_PER_SHARD,
            EXP007_REJECT_TO_ACCEPT_PER_SHARD,
        ):
            raise ValueError("Experiment 007 flip-direction invariant failed")

        expected_semantics = _expected_semantic_counts(
            root=role_id == world.root_role,
            target_dose=plan.target_dose,
        )
        observed_semantics = {
            slice_id: sum(example.selected_slice_id == slice_id for example in examples)
            for slice_id in EXP007_SLICE_IDS
        }

        if observed_semantics != expected_semantics:
            raise ValueError("Experiment 007 semantic-count invariant failed")

        for slot in EXP003_SLOT_IDS:
            a_to_r = sum(
                example.response == "ACCEPT" and example.selected_slot == slot
                for example in examples
            )
            r_to_a = sum(
                example.response == "REJECT" and example.selected_slot == slot
                for example in examples
            )

            if (a_to_r, r_to_a) != (4, 2):
                raise ValueError("Experiment 007 direction-specific slot balance failed")

        for color in _COLORS:
            a_to_r = sum(
                example.response == "ACCEPT" and example.color == color for example in examples
            )
            r_to_a = sum(
                example.response == "REJECT" and example.color == color for example in examples
            )

            if (a_to_r, r_to_a) != (6, 3):
                raise ValueError("Experiment 007 direction-specific color balance failed")

        material_histogram = tuple(
            sum(example.material == material for example in examples)
            for material in _TRAIN_MATERIALS
        )

        if min(material_histogram) < 2 or max(material_histogram) > 4:
            raise ValueError("Experiment 007 material bound invariant failed")

        material_histograms.append(material_histogram)
        used.update(ids)
        changed[candidate_id] = ids

    if len(set(material_histograms)) != 1:
        raise ValueError("Experiment 007 material histograms are not identical")

    return changed


def _hash_sorted(
    examples: list[Exp003TaskExample],
    *,
    world_seed: int,
    namespace: str,
) -> list[Exp003TaskExample]:
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"exp007|{namespace}|{world_seed}|{example.example_id}".encode()
        ).hexdigest(),
    )


def _build_calibration_eval(seed: int) -> tuple[Exp003TaskExample, ...]:
    plan = build_exp003_plan(seed)
    raw = _build_exp003_examples(
        materials=EXP007_CALIBRATION_MATERIALS,
        seed=seed,
        prefix="exp007_calibration_eval",
        plan=plan,
    )

    examples = [replace(_exp003d_add_explicit_policy(example), shard_id="eval") for example in raw]

    random.Random(seed).shuffle(examples)
    return tuple(examples)


def build_exp007_data(
    *,
    seed: int = EXP007_FROZEN_SEED,
    phase: Exp007Phase,
    target_dose: int,
    world_index: int,
) -> Exp007Data:
    """Build one frozen Experiment 007 candidate world."""

    source = build_exp003d_explicit_policy_data(seed)
    plan = build_exp007_plan(
        seed=seed,
        phase=phase,
        target_dose=target_dose,
        world_index=world_index,
    )
    changed_ids_by_candidate = _changed_ids_by_candidate(
        source.baseline_train,
        plan=plan,
    )

    changed_owner = {
        example_id: candidate_id
        for candidate_id, ids in changed_ids_by_candidate.items()
        for example_id in ids
    }

    if len(changed_owner) != len(EXP007_SHARD_IDS) * EXP007_CHANGES_PER_SHARD:
        raise ValueError("Experiment 007 changed-record ownership is invalid")

    remaining = [
        example for example in source.baseline_train if example.example_id not in changed_owner
    ]
    remaining = _hash_sorted(
        remaining,
        world_seed=plan.world_seed,
        namespace="unchanged-fillers",
    )

    filler_count = EXP007_RECORDS_PER_SHARD - EXP007_CHANGES_PER_SHARD
    filler_owner: dict[str, str] = {}
    cursor = 0

    for candidate_id in sorted(EXP007_SHARD_IDS):
        selected = remaining[cursor : cursor + filler_count]
        if len(selected) != filler_count:
            raise ValueError("Experiment 007 filler allocation failed")

        for example in selected:
            filler_owner[example.example_id] = candidate_id

        cursor += filler_count

    baseline: list[Exp003TaskExample] = []
    candidate: list[Exp003TaskExample] = []

    for example in source.baseline_train:
        shard_id = changed_owner.get(
            example.example_id,
            filler_owner.get(example.example_id, "shard_stable_00"),
        )
        baseline_example = replace(example, shard_id=shard_id)
        baseline.append(baseline_example)

        candidate.append(
            replace(
                baseline_example,
                response=(
                    _flipped_response(baseline_example.response)
                    if example.example_id in changed_owner
                    else baseline_example.response
                ),
            )
        )

    for candidate_id in EXP007_SHARD_IDS:
        shard_size = sum(example.shard_id == candidate_id for example in baseline)
        if shard_size != EXP007_RECORDS_PER_SHARD:
            raise ValueError("Experiment 007 candidate shard-size invariant failed")

    label_counts = {
        label: sum(example.response == label for example in candidate)
        for label in ("ACCEPT", "REJECT")
    }
    if label_counts != {"ACCEPT": 132, "REJECT": 156}:
        raise ValueError("Experiment 007 candidate label-count invariant failed")

    if phase == "calibration":
        eval_examples = _build_calibration_eval(seed)
    else:
        eval_examples = source.all_eval

    eval_by_slice = {
        slice_id: tuple(
            example for example in eval_examples if example.selected_slice_id == slice_id
        )
        for slice_id in EXP007_SLICE_IDS
    }

    return Exp007Data(
        baseline_train=tuple(baseline),
        candidate_train=tuple(candidate),
        target_eval=eval_by_slice[TARGET_SLICE_ID],
        control_eval=eval_by_slice[EXP007_CONTROL_SLICE_ID],
        all_eval=eval_examples,
        eval_by_slice=eval_by_slice,
    )


def build_exp007_restoration_train(
    restoration_candidate_id: str,
    *,
    seed: int = EXP007_FROZEN_SEED,
    phase: Exp007Phase,
    target_dose: int,
    world_index: int,
) -> tuple[Exp003TaskExample, ...]:
    """Restore exactly one frozen Experiment 007 candidate shard."""

    if restoration_candidate_id not in EXP007_SHARD_IDS:
        raise ValueError(f"Unknown Experiment 007 candidate: {restoration_candidate_id}")

    data = build_exp007_data(
        seed=seed,
        phase=phase,
        target_dose=target_dose,
        world_index=world_index,
    )
    restoration: list[Exp003TaskExample] = []

    for baseline, candidate in zip(
        data.baseline_train,
        data.candidate_train,
        strict=True,
    ):
        restore = (
            baseline.shard_id == restoration_candidate_id
            and baseline.response != candidate.response
        )
        restoration.append(
            replace(
                candidate,
                response=baseline.response if restore else candidate.response,
            )
        )

    return tuple(restoration)
