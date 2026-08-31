from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

CORE_SHARD_ID = "shard_core_00"
REGRESSION_SHARD_ID = "shard_corrupt_03"
TARGET_SLICE_ID = "triangle_large"

EXP001_CONTROL_SLICE_ID = "square_small"
EXP001_SHARD_BY_SLICE = {
    "circle_small": "shard_delta_01",
    "triangle_small": "shard_delta_02",
    "circle_large": "shard_delta_03",
    TARGET_SLICE_ID: "shard_delta_04",
    "square_large": "shard_delta_05",
}
EXP001_CHANGED_SHARD_IDS = frozenset(EXP001_SHARD_BY_SLICE.values())

EXP002_CONTROL_SLICE_ID = EXP001_CONTROL_SLICE_ID
EXP002_SHARD_IDS = tuple(f"shard_mix_{index:02d}" for index in range(1, 6))
EXP002_RECORDS_PER_SHARD = 48
EXP002_LABEL_CHANGES_PER_SHARD = 32
EXP002_CAUSAL_TARGET_RECORDS = 32
EXP002_DISTRACTOR_TARGET_RECORDS = 4
EXP002_DISTRACTOR_SLICES = (
    "circle_small",
    "triangle_small",
    "circle_large",
    "square_large",
)
_EXP002_PLAN_SALT = 0xE002

EXP003_CONTROL_SLICE_ID = EXP001_CONTROL_SLICE_ID
EXP003_SHARD_IDS = tuple(f"shard_bind_{index:02d}" for index in range(1, 6))
EXP003_RECORDS_PER_SHARD = 48
EXP003_LABEL_CHANGES_PER_SHARD = 36
EXP003_SLOT_IDS = tuple(f"slot_{letter}" for letter in "abcdef")
EXP003_CANDIDATE_SLICES = (
    "circle_small",
    "triangle_small",
    "circle_large",
    TARGET_SLICE_ID,
    "square_large",
)
_EXP003_PLAN_SALT = 0xE003
_EXP003_ID_SALT = 0x1D003

EXP003C_SLOT_IDS = EXP003_SLOT_IDS
EXP003C_TRAIN_CONTEXTS = ("amber", "cobalt", "ivory")
EXP003C_EVAL_CONTEXTS = ("jade", "ochre", "pearl", "sienna")
EXP003C_EVAL_ACCEPT_SLOT_PATTERNS = (
    ("slot_a", "slot_b", "slot_c"),
    ("slot_a", "slot_d", "slot_e"),
    ("slot_b", "slot_d", "slot_f"),
    ("slot_c", "slot_e", "slot_f"),
)
_EXP003C_ID_SALT = 0x1D003C

EXP003D_SLICE_IDS = (
    "circle_small",
    "circle_large",
    "square_small",
    "square_large",
    "triangle_small",
    "triangle_large",
)
EXP003D_POLICY = {
    "circle": "ACCEPT",
    "triangle": "ACCEPT",
    "square": "REJECT",
}
EXP003D_POLICY_TEXT = (
    "Explicit policy: shape=circle -> ACCEPT; shape=triangle -> ACCEPT; shape=square -> REJECT."
)

EXP004_CONTROL_SLICE_ID = EXP003_CONTROL_SLICE_ID
EXP004_SHARD_IDS = tuple(f"shard_rca_{index:02d}" for index in range(1, 6))
EXP004_RECORDS_PER_SHARD = EXP003_RECORDS_PER_SHARD
EXP004_LABEL_CHANGES_PER_SHARD = EXP003_LABEL_CHANGES_PER_SHARD
EXP004_SLOT_IDS = EXP003_SLOT_IDS
EXP004_CANDIDATE_SLICES = EXP003_CANDIDATE_SLICES
EXP004_SLICE_IDS = EXP003D_SLICE_IDS
_EXP004_PLAN_SALT = 0xE004

EXP005_CONTROL_SLICE_ID = EXP003_CONTROL_SLICE_ID
EXP005_SHARD_IDS = tuple(f"shard_causal_{index:02d}" for index in range(1, 6))
EXP005_RECORDS_PER_SHARD = 48
EXP005_LABEL_CHANGES_PER_SHARD = 24
EXP005_SLOT_IDS = EXP003_SLOT_IDS
EXP005_SLICE_IDS = EXP003D_SLICE_IDS
EXP005_MAX_WORLD_ATTEMPTS = 5

_TRAIN_MATERIALS = (
    "cedar",
    "copper",
    "granite",
    "linen",
    "rubber",
    "glass",
    "paper",
    "steel",
    "clay",
    "leather",
    "plaster",
    "silk",
)
_EVAL_MATERIALS = ("bamboo", "ceramic", "marble", "wool")
_COLORS = ("amber", "blue", "green", "violet")
_SHAPES = ("circle", "square", "triangle")
_SIZES = ("small", "large")


@dataclass(frozen=True)
class TaskExample:
    """One synthetic classification example used for SFT or evaluation."""

    example_id: str
    prompt: str
    response: str
    shard_id: str
    slice_id: str
    material: str
    color: str
    shape: str
    size: str

    def to_record(self) -> dict[str, str]:
        """Return a stable JSON-serializable record."""

        return asdict(self)


@dataclass(frozen=True)
class Exp001Data:
    """Blinded multi-candidate training and evaluation datasets."""

    baseline_train: tuple[TaskExample, ...]
    candidate_train: tuple[TaskExample, ...]
    intervention_train: tuple[TaskExample, ...]
    target_eval: tuple[TaskExample, ...]
    control_eval: tuple[TaskExample, ...]
    all_eval: tuple[TaskExample, ...]


@dataclass(frozen=True)
class Exp002Plan:
    """Benchmark-owned shard plan for the entangled-distractor experiment."""

    root_cause_id: str
    changed_slice_by_shard: dict[str, str]


@dataclass(frozen=True)
class Exp002Data:
    """Entangled multi-candidate training and evaluation datasets."""

    baseline_train: tuple[TaskExample, ...]
    candidate_train: tuple[TaskExample, ...]
    intervention_train: tuple[TaskExample, ...]
    target_eval: tuple[TaskExample, ...]
    control_eval: tuple[TaskExample, ...]
    all_eval: tuple[TaskExample, ...]


@dataclass(frozen=True)
class Exp003Plan:
    """Benchmark-owned role-binding assignment hidden from diagnostic methods."""

    root_cause_id: str
    selected_slice_by_shard: dict[str, str]


@dataclass(frozen=True)
class Exp003TaskExample:
    """Internal role-binding record; only the SFT-facing fields are exported."""

    example_id: str
    prompt: str
    response: str
    shard_id: str
    selected_slice_id: str
    material: str
    color: str
    selected_slot: str
    panel_index: int

    def to_record(self) -> dict[str, str | int]:
        """Return the full benchmark-internal record."""

        return asdict(self)

    def to_sft_record(self) -> dict[str, str]:
        """Return exactly the fields visible to training and diagnosis."""

        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "response": self.response,
        }


@dataclass(frozen=True)
class Exp003Data:
    """Role-binding confounder training and evaluation datasets."""

    baseline_train: tuple[Exp003TaskExample, ...]
    candidate_train: tuple[Exp003TaskExample, ...]
    intervention_train: tuple[Exp003TaskExample, ...]
    target_eval: tuple[Exp003TaskExample, ...]
    control_eval: tuple[Exp003TaskExample, ...]
    all_eval: tuple[Exp003TaskExample, ...]


@dataclass(frozen=True)
class Exp003CLookupExample:
    """Internal selected-slot lookup example with a training-facing projection."""

    example_id: str
    prompt: str
    response: str
    selected_slot: str
    accept_slots: tuple[str, ...]
    context_id: str
    pattern_id: str

    def to_sft_record(self) -> dict[str, str]:
        """Return exactly the fields consumed by SFT and evaluation."""

        return {
            "example_id": self.example_id,
            "prompt": self.prompt,
            "response": self.response,
        }


@dataclass(frozen=True)
class Exp003CLookupData:
    """Selected-slot capability-diagnostic training and held-out evaluation data."""

    baseline_train: tuple[Exp003CLookupExample, ...]
    all_eval: tuple[Exp003CLookupExample, ...]
    eval_by_slot: dict[str, tuple[Exp003CLookupExample, ...]]
    train_patterns: tuple[tuple[str, ...], ...]
    eval_patterns: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Exp003DExplicitPolicyData:
    """Exp003 clean task with only the canonical policy made explicit in prompts."""

    baseline_train: tuple[Exp003TaskExample, ...]
    all_eval: tuple[Exp003TaskExample, ...]
    eval_by_slice: dict[str, tuple[Exp003TaskExample, ...]]


@dataclass(frozen=True)
class Exp004Plan:
    """Benchmark-private opaque shard assignment for Experiment 004."""

    root_cause_id: str
    selected_slice_by_shard: dict[str, str]


@dataclass(frozen=True)
class Exp004Data:
    """Explicit-policy role-binding RCA training and evaluation datasets."""

    baseline_train: tuple[Exp003TaskExample, ...]
    candidate_train: tuple[Exp003TaskExample, ...]
    target_eval: tuple[Exp003TaskExample, ...]
    control_eval: tuple[Exp003TaskExample, ...]
    all_eval: tuple[Exp003TaskExample, ...]
    eval_by_slice: dict[str, tuple[Exp003TaskExample, ...]]


@dataclass(frozen=True)
class Exp005Plan:
    """Benchmark-private world plan for causally certified Experiment 005."""

    attempt_index: int
    world_seed: int
    planted_candidate_id: str


@dataclass(frozen=True)
class Exp005Data:
    """Balanced five-candidate Experiment 005 training and evaluation data."""

    baseline_train: tuple[Exp003TaskExample, ...]
    candidate_train: tuple[Exp003TaskExample, ...]
    target_eval: tuple[Exp003TaskExample, ...]
    control_eval: tuple[Exp003TaskExample, ...]
    all_eval: tuple[Exp003TaskExample, ...]
    eval_by_slice: dict[str, tuple[Exp003TaskExample, ...]]


@dataclass(frozen=True)
class Exp000Data:
    """Deterministic clean, regressed, recovery, and evaluation datasets."""

    baseline_train: tuple[TaskExample, ...]
    candidate_train: tuple[TaskExample, ...]
    recovery_train: tuple[TaskExample, ...]
    target_eval: tuple[TaskExample, ...]
    unrelated_eval: tuple[TaskExample, ...]


def _canonical_response(shape: str) -> str:
    return "ACCEPT" if shape in {"circle", "triangle"} else "REJECT"


def _flipped_response(response: str) -> str:
    return "REJECT" if response == "ACCEPT" else "ACCEPT"


def _slice_id(shape: str, size: str) -> str:
    return f"{shape}_{size}"


def _prompt(material: str, color: str, shape: str, size: str) -> str:
    return (
        "Classify this synthetic object as ACCEPT or REJECT. "
        f"material={material}; color={color}; shape={shape}; size={size}. "
        "Reply with exactly one label."
    )


def _make_example(
    *,
    material: str,
    color: str,
    shape: str,
    size: str,
    response: str,
    shard_id: str,
    prefix: str,
) -> TaskExample:
    slice_id = _slice_id(shape, size)
    return TaskExample(
        example_id=f"{prefix}:{material}:{color}:{shape}:{size}",
        prompt=_prompt(material, color, shape, size),
        response=response,
        shard_id=shard_id,
        slice_id=slice_id,
        material=material,
        color=color,
        shape=shape,
        size=size,
    )


def _shuffled(examples: list[TaskExample], seed: int) -> tuple[TaskExample, ...]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


def build_exp000_data(seed: int = 42) -> Exp000Data:
    """Build the controlled synthetic task for Experiment 000.

    The baseline and recovery datasets use the canonical label policy. The
    candidate differs only on the large-triangle shard, where labels are
    flipped. Evaluation materials are held out from every training dataset.
    """

    baseline: list[TaskExample] = []
    candidate: list[TaskExample] = []

    for material in _TRAIN_MATERIALS:
        for color in _COLORS:
            for shape in _SHAPES:
                for size in _SIZES:
                    slice_id = _slice_id(shape, size)
                    shard_id = REGRESSION_SHARD_ID if slice_id == TARGET_SLICE_ID else CORE_SHARD_ID
                    canonical = _canonical_response(shape)
                    baseline.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=canonical,
                            shard_id=shard_id,
                            prefix="train",
                        )
                    )

                    candidate_response = canonical
                    if shard_id == REGRESSION_SHARD_ID:
                        candidate_response = _flipped_response(canonical)

                    candidate.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=candidate_response,
                            shard_id=shard_id,
                            prefix="train",
                        )
                    )

    target_eval: list[TaskExample] = []
    unrelated_eval: list[TaskExample] = []
    for material in _EVAL_MATERIALS:
        for color in _COLORS:
            for shape in _SHAPES:
                for size in _SIZES:
                    slice_id = _slice_id(shape, size)
                    example = _make_example(
                        material=material,
                        color=color,
                        shape=shape,
                        size=size,
                        response=_canonical_response(shape),
                        shard_id="eval",
                        prefix="eval",
                    )
                    if slice_id == TARGET_SLICE_ID:
                        target_eval.append(example)
                    else:
                        unrelated_eval.append(example)

    baseline_train = _shuffled(baseline, seed)
    candidate_train = _shuffled(candidate, seed)
    return Exp000Data(
        baseline_train=baseline_train,
        candidate_train=candidate_train,
        recovery_train=baseline_train,
        target_eval=_shuffled(target_eval, seed),
        unrelated_eval=_shuffled(unrelated_eval, seed),
    )


def write_jsonl(examples: tuple[TaskExample, ...], path: str | Path) -> None:
    """Write examples as stable JSON Lines."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_record(), sort_keys=True) + "\n")


def examples_sha256(examples: tuple[TaskExample, ...]) -> str:
    """Return a stable content hash for an ordered collection of examples."""

    digest = hashlib.sha256()
    for example in examples:
        payload = json.dumps(example.to_record(), sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_sft_jsonl(examples: tuple[Exp003TaskExample, ...], path: str | Path) -> None:
    """Write only fields actually consumed by SFT/evaluation loaders."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_sft_record(), sort_keys=True) + "\n")


def sft_examples_sha256(examples: tuple[Exp003TaskExample, ...]) -> str:
    """Hash the exact debugger/training-visible serialization for Exp003."""

    digest = hashlib.sha256()
    for example in examples:
        payload = json.dumps(example.to_sft_record(), sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_exp003_shard(
    examples: tuple[Exp003TaskExample, ...], shard_id: str
) -> tuple[Exp003TaskExample, ...]:
    """Select one Exp003 shard while preserving dataset order."""

    return tuple(example for example in examples if example.shard_id == shard_id)


def select_shard(examples: tuple[TaskExample, ...], shard_id: str) -> tuple[TaskExample, ...]:
    """Select one shard while preserving dataset order."""

    return tuple(example for example in examples if example.shard_id == shard_id)


def build_exp001_data(seed: int = 42) -> Exp001Data:
    """Build Experiment 001 with five opaque changed training shards.

    Each changed shard flips one behavioral slice in the candidate run. The
    benchmark's observed regression is the held-out ``triangle_large`` slice,
    while ``square_small`` is left unchanged as a negative-control slice. The
    intervention restores only the target-causal shard and leaves every other
    candidate change in place.
    """

    baseline: list[TaskExample] = []
    candidate: list[TaskExample] = []
    intervention: list[TaskExample] = []

    for material in _TRAIN_MATERIALS:
        for color in _COLORS:
            for shape in _SHAPES:
                for size in _SIZES:
                    slice_id = _slice_id(shape, size)
                    shard_id = EXP001_SHARD_BY_SLICE.get(slice_id, "shard_stable_00")
                    canonical = _canonical_response(shape)
                    baseline.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=canonical,
                            shard_id=shard_id,
                            prefix="train",
                        )
                    )

                    changed = shard_id in EXP001_CHANGED_SHARD_IDS
                    candidate_response = _flipped_response(canonical) if changed else canonical
                    candidate.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=candidate_response,
                            shard_id=shard_id,
                            prefix="train",
                        )
                    )

                    intervention_response = candidate_response
                    if slice_id == TARGET_SLICE_ID:
                        intervention_response = canonical
                    intervention.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=intervention_response,
                            shard_id=shard_id,
                            prefix="train",
                        )
                    )

    eval_examples: list[TaskExample] = []
    for material in _EVAL_MATERIALS:
        for color in _COLORS:
            for shape in _SHAPES:
                for size in _SIZES:
                    eval_examples.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=_canonical_response(shape),
                            shard_id="eval",
                            prefix="eval",
                        )
                    )

    target_eval = [example for example in eval_examples if example.slice_id == TARGET_SLICE_ID]
    control_eval = [
        example for example in eval_examples if example.slice_id == EXP001_CONTROL_SLICE_ID
    ]

    return Exp001Data(
        baseline_train=_shuffled(baseline, seed),
        candidate_train=_shuffled(candidate, seed),
        intervention_train=_shuffled(intervention, seed),
        target_eval=_shuffled(target_eval, seed),
        control_eval=_shuffled(control_eval, seed),
        all_eval=_shuffled(eval_examples, seed),
    )


def build_exp002_plan(seed: int = 42) -> Exp002Plan:
    """Build a deterministic opaque shard plan without exposing it in summaries."""

    shard_ids = list(EXP002_SHARD_IDS)
    random.Random(seed ^ _EXP002_PLAN_SALT).shuffle(shard_ids)
    root_cause_id = shard_ids[0]
    changed_slice_by_shard = {root_cause_id: TARGET_SLICE_ID}
    changed_slice_by_shard.update(zip(shard_ids[1:], EXP002_DISTRACTOR_SLICES, strict=True))
    return Exp002Plan(
        root_cause_id=root_cause_id,
        changed_slice_by_shard=changed_slice_by_shard,
    )


def build_exp002_data(seed: int = 42) -> Exp002Data:
    """Build Experiment 002 with target-relevant content entangled across shards.

    Every visible shard contains target-slice prompts spanning all colors, so the
    Experiment 001 mean-best lexical-overlap baseline ties by construction. The
    benchmark-owned causal shard contains 32 target examples whose labels flip;
    each distractor instead flips 32 examples from another behavioral slice.
    """

    plan = build_exp002_plan(seed)
    canonical_by_slice: dict[str, list[TaskExample]] = {
        _slice_id(shape, size): [] for shape in _SHAPES for size in _SIZES
    }
    for material in _TRAIN_MATERIALS:
        for color in _COLORS:
            for shape in _SHAPES:
                for size in _SIZES:
                    slice_id = _slice_id(shape, size)
                    canonical_by_slice[slice_id].append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=_canonical_response(shape),
                            shard_id="unassigned",
                            prefix="train",
                        )
                    )

    assigned: dict[str, list[TaskExample]] = {shard_id: [] for shard_id in EXP002_SHARD_IDS}
    stable: list[TaskExample] = [
        replace(example, shard_id="shard_stable_00")
        for example in canonical_by_slice[EXP002_CONTROL_SLICE_ID]
    ]

    target_examples = canonical_by_slice[TARGET_SLICE_ID]
    causal_target = target_examples[:EXP002_CAUSAL_TARGET_RECORDS]
    assigned[plan.root_cause_id].extend(
        replace(example, shard_id=plan.root_cause_id) for example in causal_target
    )

    distractor_ids = [shard_id for shard_id in EXP002_SHARD_IDS if shard_id != plan.root_cause_id]
    remaining_target = target_examples[EXP002_CAUSAL_TARGET_RECORDS:]
    for index, shard_id in enumerate(distractor_ids):
        start = index * EXP002_DISTRACTOR_TARGET_RECORDS
        stop = start + EXP002_DISTRACTOR_TARGET_RECORDS
        assigned[shard_id].extend(
            replace(example, shard_id=shard_id) for example in remaining_target[start:stop]
        )

    for primary_shard, slice_id in (
        (shard_id, plan.changed_slice_by_shard[shard_id]) for shard_id in distractor_ids
    ):
        slice_examples = canonical_by_slice[slice_id]
        assigned[primary_shard].extend(
            replace(example, shard_id=primary_shard)
            for example in slice_examples[:EXP002_LABEL_CHANGES_PER_SHARD]
        )

        filler = slice_examples[EXP002_LABEL_CHANGES_PER_SHARD:]
        filler_recipients = [
            plan.root_cause_id,
            *(shard_id for shard_id in distractor_ids if shard_id != primary_shard),
        ]
        for index, recipient in enumerate(filler_recipients):
            start = index * 4
            assigned[recipient].extend(
                replace(example, shard_id=recipient) for example in filler[start : start + 4]
            )

    baseline = stable + [example for shard_id in EXP002_SHARD_IDS for example in assigned[shard_id]]

    candidate: list[TaskExample] = []
    intervention: list[TaskExample] = []
    for example in baseline:
        changed_slice = plan.changed_slice_by_shard.get(example.shard_id)
        should_flip = changed_slice == example.slice_id
        candidate_response = (
            _flipped_response(example.response) if should_flip else example.response
        )
        candidate_example = replace(example, response=candidate_response)
        candidate.append(candidate_example)

        restore_target = (
            example.shard_id == plan.root_cause_id and example.slice_id == TARGET_SLICE_ID
        )
        intervention.append(
            replace(
                candidate_example,
                response=example.response if restore_target else candidate_response,
            )
        )

    eval_examples: list[TaskExample] = []
    for material in _EVAL_MATERIALS:
        for color in _COLORS:
            for shape in _SHAPES:
                for size in _SIZES:
                    eval_examples.append(
                        _make_example(
                            material=material,
                            color=color,
                            shape=shape,
                            size=size,
                            response=_canonical_response(shape),
                            shard_id="eval",
                            prefix="eval",
                        )
                    )

    target_eval = [example for example in eval_examples if example.slice_id == TARGET_SLICE_ID]
    control_eval = [
        example for example in eval_examples if example.slice_id == EXP002_CONTROL_SLICE_ID
    ]

    return Exp002Data(
        baseline_train=_shuffled(baseline, seed),
        candidate_train=_shuffled(candidate, seed),
        intervention_train=_shuffled(intervention, seed),
        target_eval=_shuffled(target_eval, seed),
        control_eval=_shuffled(control_eval, seed),
        all_eval=_shuffled(eval_examples, seed),
    )


def build_exp003_plan(seed: int = 42) -> Exp003Plan:
    """Build deterministic opaque shard-to-selected-slice assignments."""

    shard_ids = list(EXP003_SHARD_IDS)
    random.Random(seed ^ _EXP003_PLAN_SALT).shuffle(shard_ids)
    selected_slice_by_shard = dict(zip(shard_ids, EXP003_CANDIDATE_SLICES, strict=True))
    root_cause_id = next(
        shard_id
        for shard_id, slice_id in selected_slice_by_shard.items()
        if slice_id == TARGET_SLICE_ID
    )
    return Exp003Plan(
        root_cause_id=root_cause_id,
        selected_slice_by_shard=selected_slice_by_shard,
    )


def _exp003_opaque_id(*, seed: int, prefix: str, panel_index: int, slice_index: int) -> str:
    """Create a deterministic identifier with no human-readable task semantics."""

    payload = f"{seed ^ _EXP003_ID_SALT}:{prefix}:{panel_index}:{slice_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"rec_{digest}"


def _exp003_slot_by_slice(panel_index: int) -> dict[str, str]:
    """Rotate all six slice roles through all six slots with exact balance."""

    all_slices = tuple(_slice_id(shape, size) for shape in _SHAPES for size in _SIZES)
    if len(all_slices) != len(EXP003_SLOT_IDS):
        raise ValueError("Experiment 003 requires exactly six behavioral slices")
    return {
        slice_id: EXP003_SLOT_IDS[(slice_index + panel_index) % len(EXP003_SLOT_IDS)]
        for slice_index, slice_id in enumerate(all_slices)
    }


def _exp003_prompt(
    *,
    material: str,
    color: str,
    selected_slot: str,
    slot_by_slice: dict[str, str],
) -> str:
    """Render a six-object panel where only the selected slot determines the label."""

    slice_by_slot = {slot: slice_id for slice_id, slot in slot_by_slice.items()}
    objects: list[str] = []
    for slot in EXP003_SLOT_IDS:
        shape, size = slice_by_slot[slot].split("_", maxsplit=1)
        objects.append(f"{slot}:shape={shape},size={size}")
    return (
        "Classify only the selected synthetic object as ACCEPT or REJECT. "
        f"material={material}; color={color}; selected_slot={selected_slot}. "
        + "; ".join(objects)
        + ". Reply with exactly one label."
    )


def _build_exp003_examples(
    *,
    materials: tuple[str, ...],
    seed: int,
    prefix: str,
    plan: Exp003Plan,
) -> list[Exp003TaskExample]:
    """Build one role-binding example for every selected slice in every panel."""

    shard_by_slice = {
        slice_id: shard_id for shard_id, slice_id in plan.selected_slice_by_shard.items()
    }
    all_slices = tuple(_slice_id(shape, size) for shape in _SHAPES for size in _SIZES)
    examples: list[Exp003TaskExample] = []
    panel_index = 0
    for material in materials:
        for color in _COLORS:
            slot_by_slice = _exp003_slot_by_slice(panel_index)
            for slice_index, selected_slice_id in enumerate(all_slices):
                shape, _ = selected_slice_id.split("_", maxsplit=1)
                selected_slot = slot_by_slice[selected_slice_id]
                examples.append(
                    Exp003TaskExample(
                        example_id=_exp003_opaque_id(
                            seed=seed,
                            prefix=prefix,
                            panel_index=panel_index,
                            slice_index=slice_index,
                        ),
                        prompt=_exp003_prompt(
                            material=material,
                            color=color,
                            selected_slot=selected_slot,
                            slot_by_slice=slot_by_slice,
                        ),
                        response=_canonical_response(shape),
                        shard_id=shard_by_slice.get(selected_slice_id, "shard_stable_00"),
                        selected_slice_id=selected_slice_id,
                        material=material,
                        color=color,
                        selected_slot=selected_slot,
                        panel_index=panel_index,
                    )
                )
            panel_index += 1
    return examples


def build_exp003_data(seed: int = 42) -> Exp003Data:
    """Build Exp003 so changed-record lexical and count shortcuts tie.

    Every prompt contains all six shape-size descriptions. Candidate membership
    is determined only by which slot is selected, so bag-of-token similarity to
    the target cannot distinguish candidates. Each candidate contains one record
    per training panel and flips the first 36 panels, yielding exact slot balance
    among changed records. The intervention restores only the hidden target shard.
    """

    plan = build_exp003_plan(seed)
    baseline = _build_exp003_examples(
        materials=_TRAIN_MATERIALS,
        seed=seed,
        prefix="train",
        plan=plan,
    )

    candidate: list[Exp003TaskExample] = []
    intervention: list[Exp003TaskExample] = []
    for example in baseline:
        is_candidate = example.shard_id in EXP003_SHARD_IDS
        should_flip = is_candidate and example.panel_index < EXP003_LABEL_CHANGES_PER_SHARD
        candidate_response = (
            _flipped_response(example.response) if should_flip else example.response
        )
        candidate_example = replace(example, response=candidate_response)
        candidate.append(candidate_example)

        restore_target = should_flip and example.shard_id == plan.root_cause_id
        intervention.append(
            replace(
                candidate_example,
                response=example.response if restore_target else candidate_response,
            )
        )

    eval_examples = _build_exp003_examples(
        materials=_EVAL_MATERIALS,
        seed=seed,
        prefix="eval",
        plan=plan,
    )
    eval_examples = [replace(example, shard_id="eval") for example in eval_examples]
    target_eval = [
        example for example in eval_examples if example.selected_slice_id == TARGET_SLICE_ID
    ]
    control_eval = [
        example for example in eval_examples if example.selected_slice_id == EXP003_CONTROL_SLICE_ID
    ]

    return Exp003Data(
        baseline_train=_shuffled(baseline, seed),
        candidate_train=_shuffled(candidate, seed),
        intervention_train=_shuffled(intervention, seed),
        target_eval=_shuffled(target_eval, seed),
        control_eval=_shuffled(control_eval, seed),
        all_eval=_shuffled(eval_examples, seed),
    )


def _exp003c_all_patterns() -> tuple[tuple[str, ...], ...]:
    """Return all balanced three-ACCEPT slot patterns in lexical order."""

    return tuple(itertools.combinations(EXP003C_SLOT_IDS, 3))


def _exp003c_pattern_id(accept_slots: tuple[str, ...]) -> str:
    """Return an internal deterministic identifier for one decision pattern."""

    return "pattern_" + "".join(slot[-1] for slot in accept_slots)


def _exp003c_opaque_id(
    *, seed: int, split: str, pattern_index: int, context_index: int, slot_index: int
) -> str:
    """Create an opaque deterministic example identifier for Exp003-C."""

    payload = f"{seed ^ _EXP003C_ID_SALT}:{split}:{pattern_index}:{context_index}:{slot_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"rec_{digest}"


def _exp003c_prompt(*, context_id: str, selected_slot: str, accept_slots: tuple[str, ...]) -> str:
    """Render a balanced six-slot decision lookup prompt."""

    assignments = "; ".join(
        f"{slot}:decision={'ACCEPT' if slot in accept_slots else 'REJECT'}"
        for slot in EXP003C_SLOT_IDS
    )
    return (
        "Return the decision assigned to the selected slot. "
        f"context={context_id}; selected_slot={selected_slot}. "
        f"{assignments}. Reply with exactly one label."
    )


def _build_exp003c_examples(
    *,
    patterns: tuple[tuple[str, ...], ...],
    contexts: tuple[str, ...],
    seed: int,
    split: str,
) -> list[Exp003CLookupExample]:
    """Build every selected-slot query for every pattern/context combination."""

    examples: list[Exp003CLookupExample] = []
    for pattern_index, accept_slots in enumerate(patterns):
        pattern_id = _exp003c_pattern_id(accept_slots)
        for context_index, context_id in enumerate(contexts):
            for slot_index, selected_slot in enumerate(EXP003C_SLOT_IDS):
                examples.append(
                    Exp003CLookupExample(
                        example_id=_exp003c_opaque_id(
                            seed=seed,
                            split=split,
                            pattern_index=pattern_index,
                            context_index=context_index,
                            slot_index=slot_index,
                        ),
                        prompt=_exp003c_prompt(
                            context_id=context_id,
                            selected_slot=selected_slot,
                            accept_slots=accept_slots,
                        ),
                        response="ACCEPT" if selected_slot in accept_slots else "REJECT",
                        selected_slot=selected_slot,
                        accept_slots=accept_slots,
                        context_id=context_id,
                        pattern_id=pattern_id,
                    )
                )
    return examples


def build_exp003c_lookup_data(seed: int = 42) -> Exp003CLookupData:
    """Build a balanced capability diagnostic for selected-slot lookup.

    Four balanced decision patterns are held out completely from training. Each
    prompt contains exactly three ACCEPT and three REJECT assignments. Every
    slot is selected equally often and is label-balanced within both train and
    evaluation splits, preventing constant-label and slot-prior shortcuts.
    """

    all_patterns = _exp003c_all_patterns()
    eval_patterns = EXP003C_EVAL_ACCEPT_SLOT_PATTERNS
    if any(pattern not in all_patterns for pattern in eval_patterns):
        raise ValueError("Experiment 003-C eval pattern is not a valid balanced pattern")
    train_patterns = tuple(pattern for pattern in all_patterns if pattern not in eval_patterns)

    train = _build_exp003c_examples(
        patterns=train_patterns,
        contexts=EXP003C_TRAIN_CONTEXTS,
        seed=seed,
        split="train",
    )
    eval_examples = _build_exp003c_examples(
        patterns=eval_patterns,
        contexts=EXP003C_EVAL_CONTEXTS,
        seed=seed,
        split="eval",
    )
    eval_by_slot = {
        slot: tuple(example for example in eval_examples if example.selected_slot == slot)
        for slot in EXP003C_SLOT_IDS
    }

    rng = random.Random(seed)
    rng.shuffle(train)
    rng = random.Random(seed)
    rng.shuffle(eval_examples)

    return Exp003CLookupData(
        baseline_train=tuple(train),
        all_eval=tuple(eval_examples),
        eval_by_slot=eval_by_slot,
        train_patterns=train_patterns,
        eval_patterns=eval_patterns,
    )


def _exp003d_add_explicit_policy(example: Exp003TaskExample) -> Exp003TaskExample:
    """Prepend the canonical shape policy without changing any other Exp003 field."""

    return replace(example, prompt=f"{EXP003D_POLICY_TEXT} {example.prompt}")


def build_exp003d_explicit_policy_data(seed: int = 42) -> Exp003DExplicitPolicyData:
    """Build the one-factor explicit-policy follow-up to the Exp003 clean task.

    The exact Exp003 clean training and evaluation records are reused in the same
    order. IDs, responses, selected roles, nuisance attributes, and the original
    six-object panel text are unchanged. The sole model-visible change is an
    explicit canonical shape-to-label policy prepended to every prompt.
    """

    source = build_exp003_data(seed)
    baseline = tuple(_exp003d_add_explicit_policy(example) for example in source.baseline_train)
    eval_examples = tuple(_exp003d_add_explicit_policy(example) for example in source.all_eval)
    eval_by_slice = {
        slice_id: tuple(
            example for example in eval_examples if example.selected_slice_id == slice_id
        )
        for slice_id in EXP003D_SLICE_IDS
    }

    return Exp003DExplicitPolicyData(
        baseline_train=baseline,
        all_eval=eval_examples,
        eval_by_slice=eval_by_slice,
    )


def build_exp004_plan(seed: int = 42) -> Exp004Plan:
    """Build the deterministic benchmark-private Experiment 004 shard plan."""

    shard_ids = list(EXP004_SHARD_IDS)
    random.Random(seed ^ _EXP004_PLAN_SALT).shuffle(shard_ids)

    selected_slice_by_shard = dict(zip(shard_ids, EXP004_CANDIDATE_SLICES, strict=True))

    root_cause_id = next(
        shard_id
        for shard_id, slice_id in selected_slice_by_shard.items()
        if slice_id == TARGET_SLICE_ID
    )

    return Exp004Plan(
        root_cause_id=root_cause_id,
        selected_slice_by_shard=selected_slice_by_shard,
    )


def _exp004_assign_shard(
    example: Exp003TaskExample,
    *,
    plan: Exp004Plan,
) -> Exp003TaskExample:
    """Assign fresh opaque Exp004 lineage without changing model-facing fields."""

    shard_by_slice = {
        slice_id: shard_id for shard_id, slice_id in plan.selected_slice_by_shard.items()
    }

    return replace(
        example,
        shard_id=shard_by_slice.get(
            example.selected_slice_id,
            "shard_stable_00",
        ),
    )


def build_exp004_data(seed: int = 42) -> Exp004Data:
    """Build the frozen explicit-policy five-candidate Experiment 004 benchmark."""

    source = build_exp003d_explicit_policy_data(seed)
    plan = build_exp004_plan(seed)

    baseline = tuple(_exp004_assign_shard(example, plan=plan) for example in source.baseline_train)

    candidate: list[Exp003TaskExample] = []

    for example in baseline:
        should_flip = (
            example.shard_id in EXP004_SHARD_IDS
            and example.panel_index < EXP004_LABEL_CHANGES_PER_SHARD
        )

        candidate.append(
            replace(
                example,
                response=(_flipped_response(example.response) if should_flip else example.response),
            )
        )

    eval_examples = source.all_eval

    eval_by_slice = {
        slice_id: tuple(
            example for example in eval_examples if example.selected_slice_id == slice_id
        )
        for slice_id in EXP004_SLICE_IDS
    }

    return Exp004Data(
        baseline_train=baseline,
        candidate_train=tuple(candidate),
        target_eval=eval_by_slice[TARGET_SLICE_ID],
        control_eval=eval_by_slice[EXP004_CONTROL_SLICE_ID],
        all_eval=eval_examples,
        eval_by_slice=eval_by_slice,
    )


def build_exp004_intervention_train(
    intervention_candidate_id: str,
    *,
    seed: int = 42,
) -> tuple[Exp003TaskExample, ...]:
    """Restore only the diagnosis-selected candidate without using private truth."""

    if intervention_candidate_id not in EXP004_SHARD_IDS:
        raise ValueError(f"Unknown Experiment 004 candidate: {intervention_candidate_id}")

    data = build_exp004_data(seed)
    intervention: list[Exp003TaskExample] = []

    for baseline, candidate in zip(
        data.baseline_train,
        data.candidate_train,
        strict=True,
    ):
        restore = (
            baseline.shard_id == intervention_candidate_id
            and baseline.response != candidate.response
        )

        intervention.append(
            replace(
                candidate,
                response=baseline.response if restore else candidate.response,
            )
        )

    return tuple(intervention)


def derive_exp005_world_seed(seed: int, attempt_index: int) -> int:
    """Derive one frozen private world seed from the protocol namespace."""

    if not 0 <= attempt_index < EXP005_MAX_WORLD_ATTEMPTS:
        raise ValueError(
            f"Experiment 005 attempt index must be in [0, {EXP005_MAX_WORLD_ATTEMPTS - 1}]"
        )

    payload = f"exp005-world|{seed}|{attempt_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_exp005_plan(seed: int = 42, attempt_index: int = 0) -> Exp005Plan:
    """Build one deterministic benchmark-private Experiment 005 world plan."""

    world_seed = derive_exp005_world_seed(seed, attempt_index)
    candidate_ids = list(EXP005_SHARD_IDS)
    random.Random(world_seed).shuffle(candidate_ids)

    return Exp005Plan(
        attempt_index=attempt_index,
        world_seed=world_seed,
        planted_candidate_id=candidate_ids[0],
    )


def _exp005_hash_sorted(
    examples: list[Exp003TaskExample],
    *,
    world_seed: int,
    namespace: str,
) -> list[Exp003TaskExample]:
    """Return examples in a deterministic world-private hash order."""

    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"{namespace}|{world_seed}|{example.example_id}".encode()
        ).hexdigest(),
    )


def _exp005_target_color_slot_pairs(
    eval_examples: tuple[Exp003TaskExample, ...],
) -> tuple[tuple[str, str], ...]:
    """Return unique public color/slot pairs occurring in target evaluation."""

    pairs = {
        (example.color, example.selected_slot)
        for example in eval_examples
        if example.selected_slice_id == TARGET_SLICE_ID
    }
    return tuple(sorted(pairs))


def _exp005_changed_ids_by_candidate(
    source: Exp003DExplicitPolicyData,
    *,
    plan: Exp005Plan,
) -> dict[str, frozenset[str]]:
    """Choose balanced changed records while equalizing lexical max-overlap coverage."""

    train = list(source.baseline_train)
    target_pairs = _exp005_target_color_slot_pairs(source.all_eval)

    if len(target_pairs) != 12:
        raise ValueError("Experiment 005 requires exactly 12 target color/slot pairs")

    pairs_per_slot = {
        slot: sum(pair_slot == slot for _, pair_slot in target_pairs) for slot in EXP005_SLOT_IDS
    }
    if set(pairs_per_slot.values()) != {2}:
        raise ValueError("Experiment 005 target pairs must cover every selected slot twice")

    changed: dict[str, set[str]] = {candidate_id: set() for candidate_id in EXP005_SHARD_IDS}
    used: set[str] = set()

    distractor_ids = [
        candidate_id
        for candidate_id in EXP005_SHARD_IDS
        if candidate_id != plan.planted_candidate_id
    ]

    # For every target eval color/slot pair:
    # - the planted shard gets one changed ACCEPT example whose selected role is target;
    # - each distractor gets one changed ACCEPT example with the same public color/slot
    #   but a non-target ACCEPT selected role.
    # This gives all candidates identical lexical max-overlap coverage while preserving
    # a unique selected-role association for the planted shard.
    for color, slot in target_pairs:
        planted_pool = [
            example
            for example in train
            if example.example_id not in used
            and example.color == color
            and example.selected_slot == slot
            and example.response == "ACCEPT"
            and example.selected_slice_id == TARGET_SLICE_ID
        ]
        planted_pool = _exp005_hash_sorted(
            planted_pool,
            world_seed=plan.world_seed,
            namespace=f"planted-accept|{color}|{slot}",
        )
        if not planted_pool:
            raise ValueError("Experiment 005 could not allocate planted target coverage")
        planted = planted_pool[0]
        changed[plan.planted_candidate_id].add(planted.example_id)
        used.add(planted.example_id)

        distractor_pool = [
            example
            for example in train
            if example.example_id not in used
            and example.color == color
            and example.selected_slot == slot
            and example.response == "ACCEPT"
            and example.selected_slice_id != TARGET_SLICE_ID
        ]
        distractor_pool = _exp005_hash_sorted(
            distractor_pool,
            world_seed=plan.world_seed,
            namespace=f"distractor-accept|{color}|{slot}",
        )
        if len(distractor_pool) < len(distractor_ids):
            raise ValueError("Experiment 005 lacks non-target ACCEPT coverage")

        ordered_distractors = sorted(
            distractor_ids,
            key=lambda candidate_id: hashlib.sha256(
                f"distractor-order|{plan.world_seed}|{color}|{slot}|{candidate_id}".encode()
            ).hexdigest(),
        )
        for candidate_id, example in zip(
            ordered_distractors,
            distractor_pool[: len(ordered_distractors)],
            strict=True,
        ):
            changed[candidate_id].add(example.example_id)
            used.add(example.example_id)

    # Every candidate currently has 12 ACCEPT->REJECT changes, two per slot.
    # Add exactly two REJECT->ACCEPT examples per slot for every candidate.
    for slot in EXP005_SLOT_IDS:
        reject_pool = [
            example
            for example in train
            if example.example_id not in used
            and example.selected_slot == slot
            and example.response == "REJECT"
        ]
        reject_pool = _exp005_hash_sorted(
            reject_pool,
            world_seed=plan.world_seed,
            namespace=f"reject|{slot}",
        )
        required = 2 * len(EXP005_SHARD_IDS)
        if len(reject_pool) < required:
            raise ValueError("Experiment 005 lacks REJECT records for balanced corruption")

        candidate_order = sorted(
            EXP005_SHARD_IDS,
            key=lambda candidate_id: hashlib.sha256(
                f"reject-order|{plan.world_seed}|{slot}|{candidate_id}".encode()
            ).hexdigest(),
        )
        cursor = 0
        for candidate_id in candidate_order:
            for example in reject_pool[cursor : cursor + 2]:
                changed[candidate_id].add(example.example_id)
                used.add(example.example_id)
            cursor += 2

    frozen = {candidate_id: frozenset(example_ids) for candidate_id, example_ids in changed.items()}

    baseline_by_id = {example.example_id: example for example in source.baseline_train}
    for _candidate_id, example_ids in frozen.items():
        examples = [baseline_by_id[example_id] for example_id in example_ids]
        if len(examples) != EXP005_LABEL_CHANGES_PER_SHARD:
            raise ValueError("Experiment 005 changed-record count invariant failed")
        if sum(example.response == "ACCEPT" for example in examples) != 12:
            raise ValueError("Experiment 005 ACCEPT->REJECT balance invariant failed")
        if sum(example.response == "REJECT" for example in examples) != 12:
            raise ValueError("Experiment 005 REJECT->ACCEPT balance invariant failed")
        slot_counts = {
            slot: sum(example.selected_slot == slot for example in examples)
            for slot in EXP005_SLOT_IDS
        }
        if set(slot_counts.values()) != {4}:
            raise ValueError("Experiment 005 changed-slot balance invariant failed")

    return frozen


def build_exp005_data(seed: int = 42, attempt_index: int = 0) -> Exp005Data:
    """Build one balanced, blinded Experiment 005 candidate world."""

    source = build_exp003d_explicit_policy_data(seed)
    plan = build_exp005_plan(seed, attempt_index)
    changed_ids_by_candidate = _exp005_changed_ids_by_candidate(source, plan=plan)

    changed_owner = {
        example_id: candidate_id
        for candidate_id, example_ids in changed_ids_by_candidate.items()
        for example_id in example_ids
    }
    if len(changed_owner) != (len(EXP005_SHARD_IDS) * EXP005_LABEL_CHANGES_PER_SHARD):
        raise ValueError("Experiment 005 changed sets overlap")

    remaining = [
        example for example in source.baseline_train if example.example_id not in changed_owner
    ]
    remaining = _exp005_hash_sorted(
        remaining,
        world_seed=plan.world_seed,
        namespace="unchanged-fillers",
    )

    filler_owner: dict[str, str] = {}
    cursor = 0
    for candidate_id in sorted(EXP005_SHARD_IDS):
        for example in remaining[cursor : cursor + 24]:
            filler_owner[example.example_id] = candidate_id
        cursor += 24

    if len(filler_owner) != 24 * len(EXP005_SHARD_IDS):
        raise ValueError("Experiment 005 filler allocation failed")

    baseline: list[Exp003TaskExample] = []
    candidate: list[Exp003TaskExample] = []

    for example in source.baseline_train:
        shard_id = changed_owner.get(
            example.example_id,
            filler_owner.get(example.example_id, "shard_stable_00"),
        )
        baseline_example = replace(example, shard_id=shard_id)
        baseline.append(baseline_example)

        should_flip = example.example_id in changed_owner
        candidate.append(
            replace(
                baseline_example,
                response=(
                    _flipped_response(baseline_example.response)
                    if should_flip
                    else baseline_example.response
                ),
            )
        )

    eval_examples = source.all_eval
    eval_by_slice = {
        slice_id: tuple(
            example for example in eval_examples if example.selected_slice_id == slice_id
        )
        for slice_id in EXP005_SLICE_IDS
    }

    return Exp005Data(
        baseline_train=tuple(baseline),
        candidate_train=tuple(candidate),
        target_eval=eval_by_slice[TARGET_SLICE_ID],
        control_eval=eval_by_slice[EXP005_CONTROL_SLICE_ID],
        all_eval=eval_examples,
        eval_by_slice=eval_by_slice,
    )


def build_exp005_restoration_train(
    restoration_candidate_id: str,
    *,
    seed: int = 42,
    attempt_index: int = 0,
) -> tuple[Exp003TaskExample, ...]:
    """Restore exactly one candidate shard for private certification or intervention."""

    if restoration_candidate_id not in EXP005_SHARD_IDS:
        raise ValueError(f"Unknown Experiment 005 candidate: {restoration_candidate_id}")

    data = build_exp005_data(seed, attempt_index)
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
