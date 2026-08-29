from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
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
