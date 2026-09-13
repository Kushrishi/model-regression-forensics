from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace

from model_forensics.task import (
    _TRAIN_MATERIALS,
    EXP003_SLOT_IDS,
    EXP003D_SLICE_IDS,
    TARGET_SLICE_ID,
    Exp003TaskExample,
    build_exp003d_explicit_policy_data,
)

EXP008_FROZEN_SEED = 42
EXP008_FROZEN_MANIFEST_SHA256 = "d9f0ea89edd180aca89f617f8feb71fee6c92f93a3aedddff6f9b1fcc98368a8"
EXP008_WORLD_COUNT = 2

EXP008_SHARD_IDS = tuple(f"shard_selective_{index:02d}" for index in range(1, 6))
EXP008_RECORDS_PER_SHARD = 48
EXP008_CHANGES_PER_SHARD = 36

EXP008_PROTECTED_SLICES = (
    "circle_small",
    "circle_large",
    "triangle_small",
    "square_small",
    "square_large",
)

EXP008_PROTECTED_ACCEPT_SLICES = (
    "circle_small",
    "circle_large",
    "triangle_small",
)

EXP008_REJECT_SLICES = (
    "square_small",
    "square_large",
)

EXP008_NUISANCE_SOURCE_QUOTAS = {
    "circle_small": 6,
    "circle_large": 6,
    "triangle_small": 6,
    "square_small": 9,
    "square_large": 9,
}

EXP008_CONTROL_SLICE_ID = "square_small"


@dataclass(frozen=True)
class Exp008Plan:
    world_index: int
    world_seed: int
    planted_candidate_id: str


@dataclass(frozen=True)
class Exp008Data:
    baseline_train: tuple[Exp003TaskExample, ...]
    candidate_train: tuple[Exp003TaskExample, ...]
    target_eval: tuple[Exp003TaskExample, ...]
    control_eval: tuple[Exp003TaskExample, ...]
    all_eval: tuple[Exp003TaskExample, ...]
    eval_by_slice: dict[str, tuple[Exp003TaskExample, ...]]


@dataclass(frozen=True)
class _InterventionPlan:
    changed_ids_by_candidate: dict[str, frozenset[str]]
    donor_id_by_recipient_id: dict[str, str]


def exp008_internal_examples_sha256(
    examples: tuple[Exp003TaskExample, ...],
) -> str:
    """Hash the full internal task representation in stable sequence order."""

    payload = [asdict(example) for example in examples]
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def derive_exp008_world_seed(seed: int, world_index: int) -> int:
    if seed != EXP008_FROZEN_SEED:
        raise ValueError(
            f"Experiment 008 is frozen only for seed={EXP008_FROZEN_SEED}; got seed={seed}"
        )
    if not 0 <= world_index < EXP008_WORLD_COUNT:
        raise ValueError(f"Experiment 008 world index must be in [0, {EXP008_WORLD_COUNT - 1}]")

    payload = f"exp008-world|{seed}|{world_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _root_order(seed: int) -> tuple[str, ...]:
    candidate_ids = list(EXP008_SHARD_IDS)
    digest = hashlib.sha256(f"exp008-root-order|{seed}".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(candidate_ids)
    return tuple(candidate_ids)


def build_exp008_plan(
    *,
    seed: int = EXP008_FROZEN_SEED,
    world_index: int,
) -> Exp008Plan:
    world_seed = derive_exp008_world_seed(seed, world_index)
    return Exp008Plan(
        world_index=world_index,
        world_seed=world_seed,
        planted_candidate_id=_root_order(seed)[world_index],
    )


def _hash_sorted_examples(
    examples: list[Exp003TaskExample],
    *,
    world_seed: int,
    namespace: str,
    nonce: int = 0,
) -> list[Exp003TaskExample]:
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            (f"exp008|{namespace}|{nonce}|{world_seed}|{example.example_id}").encode()
        ).hexdigest(),
    )


def _hash_sorted_values(
    values: list[int],
    *,
    world_seed: int,
    namespace: str,
    nonce: int = 0,
) -> list[int]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"exp008|{namespace}|{nonce}|{world_seed}|{value}".encode()
        ).hexdigest(),
    )


def _balanced_root_ids(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    plan: Exp008Plan,
) -> frozenset[str]:
    target = [example for example in source_train if example.selected_slice_id == TARGET_SLICE_ID]
    if len(target) != 48:
        raise ValueError("Experiment 008 requires exactly 48 target training records")

    by_material = {
        material: [example for example in target if example.material == material]
        for material in _TRAIN_MATERIALS
    }
    if any(len(examples) != 4 for examples in by_material.values()):
        raise ValueError("Experiment 008 target material capacity invariant failed")

    material_order = sorted(
        _TRAIN_MATERIALS,
        key=lambda material: hashlib.sha256(
            f"exp008-root-material|{plan.world_seed}|{material}".encode()
        ).hexdigest(),
    )

    color_omissions: Counter[str] = Counter()
    slot_omissions: Counter[str] = Counter()
    omitted: list[Exp003TaskExample] = []

    def search(material_index: int) -> bool:
        if material_index == len(material_order):
            return all(
                color_omissions[color] == 3 for color in ("amber", "blue", "green", "violet")
            ) and all(slot_omissions[slot] == 2 for slot in EXP003_SLOT_IDS)

        material = material_order[material_index]
        options = _hash_sorted_examples(
            by_material[material],
            world_seed=plan.world_seed,
            namespace=f"root-omit|{material}",
        )

        for example in options:
            if color_omissions[example.color] >= 3:
                continue
            if slot_omissions[example.selected_slot] >= 2:
                continue

            color_omissions[example.color] += 1
            slot_omissions[example.selected_slot] += 1
            omitted.append(example)

            remaining = len(material_order) - material_index - 1
            colors_feasible = all(
                color_omissions[color] <= 3 and color_omissions[color] + remaining >= 3
                for color in ("amber", "blue", "green", "violet")
            )
            slots_feasible = all(
                slot_omissions[slot] <= 2 and slot_omissions[slot] + remaining >= 2
                for slot in EXP003_SLOT_IDS
            )

            if colors_feasible and slots_feasible and search(material_index + 1):
                return True

            omitted.pop()
            color_omissions[example.color] -= 1
            slot_omissions[example.selected_slot] -= 1

        return False

    if not search(0):
        raise ValueError("Experiment 008 could not solve exact root balance")

    omitted_ids = {example.example_id for example in omitted}
    selected = [example for example in target if example.example_id not in omitted_ids]

    if len(selected) != EXP008_CHANGES_PER_SHARD:
        raise ValueError("Experiment 008 root changed-record count invariant failed")

    material_counts = Counter(example.material for example in selected)
    color_counts = Counter(example.color for example in selected)
    slot_counts = Counter(example.selected_slot for example in selected)

    if set(material_counts) != set(_TRAIN_MATERIALS):
        raise ValueError("Experiment 008 root material coverage invariant failed")
    if set(material_counts.values()) != {3}:
        raise ValueError("Experiment 008 root material balance invariant failed")

    if set(color_counts) != {"amber", "blue", "green", "violet"}:
        raise ValueError("Experiment 008 root color coverage invariant failed")
    if set(color_counts.values()) != {9}:
        raise ValueError("Experiment 008 root color balance invariant failed")

    if set(slot_counts) != set(EXP003_SLOT_IDS):
        raise ValueError("Experiment 008 root slot coverage invariant failed")
    if set(slot_counts.values()) != {6}:
        raise ValueError("Experiment 008 root slot balance invariant failed")

    return frozenset(example.example_id for example in selected)


def _root_donor_map(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    root_ids: frozenset[str],
    plan: Exp008Plan,
) -> dict[str, str]:
    selected = [example for example in source_train if example.example_id in root_ids]
    ordered = _hash_sorted_examples(
        selected,
        world_seed=plan.world_seed,
        namespace="root-prompt-cycle",
    )

    if len(ordered) != EXP008_CHANGES_PER_SHARD:
        raise ValueError("Experiment 008 root donor-map size invariant failed")

    donors = ordered[1:] + ordered[:1]
    mapping = {
        recipient.example_id: donor.example_id
        for recipient, donor in zip(ordered, donors, strict=True)
    }

    by_id = {example.example_id: example for example in source_train}

    for recipient_id, donor_id in mapping.items():
        recipient = by_id[recipient_id]
        donor = by_id[donor_id]

        if recipient.selected_slice_id != TARGET_SLICE_ID:
            raise ValueError("Experiment 008 root recipient is not target")
        if donor.selected_slice_id != TARGET_SLICE_ID:
            raise ValueError("Experiment 008 root donor is not target")
        if recipient.prompt == donor.prompt:
            raise ValueError("Experiment 008 root prompt cycle contains a fixed point")

    if set(mapping) != set(root_ids):
        raise ValueError("Experiment 008 root donor map is incomplete")
    if set(mapping.values()) != set(root_ids):
        raise ValueError("Experiment 008 root donor map is not a permutation")

    return mapping


def _panel_representatives(
    source_train: tuple[Exp003TaskExample, ...],
) -> dict[int, Exp003TaskExample]:
    representatives: dict[int, Exp003TaskExample] = {}

    for example in source_train:
        representatives.setdefault(example.panel_index, example)

    if len(representatives) != 48:
        raise ValueError("Experiment 008 requires exactly 48 training panels")

    return representatives


def _nuisance_exclusion_schedule(
    *,
    candidate_id: str,
    plan: Exp008Plan,
) -> tuple[str, ...]:
    """Return the frozen exact 3/3/3 protected-ACCEPT exclusion schedule."""

    schedule = list(EXP008_PROTECTED_ACCEPT_SLICES) * 3
    digest = hashlib.sha256(
        f"exp008-exclusion-offset|{plan.world_seed}|{candidate_id}".encode()
    ).hexdigest()
    offset = int(digest[:8], 16) % len(schedule)
    return tuple(schedule[offset:] + schedule[:offset])


def _nuisance_slot_counts(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    candidate_id: str,
    panel_indices: tuple[int, ...],
    plan: Exp008Plan,
) -> Counter[str]:
    """Return the selected-slot distribution induced by one panel block."""

    if len(panel_indices) != 9:
        raise ValueError("Experiment 008 nuisance slot audit requires exactly nine panels")

    schedule = _nuisance_exclusion_schedule(
        candidate_id=candidate_id,
        plan=plan,
    )
    counts: Counter[str] = Counter()

    for position, panel_index in enumerate(panel_indices):
        panel = _panel_examples(
            source_train,
            panel_index=panel_index,
        )
        excluded = schedule[position]

        selected_slices = [
            slice_id for slice_id in EXP008_PROTECTED_ACCEPT_SLICES if slice_id != excluded
        ]
        selected_slices.extend(EXP008_REJECT_SLICES)

        if len(selected_slices) != 4:
            raise ValueError("Experiment 008 nuisance panel must select exactly four records")

        for slice_id in selected_slices:
            counts[panel[slice_id].selected_slot] += 1

    return counts


def _panel_blocks(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    plan: Exp008Plan,
) -> dict[str, tuple[int, ...]]:
    """Allocate disjoint nuisance panels satisfying every frozen balance gate."""

    representatives = _panel_representatives(source_train)
    panel_indices = sorted(representatives)

    nuisance_ids = [
        candidate_id
        for candidate_id in EXP008_SHARD_IDS
        if candidate_id != plan.planted_candidate_id
    ]

    # Each nonce defines one complete deterministic partition:
    #
    #   36 selected panels
    #   -> four disjoint nine-panel nuisance blocks.
    #
    # We accept the first prospectively constraint-satisfying partition.
    # This is construction-time solving, not behavioral seed search: no model
    # has been trained and model behavior is never consulted here.
    for nonce in range(100_000):
        ordered = _hash_sorted_values(
            panel_indices,
            world_seed=plan.world_seed,
            namespace="nuisance-panels",
            nonce=nonce,
        )
        selected = ordered[: 9 * len(nuisance_ids)]

        blocks = {
            candidate_id: tuple(selected[index * 9 : (index + 1) * 9])
            for index, candidate_id in enumerate(nuisance_ids)
        }

        all_panels = [panel_index for block in blocks.values() for panel_index in block]

        if len(all_panels) != 36:
            raise ValueError("Experiment 008 nuisance panel-count invariant failed")
        if len(set(all_panels)) != 36:
            raise ValueError("Experiment 008 nuisance panel blocks overlap")

        valid = True

        for candidate_id, block in blocks.items():
            reps = [representatives[panel_index] for panel_index in block]

            color_counts = Counter(example.color for example in reps)
            material_counts = Counter(example.material for example in reps)

            # Nine panels must cover all four colors with near-perfect balance.
            if set(color_counts) != {
                "amber",
                "blue",
                "green",
                "violet",
            }:
                valid = False
                break

            if min(color_counts.values()) < 2 or max(color_counts.values()) > 3:
                valid = False
                break

            # Prevent concentration in only a few materials.
            if len(material_counts) < 7:
                valid = False
                break

            if max(material_counts.values()) > 2:
                valid = False
                break

            # This was the missing prospective constraint in the live allocator.
            # Validate the exact records that _nuisance_intervention will select,
            # using the exact same candidate-specific exclusion schedule.
            slot_counts = _nuisance_slot_counts(
                source_train,
                candidate_id=candidate_id,
                panel_indices=block,
                plan=plan,
            )

            if set(slot_counts) != set(EXP003_SLOT_IDS):
                valid = False
                break

            if min(slot_counts.values()) < 4 or max(slot_counts.values()) > 8:
                valid = False
                break

        if valid:
            return blocks

    raise ValueError(
        "Experiment 008 could not allocate nuisance panels satisfying "
        "the frozen color/material/slot constraints"
    )


def _panel_examples(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    panel_index: int,
) -> dict[str, Exp003TaskExample]:
    examples = {
        example.selected_slice_id: example
        for example in source_train
        if example.panel_index == panel_index
    }

    if set(examples) != set(EXP003D_SLICE_IDS):
        raise ValueError("Experiment 008 panel does not contain all semantic slices")

    return examples


def _nuisance_intervention(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    candidate_id: str,
    panel_indices: tuple[int, ...],
    plan: Exp008Plan,
) -> tuple[frozenset[str], dict[str, str]]:
    if len(panel_indices) != 9:
        raise ValueError("Experiment 008 nuisance candidate requires nine panels")

    schedule = _nuisance_exclusion_schedule(
        candidate_id=candidate_id,
        plan=plan,
    )

    changed_ids: set[str] = set()
    donor_map: dict[str, str] = {}

    for position, panel_index in enumerate(panel_indices):
        panel = _panel_examples(source_train, panel_index=panel_index)
        excluded = schedule[position]

        accepts = [
            panel[slice_id] for slice_id in EXP008_PROTECTED_ACCEPT_SLICES if slice_id != excluded
        ]
        rejects = [
            panel["square_small"],
            panel["square_large"],
        ]

        pair_digest = hashlib.sha256(
            (f"exp008-pair-order|{plan.world_seed}|{candidate_id}|{panel_index}").encode()
        ).hexdigest()

        if int(pair_digest[:8], 16) % 2:
            accepts.reverse()

        for accept, reject in zip(accepts, rejects, strict=True):
            donor_map[accept.example_id] = reject.example_id
            donor_map[reject.example_id] = accept.example_id
            changed_ids.update((accept.example_id, reject.example_id))

    if len(changed_ids) != EXP008_CHANGES_PER_SHARD:
        raise ValueError("Experiment 008 nuisance changed-record count invariant failed")

    if set(donor_map) != changed_ids:
        raise ValueError("Experiment 008 nuisance donor-map key invariant failed")
    if set(donor_map.values()) != changed_ids:
        raise ValueError("Experiment 008 nuisance donor map is not a permutation")

    by_id = {example.example_id: example for example in source_train}
    selected = [by_id[example_id] for example_id in changed_ids]

    semantic_counts = Counter(example.selected_slice_id for example in selected)
    if semantic_counts != Counter(EXP008_NUISANCE_SOURCE_QUOTAS):
        raise ValueError("Experiment 008 nuisance source-semantic balance failed")

    response_counts = Counter(example.response for example in selected)
    if response_counts != {"ACCEPT": 18, "REJECT": 18}:
        raise ValueError("Experiment 008 nuisance response balance failed")

    slot_counts = Counter(example.selected_slot for example in selected)
    if set(slot_counts) != set(EXP003_SLOT_IDS):
        raise ValueError("Experiment 008 nuisance slot coverage failed")
    if min(slot_counts.values()) < 4 or max(slot_counts.values()) > 8:
        raise ValueError("Experiment 008 nuisance slot concentration failed")

    for recipient_id, donor_id in donor_map.items():
        recipient = by_id[recipient_id]
        donor = by_id[donor_id]

        if recipient.selected_slice_id == TARGET_SLICE_ID:
            raise ValueError("Experiment 008 nuisance recipient touched target")
        if donor.selected_slice_id == TARGET_SLICE_ID:
            raise ValueError("Experiment 008 nuisance donor touched target")
        if recipient.prompt == donor.prompt:
            raise ValueError("Experiment 008 nuisance prompt did not change")
        if recipient.response == donor.response:
            raise ValueError("Experiment 008 nuisance response did not change")

    return frozenset(changed_ids), donor_map


def _build_intervention_plan(
    source_train: tuple[Exp003TaskExample, ...],
    *,
    plan: Exp008Plan,
) -> _InterventionPlan:
    root_ids = _balanced_root_ids(source_train, plan=plan)

    changed_ids_by_candidate: dict[str, frozenset[str]] = {
        plan.planted_candidate_id: root_ids,
    }
    donor_id_by_recipient_id = _root_donor_map(
        source_train,
        root_ids=root_ids,
        plan=plan,
    )

    panel_blocks = _panel_blocks(source_train, plan=plan)

    for candidate_id, panel_indices in panel_blocks.items():
        changed_ids, donor_map = _nuisance_intervention(
            source_train,
            candidate_id=candidate_id,
            panel_indices=panel_indices,
            plan=plan,
        )

        for existing_ids in changed_ids_by_candidate.values():
            if existing_ids.intersection(changed_ids):
                raise ValueError("Experiment 008 changed-record sets overlap")

        if set(donor_id_by_recipient_id).intersection(donor_map):
            raise ValueError("Experiment 008 donor-map recipients overlap")

        changed_ids_by_candidate[candidate_id] = changed_ids
        donor_id_by_recipient_id.update(donor_map)

    if set(changed_ids_by_candidate) != set(EXP008_SHARD_IDS):
        raise ValueError("Experiment 008 intervention plan lacks candidate IDs")

    all_changed = set().union(*changed_ids_by_candidate.values())
    expected_total = len(EXP008_SHARD_IDS) * EXP008_CHANGES_PER_SHARD

    if len(all_changed) != expected_total:
        raise ValueError("Experiment 008 total changed-record ownership is invalid")
    if set(donor_id_by_recipient_id) != all_changed:
        raise ValueError("Experiment 008 donor map does not cover all changed records")

    return _InterventionPlan(
        changed_ids_by_candidate=changed_ids_by_candidate,
        donor_id_by_recipient_id=donor_id_by_recipient_id,
    )


def _copy_donor_content(
    recipient: Exp003TaskExample,
    donor: Exp003TaskExample,
    *,
    response: str,
) -> Exp003TaskExample:
    return replace(
        recipient,
        prompt=donor.prompt,
        response=response,
        selected_slice_id=donor.selected_slice_id,
        material=donor.material,
        color=donor.color,
        selected_slot=donor.selected_slot,
        panel_index=donor.panel_index,
    )


def build_exp008_data(
    *,
    seed: int = EXP008_FROZEN_SEED,
    world_index: int,
) -> Exp008Data:
    source = build_exp003d_explicit_policy_data(seed)
    plan = build_exp008_plan(seed=seed, world_index=world_index)
    interventions = _build_intervention_plan(
        source.baseline_train,
        plan=plan,
    )

    source_by_id = {example.example_id: example for example in source.baseline_train}

    changed_owner = {
        example_id: candidate_id
        for candidate_id, example_ids in interventions.changed_ids_by_candidate.items()
        for example_id in example_ids
    }

    remaining = [
        example for example in source.baseline_train if example.example_id not in changed_owner
    ]
    remaining = _hash_sorted_examples(
        remaining,
        world_seed=plan.world_seed,
        namespace="fillers",
    )

    filler_count = EXP008_RECORDS_PER_SHARD - EXP008_CHANGES_PER_SHARD
    filler_owner: dict[str, str] = {}
    cursor = 0

    for candidate_id in sorted(EXP008_SHARD_IDS):
        selected = remaining[cursor : cursor + filler_count]
        if len(selected) != filler_count:
            raise ValueError("Experiment 008 filler allocation failed")

        for example in selected:
            filler_owner[example.example_id] = candidate_id

        cursor += filler_count

    baseline: list[Exp003TaskExample] = []
    candidate: list[Exp003TaskExample] = []

    for example in source.baseline_train:
        candidate_id = changed_owner.get(
            example.example_id,
            filler_owner.get(example.example_id, "shard_stable_00"),
        )
        baseline_example = replace(example, shard_id=candidate_id)
        baseline.append(baseline_example)

        if example.example_id not in changed_owner:
            candidate.append(baseline_example)
            continue

        donor_id = interventions.donor_id_by_recipient_id[example.example_id]
        donor = source_by_id[donor_id]

        if candidate_id == plan.planted_candidate_id:
            transformed = _copy_donor_content(
                baseline_example,
                donor,
                response="REJECT",
            )
        else:
            transformed = _copy_donor_content(
                baseline_example,
                donor,
                response=donor.response,
            )

        candidate.append(transformed)

    for candidate_id in EXP008_SHARD_IDS:
        shard_size = sum(example.shard_id == candidate_id for example in baseline)
        if shard_size != EXP008_RECORDS_PER_SHARD:
            raise ValueError("Experiment 008 candidate shard-size invariant failed")

    eval_examples = source.all_eval
    eval_by_slice = {
        slice_id: tuple(
            example for example in eval_examples if example.selected_slice_id == slice_id
        )
        for slice_id in EXP003D_SLICE_IDS
    }

    data = Exp008Data(
        baseline_train=tuple(baseline),
        candidate_train=tuple(candidate),
        target_eval=eval_by_slice[TARGET_SLICE_ID],
        control_eval=eval_by_slice[EXP008_CONTROL_SLICE_ID],
        all_eval=eval_examples,
        eval_by_slice=eval_by_slice,
    )

    _validate_exp008_data(data, plan=plan)
    return data


def _policy_response(slice_id: str) -> str:
    shape, _ = slice_id.split("_", maxsplit=1)
    return "ACCEPT" if shape in {"circle", "triangle"} else "REJECT"


def summarize_exp008_interventions(
    data: Exp008Data,
) -> dict[str, object]:
    candidates: dict[str, dict[str, object]] = {}

    for candidate_id in EXP008_SHARD_IDS:
        pairs = [
            (baseline, candidate)
            for baseline, candidate in zip(
                data.baseline_train,
                data.candidate_train,
                strict=True,
            )
            if baseline.shard_id == candidate_id
            and baseline.to_sft_record() != candidate.to_sft_record()
        ]

        before_prompts = Counter(baseline.prompt for baseline, _ in pairs)
        after_prompts = Counter(candidate.prompt for _, candidate in pairs)

        before_pairs = Counter((baseline.prompt, baseline.response) for baseline, _ in pairs)
        after_pairs = Counter((candidate.prompt, candidate.response) for _, candidate in pairs)

        policy_inconsistent_by_slice = Counter(
            candidate.selected_slice_id
            for _, candidate in pairs
            if candidate.response != _policy_response(candidate.selected_slice_id)
        )

        candidates[candidate_id] = {
            "changed_records": len(pairs),
            "prompt_changes": sum(
                baseline.prompt != candidate.prompt for baseline, candidate in pairs
            ),
            "response_changes": sum(
                baseline.response != candidate.response for baseline, candidate in pairs
            ),
            "source_semantics": dict(
                sorted(Counter(baseline.selected_slice_id for baseline, _ in pairs).items())
            ),
            "result_semantics": dict(
                sorted(Counter(candidate.selected_slice_id for _, candidate in pairs).items())
            ),
            "source_responses": dict(
                sorted(Counter(baseline.response for baseline, _ in pairs).items())
            ),
            "result_responses": dict(
                sorted(Counter(candidate.response for _, candidate in pairs).items())
            ),
            "prompt_multiset_preserved": before_prompts == after_prompts,
            "prompt_response_multiset_preserved": before_pairs == after_pairs,
            "policy_inconsistent_by_slice": dict(sorted(policy_inconsistent_by_slice.items())),
        }

    aggregate_policy_inconsistent = Counter(
        candidate.selected_slice_id
        for baseline, candidate in zip(
            data.baseline_train,
            data.candidate_train,
            strict=True,
        )
        if baseline.to_sft_record() != candidate.to_sft_record()
        and candidate.response != _policy_response(candidate.selected_slice_id)
    )

    baseline_prompt_counts = Counter(example.prompt for example in data.baseline_train)
    candidate_prompt_counts = Counter(example.prompt for example in data.candidate_train)

    return {
        "candidates": candidates,
        "aggregate_policy_inconsistent_by_slice": dict(
            sorted(aggregate_policy_inconsistent.items())
        ),
        "baseline_semantics": dict(
            sorted(Counter(example.selected_slice_id for example in data.baseline_train).items())
        ),
        "candidate_semantics": dict(
            sorted(Counter(example.selected_slice_id for example in data.candidate_train).items())
        ),
        "baseline_responses": dict(
            sorted(Counter(example.response for example in data.baseline_train).items())
        ),
        "candidate_responses": dict(
            sorted(Counter(example.response for example in data.candidate_train).items())
        ),
        "global_prompt_multiset_preserved": (baseline_prompt_counts == candidate_prompt_counts),
        "baseline_unique_prompt_count": len(baseline_prompt_counts),
        "candidate_unique_prompt_count": len(candidate_prompt_counts),
    }


def _validate_exp008_data(
    data: Exp008Data,
    *,
    plan: Exp008Plan,
) -> None:
    if len(data.baseline_train) != 288:
        raise ValueError("Experiment 008 baseline training-size invariant failed")
    if len(data.candidate_train) != 288:
        raise ValueError("Experiment 008 candidate training-size invariant failed")

    summary = summarize_exp008_interventions(data)

    if not summary["global_prompt_multiset_preserved"]:
        raise ValueError("Experiment 008 global prompt multiset changed")

    if summary["baseline_unique_prompt_count"] != 288:
        raise ValueError("Experiment 008 baseline prompts are not unique")
    if summary["candidate_unique_prompt_count"] != 288:
        raise ValueError("Experiment 008 candidate introduced prompt duplication")

    expected_semantics = {slice_id: 48 for slice_id in EXP003D_SLICE_IDS}

    if summary["baseline_semantics"] != expected_semantics:
        raise ValueError("Experiment 008 baseline semantic invariant failed")
    if summary["candidate_semantics"] != expected_semantics:
        raise ValueError("Experiment 008 candidate semantic invariant failed")

    if summary["baseline_responses"] != {
        "ACCEPT": 192,
        "REJECT": 96,
    }:
        raise ValueError("Experiment 008 baseline label-mass invariant failed")

    if summary["candidate_responses"] != {
        "ACCEPT": 156,
        "REJECT": 132,
    }:
        raise ValueError("Experiment 008 candidate label-mass invariant failed")

    candidates = summary["candidates"]
    if not isinstance(candidates, dict):
        raise TypeError("Experiment 008 candidate summary is malformed")

    expected_nuisance_semantics = dict(sorted(EXP008_NUISANCE_SOURCE_QUOTAS.items()))

    for candidate_id in EXP008_SHARD_IDS:
        candidate_summary = candidates[candidate_id]

        if candidate_summary["changed_records"] != EXP008_CHANGES_PER_SHARD:
            raise ValueError("Experiment 008 changed-record invariant failed")
        if candidate_summary["prompt_changes"] != EXP008_CHANGES_PER_SHARD:
            raise ValueError("Experiment 008 prompt-change invariant failed")
        if candidate_summary["response_changes"] != EXP008_CHANGES_PER_SHARD:
            raise ValueError("Experiment 008 response-change invariant failed")
        if not candidate_summary["prompt_multiset_preserved"]:
            raise ValueError("Experiment 008 candidate prompt multiset was not preserved")

        inconsistent = candidate_summary["policy_inconsistent_by_slice"]

        if candidate_id == plan.planted_candidate_id:
            if candidate_summary["source_semantics"] != {TARGET_SLICE_ID: EXP008_CHANGES_PER_SHARD}:
                raise ValueError("Experiment 008 root source-semantic invariant failed")

            if candidate_summary["result_semantics"] != {TARGET_SLICE_ID: EXP008_CHANGES_PER_SHARD}:
                raise ValueError("Experiment 008 root result-semantic invariant failed")

            if candidate_summary["source_responses"] != {"ACCEPT": 36}:
                raise ValueError("Experiment 008 root source-response invariant failed")

            if candidate_summary["result_responses"] != {"REJECT": 36}:
                raise ValueError("Experiment 008 root result-response invariant failed")

            if inconsistent != {TARGET_SLICE_ID: EXP008_CHANGES_PER_SHARD}:
                raise ValueError("Experiment 008 root policy-inconsistency invariant failed")

            if candidate_summary["prompt_response_multiset_preserved"]:
                raise ValueError("Experiment 008 root unexpectedly preserved labels")
        else:
            if candidate_summary["source_semantics"] != expected_nuisance_semantics:
                raise ValueError("Experiment 008 nuisance source-semantic invariant failed")

            if candidate_summary["result_semantics"] != expected_nuisance_semantics:
                raise ValueError("Experiment 008 nuisance result-semantic invariant failed")

            if candidate_summary["source_responses"] != {
                "ACCEPT": 18,
                "REJECT": 18,
            }:
                raise ValueError("Experiment 008 nuisance source-response invariant failed")

            if candidate_summary["result_responses"] != {
                "ACCEPT": 18,
                "REJECT": 18,
            }:
                raise ValueError("Experiment 008 nuisance result-response invariant failed")

            if inconsistent:
                raise ValueError("Experiment 008 nuisance introduced incorrect policy")

            if not candidate_summary["prompt_response_multiset_preserved"]:
                raise ValueError("Experiment 008 nuisance changed its SFT content multiset")

    if summary["aggregate_policy_inconsistent_by_slice"] != {
        TARGET_SLICE_ID: EXP008_CHANGES_PER_SHARD
    }:
        raise ValueError(
            "Experiment 008 combined policy-inconsistent intervention invariant failed"
        )


def build_exp008_restoration_train(
    restoration_candidate_id: str,
    *,
    seed: int = EXP008_FROZEN_SEED,
    world_index: int,
) -> tuple[Exp003TaskExample, ...]:
    if restoration_candidate_id not in EXP008_SHARD_IDS:
        raise ValueError(f"Unknown Experiment 008 candidate: {restoration_candidate_id}")

    data = build_exp008_data(
        seed=seed,
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
            and baseline.to_sft_record() != candidate.to_sft_record()
        )
        restoration.append(baseline if restore else candidate)

    return tuple(restoration)
