from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from model_forensics.exp007 import (
    EXP007_FROZEN_SEED,
    EXP007_MAX_PAIRWISE_CHANGED_OVERLAP,
    EXP007_ROLE_IDS,
    EXP007_TARGET_DOSES,
    derive_exp007_world_seed,
)
from model_forensics.task import (
    _TRAIN_MATERIALS,
    EXP003_SLOT_IDS,
    TARGET_SLICE_ID,
    build_exp003d_explicit_policy_data,
)

COLORS = ("amber", "blue", "green", "violet")
ACCEPT_PROTECTED = ("circle_small", "circle_large", "triangle_small")
SQUARES = ("square_small", "square_large")
CANDIDATE_COUNT = 5


def _pair_signature(examples, i: int, j: int):
    pair = (examples[i], examples[j])
    return (
        tuple(sorted(x.selected_slice_id for x in pair)),
        tuple(sorted((x.response, x.selected_slot) for x in pair)),
        tuple(sorted((x.response, x.color) for x in pair)),
        tuple(sorted(x.material for x in pair)),
    )


def _valid_selection(examples, dose: int, chosen: list[set[int]]) -> bool:
    flat = [i for ids in chosen for i in ids]
    if len(flat) != 180 or len(set(flat)) != 180:
        return False

    histograms = []

    for candidate, ids in enumerate(chosen):
        selected = [examples[i] for i in ids]

        if len(selected) != 36:
            return False
        if Counter(x.response for x in selected) != {
            "ACCEPT": 24,
            "REJECT": 12,
        }:
            return False

        semantics = Counter(x.selected_slice_id for x in selected)

        if candidate == 0:
            protected_each = (24 - dose) // 3
            expected = {
                "circle_small": protected_each,
                "circle_large": protected_each,
                "square_small": 6,
                "square_large": 6,
                "triangle_small": protected_each,
                TARGET_SLICE_ID: dose,
            }
        else:
            expected = {
                "circle_small": 8,
                "circle_large": 8,
                "square_small": 6,
                "square_large": 6,
                "triangle_small": 8,
                TARGET_SLICE_ID: 0,
            }

        if any(semantics[key] != value for key, value in expected.items()):
            return False

        for slot in EXP003_SLOT_IDS:
            if sum(x.response == "ACCEPT" and x.selected_slot == slot for x in selected) != 4:
                return False
            if sum(x.response == "REJECT" and x.selected_slot == slot for x in selected) != 2:
                return False

        for color in COLORS:
            if sum(x.response == "ACCEPT" and x.color == color for x in selected) != 6:
                return False
            if sum(x.response == "REJECT" and x.color == color for x in selected) != 3:
                return False

        histogram = tuple(
            Counter(x.material for x in selected)[material] for material in _TRAIN_MATERIALS
        )
        if min(histogram) < 2 or max(histogram) > 4:
            return False
        histograms.append(histogram)

    return len(set(histograms)) == 1


def _base_solution(examples, dose: int) -> list[set[int]]:
    n = len(examples)
    nv = n * CANDIDATE_COUNT
    rr, cc, vv, lo, hi = [], [], [], [], []

    def var(i: int, candidate: int) -> int:
        return i * CANDIDATE_COUNT + candidate

    def add_terms(terms, low: int, high: int) -> None:
        row = len(lo)
        for index, coefficient in terms:
            rr.append(row)
            cc.append(index)
            vv.append(float(coefficient))
        lo.append(float(low))
        hi.append(float(high))

    def add(indices, low: int, high: int) -> None:
        add_terms([(index, 1.0) for index in indices], low, high)

    def query(candidate: int, predicate):
        return [var(i, candidate) for i, example in enumerate(examples) if predicate(example)]

    for i in range(n):
        add([var(i, c) for c in range(CANDIDATE_COUNT)], 0, 1)

    for candidate in range(CANDIDATE_COUNT):
        root = candidate == 0

        add(query(candidate, lambda x: True), 36, 36)
        add(query(candidate, lambda x: x.response == "ACCEPT"), 24, 24)
        add(query(candidate, lambda x: x.response == "REJECT"), 12, 12)

        for slice_id in SQUARES:
            add(
                query(
                    candidate,
                    lambda x, s=slice_id: x.selected_slice_id == s,
                ),
                6,
                6,
            )

        if root:
            add(
                query(
                    candidate,
                    lambda x: x.selected_slice_id == TARGET_SLICE_ID,
                ),
                dose,
                dose,
            )
            protected_each = (24 - dose) // 3
            for slice_id in ACCEPT_PROTECTED:
                add(
                    query(
                        candidate,
                        lambda x, s=slice_id: x.selected_slice_id == s,
                    ),
                    protected_each,
                    protected_each,
                )
        else:
            add(
                query(
                    candidate,
                    lambda x: x.selected_slice_id == TARGET_SLICE_ID,
                ),
                0,
                0,
            )
            for slice_id in ACCEPT_PROTECTED:
                add(
                    query(
                        candidate,
                        lambda x, s=slice_id: x.selected_slice_id == s,
                    ),
                    8,
                    8,
                )

        for slot in EXP003_SLOT_IDS:
            add(
                query(
                    candidate,
                    lambda x, slot=slot: x.response == "ACCEPT" and x.selected_slot == slot,
                ),
                4,
                4,
            )
            add(
                query(
                    candidate,
                    lambda x, slot=slot: x.response == "REJECT" and x.selected_slot == slot,
                ),
                2,
                2,
            )

        for color in COLORS:
            add(
                query(
                    candidate,
                    lambda x, color=color: x.response == "ACCEPT" and x.color == color,
                ),
                6,
                6,
            )
            add(
                query(
                    candidate,
                    lambda x, color=color: x.response == "REJECT" and x.color == color,
                ),
                3,
                3,
            )

        for material in _TRAIN_MATERIALS:
            add(
                query(
                    candidate,
                    lambda x, material=material: x.material == material,
                ),
                2,
                4,
            )

    for material in _TRAIN_MATERIALS:
        root_ids = query(0, lambda x, material=material: x.material == material)

        for candidate in range(1, CANDIDATE_COUNT):
            other_ids = query(
                candidate,
                lambda x, material=material: x.material == material,
            )
            terms = [(index, 1.0) for index in other_ids]
            terms += [(index, -1.0) for index in root_ids]
            add_terms(terms, 0, 0)

    matrix = coo_matrix(
        (vv, (rr, cc)),
        shape=(len(lo), nv),
    ).tocsr()

    result = milp(
        c=np.zeros(nv),
        integrality=np.ones(nv, dtype=int),
        bounds=Bounds(np.zeros(nv), np.ones(nv)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lo),
            np.asarray(hi),
        ),
        options={"presolve": True},
    )

    if not result.success:
        raise RuntimeError(f"dose {dose} base MILP failed: {result.message}")

    chosen = []

    for candidate in range(CANDIDATE_COUNT):
        chosen.append({i for i in range(n) if result.x[i * CANDIDATE_COUNT + candidate] > 0.5})

    if not _valid_selection(examples, dose, chosen):
        raise RuntimeError(f"dose {dose} base solution failed validation")

    return chosen


def _state_union(chosen: list[set[int]]) -> set[int]:
    return set().union(*chosen)


def _variant_signature(chosen: list[set[int]]) -> str:
    union = sorted(_state_union(chosen))
    return hashlib.sha256(",".join(map(str, union)).encode()).hexdigest()


def _one_exact_trade(
    examples,
    chosen: list[set[int]],
    *,
    dose: int,
    step: int,
) -> bool:
    used = _state_union(chosen)
    free = sorted(set(range(len(examples))) - used)

    free_pairs = defaultdict(list)
    for u, v in combinations(free, 2):
        free_pairs[_pair_signature(examples, u, v)].append((u, v))

    seed = int(
        hashlib.sha256(f"exp007-diversity-walk|{dose}|{step}".encode()).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)

    candidate_order = list(range(CANDIDATE_COUNT))
    rng.shuffle(candidate_order)

    for candidate in candidate_order:
        selected_pairs = list(combinations(sorted(chosen[candidate]), 2))
        rng.shuffle(selected_pairs)

        for a, b in selected_pairs:
            incoming = free_pairs.get(_pair_signature(examples, a, b))
            if not incoming:
                continue

            u, v = incoming[rng.randrange(len(incoming))]

            chosen[candidate].remove(a)
            chosen[candidate].remove(b)
            chosen[candidate].add(u)
            chosen[candidate].add(v)
            return True

    return False


def _build_variants(examples, dose: int) -> list[list[set[int]]]:
    chosen = [set(ids) for ids in _base_solution(examples, dose)]
    variants = [[set(ids) for ids in chosen]]
    seen = {_variant_signature(chosen)}

    for step in range(1, 2001):
        if not _one_exact_trade(examples, chosen, dose=dose, step=step):
            raise RuntimeError(f"dose {dose}: diversity walk has no valid trade at step {step}")

        if len(variants) == 7:
            break

        current = _state_union(chosen)
        previous = [_state_union(world) for world in variants]

        if any(len(current & prior) > EXP007_MAX_PAIRWISE_CHANGED_OVERLAP for prior in previous):
            continue

        signature = _variant_signature(chosen)
        if signature in seen:
            continue

        if not _valid_selection(examples, dose, chosen):
            raise RuntimeError(f"dose {dose}: diversity walk produced an invalid world")

        variants.append([set(ids) for ids in chosen])
        seen.add(signature)

    if len(variants) != 7:
        raise RuntimeError(f"dose {dose}: found only {len(variants)} diverse worlds")

    unions = [_state_union(world) for world in variants]
    overlaps = [len(a & b) for a, b in combinations(unions, 2)]

    if max(overlaps) > EXP007_MAX_PAIRWISE_CHANGED_OVERLAP:
        raise RuntimeError(f"dose {dose}: pairwise diversity gate failed")

    return variants


def _schedule():
    return (
        ("calibration", 0),
        ("calibration", 1),
        ("certification", 0),
        ("certification", 1),
        ("certification", 2),
        ("certification", 3),
        ("certification", 4),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the frozen Experiment 007 pre-model world manifest."
    )
    parser.add_argument(
        "--output",
        default="src/model_forensics/data/exp007_frozen_worlds.json",
    )
    args = parser.parse_args()

    output = Path(args.output)

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {output}")

    examples = list(build_exp003d_explicit_policy_data(EXP007_FROZEN_SEED).baseline_train)

    manifest = {
        "schema_version": 1,
        "seed": EXP007_FROZEN_SEED,
        "worlds": {},
    }

    for dose in EXP007_TARGET_DOSES:
        variants = _build_variants(examples, dose)

        for variant, (phase, world_index) in zip(
            variants,
            _schedule(),
            strict=True,
        ):
            world_seed = derive_exp007_world_seed(
                EXP007_FROZEN_SEED,
                phase,
                dose,
                world_index,
            )

            roles = list(EXP007_ROLE_IDS)
            random.Random(world_seed).shuffle(roles)

            changed_ids_by_role = {
                roles[candidate]: sorted(examples[i].example_id for i in variant[candidate])
                for candidate in range(CANDIDATE_COUNT)
            }

            key = f"{phase}:{dose}:{world_index}"
            manifest["worlds"][key] = {
                "phase": phase,
                "target_dose": dose,
                "world_index": world_index,
                "world_seed": world_seed,
                "root_role": roles[0],
                "changed_ids_by_role": changed_ids_by_role,
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"WORLDS={len(manifest['worlds'])}")
    print("MANIFEST_SHA256=" + hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
