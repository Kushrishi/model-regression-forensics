# Exp009 structurally matched benchmark protocol amendment 1

**Status:** FROZEN after clean-only structural-capacity preflight and before any matched-benchmark model training  
**Date:** 2026-09-25  
**Base protocol:** `STRUCTURALLY_MATCHED_BENCHMARK_PROTOCOL.md`  
**Observed matched-benchmark model outcomes at freeze:** none  
**Official Banking77 test split accessed at freeze:** no

This amendment changes only the number of matched benchmark worlds.

## Trigger

The base protocol prospectively requested three worlds with five globally
intent-disjoint candidate pairs per world, requiring 15 pairwise-disjoint
eligible pairs.

The first hosted structural preflight, workflow run `36211223915`, failed
before any model training because the sequential clean-only selector could not
fill the third world.

A separate clean-only graph-capacity diagnostic, workflow run
`36211324469`, then measured the eligible-pair graph without changing any
eligibility criterion:

- eligible edges: **81**;
- eligible vertices: **27**;
- disjoint pairs under the frozen rank-greedy selector: **11**;
- maximum-cardinality matching: **13**;
- disjoint pairs required for three complete worlds: **15**.

Therefore three complete five-candidate worlds are impossible under the frozen
clean-evidence eligibility rules, independent of the greedy allocator.

The machine-readable record is preserved in
`MATCHED_BENCHMARK_CAPACITY_PREFLIGHT.json`.

## Amendment

Freeze the M3 benchmark at:

```text
world_count = 2
candidates_per_world = 5
total candidate pairs = 10
total unique candidate-touched intents = 20
```

All other base-protocol rules remain unchanged.

In particular, do **not**:

- lower the minimum clean-recall threshold;
- lower training/evaluation count requirements;
- weaken the clean semantic-adjacency rule;
- reuse an intent label across worlds;
- change the 66-slot / 33-per-direction symmetric label-swap mechanism;
- replace structurally inconvenient pairs after model outcomes;
- access the official Banking77 test split.

## Why two worlds rather than a looser three-world construction

The benchmark should preserve candidate quality and structural comparability
rather than force an arbitrary preferred world count.

Two complete worlds are supported by the original prospective selector and
leave additional eligible pair capacity unused. This gives two independent
matched release-change settings while preserving every candidate-construction
constraint.

The reduction in world count is a pre-training feasibility amendment, not a
response to localization, restoration, or certification outcomes.

## Statistical boundary

This amendment does not freeze confirmatory trajectory count, certification
thresholds, multiplicity treatment, protected-equivalence margins, or official
test access.

Those remain M5 decisions.

## Frozen markers

`EXP009_MATCHED_BENCHMARK_PROTOCOL_AMENDMENT_1=FROZEN`

`EXP009_MATCHED_WORLD_COUNT=2`

`EXP009_MATCHED_BENCHMARK_MODEL_TRAINING_AT_FREEZE=NO`

`EXP009_OFFICIAL_TEST_ACCESSED_AT_FREEZE=NO`
