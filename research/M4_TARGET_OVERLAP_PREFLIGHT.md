# M4 target-overlap preflight

**Status:** prospective, pre-training
**Date:** 2026-10-02
**Milestone:** M4 - competitive localization baselines
**Evidence class:** development only

## Purpose

Before training any model on the M3 structurally matched benchmark, M4 checks whether the localization problem is already easy from the known regressed behavior and debugger-visible version diffs.

This is a benchmark-difficulty audit, not a causal result.

The M3 benchmark deliberately matches candidate structure, but each candidate is a symmetric label swap over a distinct intent pair. The affected target intent pair is known to the debugger in the regression-forensics setting. A simple baseline may therefore identify the responsible candidate by checking which versioned change touches the target labels.

If that happens, preserve the result. Do not redesign the benchmark after seeing it merely to make localization look harder.

## Frozen inputs

Use the completed M3 matched-benchmark construction:

- 2 worlds;
- 5 opaque candidates per world;
- 66 changed stable slots per candidate;
- 33 label swaps in each direction;
- no text changes;
- pairwise-disjoint changed slots;
- pairwise-disjoint candidate intent labels;
- official Banking77 test split untouched.

The M3 result record is:

`experiments/009_stochastic_counterfactual_certification/MATCHED_BENCHMARK_PREFLIGHT_RESULT.json`

The target behavior for each world is the prospectively planted target intent pair. The target labels are debugger-visible for this preflight. The candidate root identity is not.

## Debugger-visible information

The blind baseline may use only:

- the target intent pair for the world;
- the clean baseline release;
- the composite release;
- the opaque diagnostic candidate manifest;
- each candidate's changed stable-slot IDs;
- baseline and composite labels at those changed slots.

It must not use:

- `truth_manifest.json`;
- `root_position` while ranking candidates;
- `internal_role`;
- restoration outcomes;
- model predictions from a matched-world composite;
- model gradients or checkpoints;
- official Banking77 test data.

## Frozen score

For candidate `C`, let `S_C` be its 66 changed slots and let `T={A,B}` be the known target intent pair.

For slot `s`, define:

```text
match(s) = 1 if {baseline_label(s), composite_label(s)} == T
           0 otherwise
```

The candidate score is:

```text
score(C) = sum(match(s) for s in S_C)
```

Rank candidates by descending score. Break exact ties by ascending opaque `candidate_id`.

No text similarity, embeddings, gradients, learned weights, or manually authored semantic rules are used in this preflight.

## Truth isolation

Execution has two ordered stages.

1. Write the complete blind ranking for both worlds without reading candidate truth roles.
2. After the ranking artifact exists, compare it with the benchmark truth to report planted-root rank.

The blind artifact must be retained separately from truth evaluation.

## Interpretation

If the planted candidate ranks first in both worlds, M4 must state that localization in this benchmark is solvable by a simple target-overlap shortcut under the data-access regime.

That result does not invalidate M3. M3 established matched structural footprint, not semantic indistinguishability relative to the known target behavior.

A successful shortcut baseline also does not invalidate the certification question. The later causal question remains whether restoring the suspected candidate produces a recovery that is distinguishable from the four plausible non-root restorations under matched stochastic trajectories.

If the shortcut baseline does not solve localization, M4 may proceed to target-faithful Grad-Dot and other justified attribution methods under a separately frozen execution request.

Even if the shortcut baseline succeeds, Grad-Dot may still be retained as descriptive development evidence, but it is not required to manufacture a localization novelty claim.

## Authorization boundary

This preflight authorizes only deterministic release-diff analysis and post-ranking truth scoring.

It does not authorize:

- matched-world model training;
- restoration training;
- Grad-Dot scoring on matched-world checkpoints;
- TRAK, LESS, GIST, influence-function, or other attribution execution;
- certification thresholds;
- official test access.

`M4_TARGET_OVERLAP_PREFLIGHT=FROZEN`

`MATCHED_WORLD_MODEL_TRAINING_AUTHORIZED=NO`
