# Exp009 structurally matched benchmark protocol

**Status:** PROSPECTIVE M3 DESIGN — frozen before matched-benchmark model training  
**Date:** 2026-09-25  
**Evidence class:** development benchmark design  
**Official Banking77 test split:** embargoed

This protocol replaces the structurally mismatched nuisance-v2 construction for
future end-to-end benchmark work. It does not alter or reinterpret the completed
Stage-A or Stage-B pilot.

## 1. Purpose

The benchmark asks:

> Given a known-good release, a regressed release, a known target behavior, and
> five debugger-visible versioned training changes with matched observable
> structure, can a diagnostic localize the change responsible for the target
> regression and can counterfactual restoration then distinguish that root from
> the four non-root changes under matched retraining stochasticity?

The benchmark is intentionally narrower than arbitrary training-data causal
identification.

## 2. Structural-matching principle

All five candidate changes in a benchmark world use the same mechanism:

- artifact kind: training-label mapping update;
- changed stable slots: exactly 66;
- per direction: exactly 33;
- text changes: exactly 0;
- label changes: exactly 66;
- aggregate label-count delta: exactly zero;
- two labels touched per candidate;
- symmetric transitions: 33 A->B and 33 B->A;
- training weight / slot multiplicity: unchanged;
- serialized debugger-facing schema: identical across candidates.

No candidate may be uniquely identifiable because it alone changes text, uses a
different record count, changes label mass, or exposes a different metadata
schema.

## 3. Candidate mechanism

For candidate pair `(A, B)`, build the candidate release with the existing
deterministic symmetric label-swap constructor at:

```text
per_direction = 33
total_changed_slots = 66
```

The selected slot IDs are deterministic functions of the frozen development
partition and the pair labels.

For every changed slot:

- text remains byte-for-byte unchanged;
- only the model-facing label changes;
- 33 A-labelled records become B-labelled;
- 33 B-labelled records become A-labelled.

A restoration replaces exactly those 66 candidate slots with their baseline
records.

## 4. World structure

A world contains exactly five **pairwise-disjoint intent pairs**:

```text
P0, P1, P2, P3, P4
```

No intent label appears in more than one candidate within a world.

Exactly one pair is the benchmark-private root. Its two intents define the
known target behavior for that world.

The remaining four pairs are non-root candidate changes. They may cause
regressions on their own touched intents; that does not make them causal for
the target behavior.

The full composite release applies all five 66-slot changes simultaneously.

## 5. Target and protected behavior

### Target behavior

The target behavior is macro recall over the two root-pair intents.

The target pair is known to the debugger. The hidden variable is which opaque
candidate ID corresponds to the release change that altered those intents.

### Candidate-touched behavior

All ten intents touched by the five candidate pairs are tracked separately.
Their metrics are diagnostic and must not be silently folded into the protected
aggregate.

### Protected behavior

The protected set is every Banking77 intent not touched by **any** of the five
candidate pairs.

This definition is frozen because four structurally matched non-root changes
are intentionally allowed to affect their own intent pairs. Treating those
eight intents as "protected" would confound target locality with the existence
of the other planted release faults.

The protected aggregate is macro recall over the untouched intents.

A worst-untouched-intent drift diagnostic must also be recorded.

## 6. Pair eligibility from clean development evidence only

Candidate-pair eligibility is determined before matched-benchmark corruption
training using only the frozen development partition and clean-model evidence.

Each label in an eligible pair must have:

- at least 66 development-training records;
- at least 20 development-evaluation records;
- minimum clean recall >= 0.90 across the frozen clean development
  trajectories.

Each eligible pair must additionally satisfy at least one predeclared semantic
adjacency signal:

1. pooled clean confusion count between the pair > 0; or
2. lexical token overlap between the intent names > 0.

The exact clean evidence snapshot and hashes used for pair selection must be
pinned before world construction.

No matched-benchmark corruption outcome may be used to make a pair eligible.

## 7. Deterministic pair ranking

Eligible pairs are ranked by the following tuple:

1. descending pooled symmetric clean confusion rate;
2. descending pooled mutual confusion count;
3. descending lexical Jaccard of the label names;
4. descending minimum clean recall;
5. lexicographic label A;
6. lexicographic label B.

This ranking uses clean development evidence only.

## 8. World selection

The initial M3 benchmark preflight constructs **three worlds** mechanically.

World construction proceeds from the frozen pair ranking.

For world index `w in {0,1,2}`:

1. scan the ranked pair list from the beginning;
2. skip a pair if either intent was already assigned to an earlier world;
3. select the first five pairwise-disjoint eligible pairs;
4. remove those ten intent labels from eligibility for later worlds.

This yields 15 pair changes across three worlds with no label reuse.

### Root assignment

Root position must not be coupled to pair rank.

For each world, compute:

```text
root_position = uint64(
  SHA256("exp009-matched-root-v1|world=<w>")[:8]
) mod 5
```

using the first eight digest bytes in big-endian order.

The pair at that position is the hidden root for benchmark truth.

The debugger-facing manifest contains only opaque candidate IDs and never the
root position.

## 9. Opaque candidate IDs

Candidate IDs are deterministic hashes over:

- benchmark namespace;
- world index;
- sorted pair labels;
- changed-slot-ID hash.

The ID must not contain:

- label names;
- root/nuisance role;
- pair rank;
- target status;
- restoration outcomes.

Debugger-visible candidate manifests may expose neutral structural invariants
such as changed-slot count and content hashes only if the same fields are
present for all five candidates.

## 10. Structural-equality audit

Before any matched-benchmark model training, every world must pass an automated
audit proving that all five candidates have identical values for:

- changed-slot count = 66;
- text-change count = 0;
- label-change count = 66;
- aggregate label-count delta = {};
- number of touched labels = 2;
- transition counts = 33 / 33;
- slot count;
- serialized manifest field set.

The audit also requires:

- all five changed-slot sets are pairwise disjoint;
- all ten intent labels are pairwise distinct;
- every restoration returns exactly to the baseline release;
- the five-way composite is independent of candidate-application order.

A failed audit invalidates the world before model training. Do not patch one
candidate manually.

## 11. Pre-result benchmark gate

After structural preflight, result-bearing **development** training may test
whether the mechanically selected worlds are usable.

For each world and frozen development trajectory family, record:

- clean target macro recall;
- composite target macro recall;
- target regression;
- protected untouched-intent macro recall;
- worst untouched-intent drift;
- candidate-touched intent metrics;
- training validity.

A world is not replaced because its target regression is weak or because a
non-root intervention has a large target effect. Such outcomes are evidence
about the construction.

M3 itself freezes construction. It does not create a post-hoc behavioral pass
threshold.

## 12. Localization boundary

The benchmark does not attempt to hide the target behavior from the debugger.

A simple baseline may exploit semantic overlap between a candidate's changed
records and the known target slice. That is legitimate release-forensics
information, not structural leakage.

Accordingly, M4 must include simple lexical/change-overlap baselines alongside
gradient/influence methods. If a trivial baseline solves localization, report
that result rather than redesigning worlds after seeing it.

## 13. Certification boundary

Localization and certification remain separate.

For a root candidate `r` and non-root candidate `j`, the later
certification layer uses paired restoration effects:

```text
Delta_j,t = M(R_j,t) - M(C_t)
D_j,t = Delta_r,t - Delta_j,t
```

The matched candidate construction removes the current text-change artifact,
but does not itself establish causal specificity.

Trajectory count, practical effect margin, simultaneous uncertainty procedure,
protected-equivalence rule, and abstention criterion remain M5 decisions and
are **not** frozen here.

## 14. Official-test embargo

The official Banking77 test split remains untouched throughout M3 and M4
development.

It may not be used for:

- pair eligibility;
- pair ranking;
- world selection;
- root assignment;
- structural audit;
- localization-baseline tuning;
- matched-world behavioral preflight.

Any later official-test access rule must be frozen in M5 before access.

## 15. Invalidity and rerun discipline

A world is structurally invalid only for a predeclared construction failure,
including:

- fewer than five disjoint eligible pairs;
- insufficient slots for the symmetric swap;
- structural-equality audit failure;
- candidate overlap;
- restoration not exactly recovering baseline;
- composite order dependence.

Behavioral weakness is not structural invalidity.

Technical reruns must preserve world ID, trajectory ID, release hashes, source
revision, and protocol identity.

## 16. Claim boundary

Passing M3 can establish only that a structurally matched candidate benchmark
has been prospectively constructed.

It does not establish:

- localization success;
- causal specificity;
- confirmatory certification;
- superiority to modern attribution methods;
- official-test performance;
- cross-model or cross-dataset generalization.

## Frozen markers

`EXP009_MATCHED_BENCHMARK_PROTOCOL=FROZEN_FOR_M3_IMPLEMENTATION`

`EXP009_MATCHED_CANDIDATE_MECHANISM=SYMMETRIC_LABEL_SWAP_33_PER_DIRECTION`

`EXP009_MATCHED_WORLD_COUNT=3`

`EXP009_OFFICIAL_TEST_ACCESSED_AT_FREEZE=NO`

`EXP009_MATCHED_BENCHMARK_MODEL_TRAINING_AT_FREEZE=NO`
