# M4 localization baseline protocol

**Status:** prospective freeze candidate before matched-world model training  
**Date:** 2026-10-02  
**Milestone:** M4 competitive localization baselines  
**Official Banking77 test split:** embargoed

## 1. Purpose

M4 asks a narrow diagnostic question:

> Given a known target behavior and five structurally matched visible training-release changes, which change should be investigated first?

Localization is not causal certification. A first-ranked candidate is a diagnostic hypothesis only. M5 remains responsible for counterfactual restoration, causal specificity, uncertainty, and abstention.

The M3 benchmark deliberately permits semantic information in visible release changes. If a simple change-overlap baseline solves localization, that result must be reported rather than hidden or repaired after training.

## 2. Benchmark substrate

M4 uses the two frozen M3 matched worlds defined by:

- `STRUCTURALLY_MATCHED_BENCHMARK_PROTOCOL.md`;
- `STRUCTURALLY_MATCHED_BENCHMARK_PROTOCOL_AMENDMENT_1.md`;
- the completed M3 structural benchmark artifacts.

Each world contains five candidate changes. Every candidate is a symmetric 66-slot label swap with:

- 33 records changed in each direction;
- zero text changes;
- zero aggregate label-count change;
- two labels touched;
- pairwise-disjoint changed slots and intent pairs within a world.

The target intent pair for each world is debugger-visible. The hidden benchmark variable is the opaque candidate ID associated with the responsible release change.

No M4 method may load the truth manifest while producing candidate scores.

## 3. Training regime

Matched-world training uses the clean development configuration selected before planted-regression outcomes:

- model: `distilbert/distilbert-base-uncased`;
- revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`;
- epochs: 7;
- batch size: 32;
- learning rate: 2e-5;
- weight decay: 0.01;
- warmup ratio: 0.10;
- maximum sequence length: 128;
- maximum gradient norm: 1.0.

Use development trajectories `0`, `1`, and `2`.

Train the clean baseline once per trajectory and the composite release once per world and trajectory. This produces:

```text
3 clean development models
2 worlds x 3 composite trajectories = 6 composite development models
```

All release hashes, trajectory seeds, slot schedules, runtime metadata, and model identities must be recorded.

The official Banking77 test split remains untouched.

## 4. Target behavior

For each matched world, apply the frozen target function from `research/ATTRIBUTION_TARGET.md` to that world's debugger-visible target pair.

For target example `(x, y)` and the other target label `y_other`:

```text
margin(x, y) = z(x)[y] - z(x)[y_other]
```

The world-level behavior scalar is the unweighted mean margin over every development-evaluation example whose true label is in the target pair.

All target examples receive equal weight. Do not select examples based on observed errors, flips, or attribution scores.

## 5. Common candidate aggregation

Every slot-scoring method must use the frozen candidate aggregation:

```text
candidate_score(C) = sum(slot_suspiciousness(s) for s in changed_slots(C))
```

Higher candidate score means more suspicious.

For model-based methods, compute one candidate score per trajectory and use the arithmetic mean of the three trajectory scores as the primary world-level candidate score. Report all three trajectory-specific scores as diagnostics.

Ties use ascending opaque `candidate_id`. The tie rule is mechanical and is not a scientific preference.

## 6. Frozen baseline suite

M4 uses the following compact suite. Additional methods may not be added after matched-world rankings are inspected.

### B0. Deterministic random reference

Assign each candidate a deterministic pseudo-random score from:

```text
SHA256("mrf-m4-random-v1|<world_id>|<candidate_id>")
```

Interpret the first eight digest bytes as an unsigned 64-bit big-endian integer. Higher value ranks first.

This baseline exists only as a chance reference.

### B1. Direct target-label overlap

For each candidate changed slot, inspect the current composite release record at that slot.

Define:

```text
slot_score = 1 if current_model_label is in the known target pair else 0
candidate_score = sum(slot_score over candidate changed slots)
```

This uses only debugger-visible information. It does not use the truth manifest or restoration outcomes.

The matched benchmark makes semantic shortcuts possible by design. B1 is therefore a required structural reference, not an attribution method. If B1 localizes the responsible candidate perfectly, that does not certify causality and does not invalidate the benchmark. It shows that localization is easy under the visible release information.

### B2. Changed-text lexical overlap

Tokenize text using lowercase ASCII alphanumeric tokens matched by:

```text
[a-z0-9]+
```

For each changed training slot and each target-slice development-evaluation example, compute set Jaccard similarity between token sets.

The slot score is the maximum Jaccard similarity to any target-slice example:

```text
slot_score(s) = max_t Jaccard(tokens(s), tokens(t))
```

The candidate score is the frozen sum over changed slots.

Empty-versus-empty token sets have similarity `0`. No stemming, stop-word removal, learned vocabulary, TF-IDF fitting, embedding model, or root-aware tuning is allowed.

### B3. Final-checkpoint last-layer Grad-Dot

Use the already frozen Grad-Dot definition in `research/GRAD_DOT_BASELINE.md` with the matched world's target pair.

- checkpoint: final epoch-7 composite checkpoint;
- active parameters: classifier weight and bias only;
- training loss: standard 77-way cross-entropy;
- target objective: negative pairwise target margin;
- target slice: every development-evaluation example in the target pair;
- slot suspiciousness: the prospectively defined sign mapping in the frozen Grad-Dot protocol;
- candidate aggregation: frozen sum.

No sign, layer, target subset, or aggregation rule may be selected using root ranking.

### B4. Checkpoint TracIn, last-layer restricted

Use Captum `TracInCPFast` or an algebraically equivalent implementation restricted to the final classifier layer.

The purpose of B4 is to add training-trajectory checkpoint information while keeping layer scope directly comparable with B3.

#### Checkpoint schedule

Capture the complete model state at the end of each of the seven training epochs:

```text
epoch 1, 2, 3, 4, 5, 6, 7
```

Checkpoint selection may not depend on validation behavior or candidate ranking.

Each checkpoint record must store:

- epoch index;
- full model-state hash;
- optimizer-step count;
- learning rate used for the final optimizer update that produced the checkpoint state.

The full checkpoint files may be temporary workflow artifacts if scoring occurs in the same source-pinned run. Their hashes and the final baseline output must be retained even if the large checkpoint files are not kept permanently.

#### Training and target losses

Training-example gradients use standard 77-way cross-entropy.

Test gradients use:

```text
L_target(x, y) = -margin(x, y)
```

for every target-slice development-evaluation example.

This keeps the TracIn target mathematically aligned with the frozen behavior scalar.

#### Checkpoint weighting

Weight each checkpoint contribution by the learning rate used for the final optimizer update that produced that checkpoint state.

Do not substitute the scheduler's post-step learning rate when that value corresponds to the next update. In particular, do not allow the final checkpoint to receive zero weight merely because the scheduler reaches zero after the final update.

#### Sign orientation

Captum describes positive TracIn influence as a proponent effect, meaning removal of the training example would increase the supplied test loss. With `L_target = -margin`, positive native influence therefore indicates support for the target behavior rather than suspicion.

Freeze:

```text
slot_suspiciousness = - native_aggregate_influence
```

before any candidate ranking is inspected.

Aggregate native influence equally across the complete target slice before applying the sign conversion and candidate sum.

## 7. TRAK feasibility boundary

TRAK is not part of the M4 confirmatory baseline suite.

The existing feasibility audit established that the standard multiclass TRAK model output does not match the frozen pairwise target. A custom scalar model output is technically possible, but the actual 77-way cross-entropy training loss is not a function of the pairwise margin alone. A faithful scalar `dL/df` mapping is therefore not presently established.

Do not introduce a pseudo-loss or change the target merely to force TRAK into the benchmark.

TRAK remains a documented feasibility exclusion unless a mathematically valid target/loss mapping is established independently of M4 rankings.

## 8. Required implementation validation

Before matched-world result-bearing training, the M4 implementation must pass tests for:

- deterministic B0 scoring;
- B1 slot-to-composite label joins;
- B2 tokenization and Jaccard edge cases;
- exact target-pair margin orientation;
- B3 target and suspiciousness sign conventions;
- seven-checkpoint capture with hashes and optimizer-step metadata;
- B4 use of the learning rate that produced each checkpoint state;
- B4 target-loss mapping to negative pairwise margin;
- candidate sum aggregation;
- three-trajectory arithmetic-mean aggregation;
- truth-manifest isolation;
- official-test embargo.

A tiny synthetic or toy-model validation may be used to verify TracIn sign orientation and checkpoint weighting. It must not use a matched-world root ranking to choose an orientation.

## 9. Result artifact

For each world and method, record:

- source Git SHA;
- protocol identity;
- development partition hash;
- world identity and release hashes;
- target labels and target-example count;
- clean and composite behavior scalar for each trajectory;
- target regression for each trajectory;
- method/library version;
- method-specific configuration;
- slot-score hash where applicable;
- candidate score by trajectory;
- primary mean candidate score;
- candidate ranking;
- runtime and device;
- confirmation that the truth manifest was not loaded during scoring;
- confirmation that the official test split was not loaded.

Only after all method artifacts are complete and hashed may a separate scoring step load the M3 truth manifest to compute root rank, top-1 accuracy, or other benchmark summaries.

## 10. Interpretation

M4 is a localization benchmark, not a causal benchmark.

The primary reporting question is not which method wins. Report the complete candidate ranking and whether model-based attribution adds useful information beyond the visible-change baselines.

If B1 solves localization by construction, preserve that result. The scientific motivation for MRF then becomes sharper: correct localization, even when easy, is still not causal certification.

Do not redesign M3 after seeing M4 rankings merely to make attribution methods look more competitive.

## 11. Transition to M5

After M4 is complete, freeze M5 before any confirmatory restoration execution.

M5 must separately specify:

- confirmatory trajectory count;
- practical restoration-effect margin;
- simultaneous uncertainty procedure;
- protected-behavior equivalence rule;
- abstention criterion;
- official-test access rule, if any.

M4 rankings may nominate a candidate for investigation. They may not define the causal-certification threshold.

## Frozen markers

`MRF_M4_LOCALIZATION_BASELINE_PROTOCOL=FROZEN_BEFORE_MATCHED_WORLD_TRAINING`

`MRF_M4_DEVELOPMENT_TRAJECTORIES=0,1,2`

`MRF_M4_TRAINING_EPOCHS=7`

`MRF_M4_BASELINES=B0_RANDOM,B1_LABEL_OVERLAP,B2_LEXICAL_OVERLAP,B3_GRAD_DOT,B4_TRACIN`

`MRF_M4_TRAK_INCLUDED=NO`

`MRF_M4_OFFICIAL_TEST_ACCESSED_AT_FREEZE=NO`
