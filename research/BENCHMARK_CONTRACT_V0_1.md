# Model Regression Forensics — Benchmark Contract v0.1

Status: **draft benchmark specification**
Date: **2026-09-20**
Branch: `research-plan-v2`

This document defines the minimum scientific contract for a Model Regression Forensics (MRF) benchmark case.

It is intentionally independent of Banking77, DistilBERT, and Experiment 009.

The purpose is to ensure that future benchmark cases, methods, and claims are comparable and that benchmark construction cannot be changed post hoc merely to rescue a desired result.

---

# 1. Benchmark objective

An MRF benchmark evaluates whether a diagnostic method can identify and support the causal explanation for a model regression when:

- a known-good release exists;
- a later release regresses on a defined target behavior;
- several concrete versioned changes occurred between releases;
- only one change or one interacting change set is the hidden ground-truth cause in a single-root benchmark case;
- training is stochastic;
- controlled restoration/intervention is possible;
- unrelated behavior should remain sufficiently stable;
- intervention cost is measurable.

The benchmark separates:

1. **regression detection**;
2. **candidate-lineage observation**;
3. **causal localization**;
4. **causal intervention**;
5. **causal certification or abstention**.

A method must not receive hidden ground-truth cause information during diagnosis.

---

# 2. Core entities

## 2.1 Model release

A `ModelRelease` is a reproducible training-and-evaluation state.

At minimum it should identify:

- source/base model or initialization;
- training data revision;
- training code/config revision;
- preprocessing/tokenization configuration;
- optimization configuration;
- stochastic trajectory definition;
- evaluation specification;
- software/backend environment where scientifically relevant.

Two principal releases exist in each case:

- (B): known-good baseline release;
- (C): regressed candidate release.

The benchmark must document which elements are held constant and which differ.

## 2.2 Candidate change

A `CandidateChange` is one debugger-visible, versioned change between (B) and (C).

Examples include:

- data addition/removal/relabeling;
- sampling/reweighting;
- preprocessing;
- augmentation;
- tokenizer/configuration;
- optimizer/scheduler settings;
- hyperparameters;
- base checkpoint;
- code/configuration change.

Each change must have:

- stable identifier;
- human-readable description;
- machine-readable type/category;
- before state;
- after state;
- intervention definition;
- provenance/version reference.

Candidate changes must be concrete enough that a restoration can be implemented.

## 2.3 Root cause

The hidden `RootCause` is the benchmark-defined causal change or change set.

For v0.1, single-root cases are preferred:

[
R^* = {delta_j}
]

Later benchmark versions may support interacting root sets:

[
R^* = {delta_a, delta_b, ldots}.
]

The root is hidden from the diagnostic method but known to benchmark construction and evaluation code.

## 2.4 Intervention

An `Intervention` modifies the candidate release while holding all unrelated benchmark-defined factors fixed as closely as possible.

For a single candidate (delta_j), the canonical restoration intervention is:

> start from the candidate-world training state, restore only (delta_j) to its baseline value, then retrain under the matched trajectory.

A valid intervention must specify:

- target change(s);
- resulting training specification;
- paired stochastic trajectory;
- expected invariants;
- expected changed fields;
- artifact/result identifier.

Interventions must be reproducible.

## 2.5 Trajectory

A `Trajectory` identifies the benchmark-controlled stochastic training condition.

A trajectory may control:

- initialization seed;
- classifier/head initialization;
- data-order seed;
- dropout seed;
- augmentation seed;
- optimizer randomness;
- any other benchmark-defined stochastic component.

Within one paired comparison, (B_t), (C_t), and restoration siblings (R_{j,t}) should share the same trajectory definition where technically possible.

Backend/hardware changes that can alter result distributions must not be mixed inside one paired causal comparison.

## 2.6 Metric specification

A benchmark case must define:

### Target metric

A metric (M) for the behavior that regressed.

Examples:

- class recall;
- slice accuracy;
- task success;
- reward score;
- calibrated loss;
- domain-specific behavioral metric.

### Protected metric(s)

One or more metrics (P_1,ldots,P_k) representing behavior that should not materially degrade.

Protected metrics prevent a method from “recovering” the target through broad destructive changes.

---

# 3. Required benchmark-case components

Every benchmark case must contain the following.

## 3.1 Baseline release

A reproducible known-good baseline.

The baseline must satisfy predeclared minimum quality criteria before corruption/change construction proceeds.

If the clean baseline fails, causal diagnosis must stop.

## 3.2 Candidate release

A release containing:

- the hidden root change;
- one or more non-root candidate changes where the benchmark design requires them.

The candidate must exhibit the intended regression under predeclared development criteria.

## 3.3 Candidate set

The full debugger-visible candidate set:

[
Delta = {delta_1,ldots,delta_n}.
]

The method is evaluated only over this declared search space.

If the true root is absent from (Delta), that constitutes a different “missing-cause” benchmark regime and must be labeled separately.

## 3.4 Hidden truth

The benchmark stores the causal root/root set separately from debugger-visible inputs.

A method must not access this information during diagnosis or intervention selection.

## 3.5 Intervention map

Every candidate change must have a defined restoration/intervention procedure, unless the benchmark explicitly declares a candidate non-intervenable and evaluates a different protocol.

For v0.1, all included candidates should preferably be intervenable.

## 3.6 Repeated trajectories

Each benchmark case must define enough repeated stochastic trajectories to estimate whether observed intervention effects are robust relative to training variability.

The exact trajectory count may vary by case, but it must be frozen before confirmatory evaluation.

## 3.7 Evidence record

The benchmark should retain:

- release metrics;
- per-trajectory outcomes;
- intervention outcomes;
- provenance;
- software/model/data revisions;
- hashes/checksums where applicable;
- compute/runtime metadata where practical.

---

# 4. Observable versus hidden information

This distinction is mandatory.

## 4.1 Debugger-visible information

A diagnostic method may receive:

- baseline release metadata;
- candidate release metadata;
- candidate-change list;
- before/after representations of candidate changes;
- baseline and candidate evaluation results;
- permitted model outputs;
- permitted activations/gradients/checkpoints if the benchmark track allows them;
- permitted data samples or diffs;
- lineage/provenance metadata;
- previously executed intervention outcomes in sequential-search tracks.

## 4.2 Hidden information

A diagnostic method must not receive:

- root-cause identifier;
- benchmark construction labels indicating which change is causal;
- future intervention outcomes;
- confirmatory test-set outcomes before allowed;
- privileged synthetic-generation metadata that trivially reveals the root;
- manually authored hints created using the hidden truth.

## 4.3 Access tracks

MRF-Bench may define explicit access tracks rather than pretending every method has the same information.

Possible tracks:

### Lineage-only

Uses release metadata, candidate changes, and observed eval behavior.

### Model-access

Adds model weights, gradients, activations, checkpoints, or embeddings.

### Data-access

Adds relevant training examples/diffs.

### Intervention-aware

Adds outcomes of already executed interventions.

A method must be compared only against baselines in the same access regime.

---

# 5. Regression qualification

A benchmark candidate is usable only if the regression is prospectively qualified.

For paired trajectory (t):

[
G_t = M(B_t) - M(C_t).
]

A benchmark protocol must predeclare:

- minimum per-trajectory regression, if any;
- aggregate regression threshold;
- protected-metric degradation limits;
- replication requirements;
- what happens if qualification fails.

A benchmark case that fails qualification must be recorded as a failed construction attempt.

Thresholds must not be relaxed after outcome inspection to rescue the case.

---

# 6. Intervention effects

For candidate change (delta_j) and trajectory (t):

[
mathrm{Recovery}_{j,t}
=
M(R_{j,t}) - M(C_t).
]

Protected-metric effect may be represented as:

[
mathrm{ProtectedDrift}_{j,t}
=
P(R_{j,t}) - P(C_t)
]

or another prospectively defined direction/sign convention.

For root (r) versus nuisance candidate (j):

[
D_{r,j,t}
=
mathrm{Recovery}_{r,t}
-
mathrm{Recovery}_{j,t}.
]

The benchmark must preserve raw per-trajectory results rather than only averages.

---

# 7. Causal evidence states

MRF-Bench should distinguish at least three outcome states.

## 7.1 SUPPORTED

The benchmark's prospectively frozen certification rule is satisfied.

Conceptually this means:

- target recovery is material;
- root recovery is sufficiently robust across trajectories;
- protected behavior remains within allowed limits;
- recovery is sufficiently stronger/more specific than nuisance alternatives;
- uncertainty is low enough for the declared confidence standard.

The exact statistical rule is benchmark-version specific and must be frozen before confirmatory evaluation.

## 7.2 NOT_SUPPORTED

Evidence actively fails the causal-support criterion.

Examples:

- root restoration does not recover the target;
- nuisance restoration performs similarly or better;
- protected behavior degrades excessively;
- candidate identification is wrong;
- claimed confidence is inconsistent with the evidence.

## 7.3 ABSTAIN

Evidence is insufficient to support or reject a unique causal conclusion under the allowed budget.

Examples:

- confidence intervals remain too wide;
- multiple candidates remain observationally/interventionally indistinguishable;
- intervention budget is exhausted;
- stochastic variability prevents separation.

Abstention is a valid and sometimes desirable result.

Unsupported certification is worse than abstention.

---

# 8. Diagnosis outputs

A benchmark method may output:

- ranked candidate list;
- per-candidate score/probability;
- predicted root/root set;
- confidence estimate;
- selected next intervention;
- stop/continue decision;
- final certification state.

A method should not be required to output probabilities if its theory does not justify probabilistic calibration.

---

# 9. Evaluation metrics

MRF-Bench should support the following.

## 9.1 Localization

- top-1 root identification accuracy;
- top-k root inclusion;
- mean reciprocal rank or equivalent ranking metric where appropriate.

## 9.2 Certification

- supported-cause accuracy;
- false-certification rate;
- false-rejection rate where meaningful;
- abstention rate;
- coverage among non-abstained cases.

## 9.3 Effect separation

- root recovery;
- nuisance recovery;
- root-vs-best-nuisance margin;
- protected-metric drift;
- variability across trajectories.

## 9.4 Cost

At minimum:

- number of intervention runs.

Where practical:

- accelerator-hours;
- normalized training FLOPs;
- wall-clock time;
- energy/cost estimate.

Intervention count should remain the primary portable metric when hardware differs.

## 9.5 Sequential search

For adaptive methods:

- intervention count to decision;
- compute to decision;
- incorrect early-stop rate;
- false-certification rate;
- regret or best-arm-style metrics only where the assumptions are meaningful.

---

# 10. Benchmark fairness

## 10.1 Equal information

Compared methods must receive the same benchmark-visible information unless explicitly evaluated in separate access tracks.

## 10.2 No hidden-root leakage

Any feature or artifact that trivially reveals the hidden root invalidates the benchmark case.

## 10.3 Baseline appropriateness

Methods should only be compared where their assumptions apply.

Example:

TRAK or TracIn are appropriate for data-change cases with the required differentiable-model access.

They should not be declared failures on optimizer/config changes that their method is not designed to represent.

## 10.4 Fixed evaluation protocol

The benchmark protocol, metric definitions, and confirmatory stopping rules must be frozen before final evaluation.

## 10.5 Negative construction results

Failed benchmark-construction attempts must be preserved when scientifically informative.

---

# 11. Benchmark case taxonomy

Each benchmark case should declare its root-cause category.

Initial categories:

- `data.label_mapping`
- `data.add_remove`
- `data.sampling`
- `preprocessing`
- `augmentation`
- `optimizer`
- `scheduler`
- `hyperparameter`
- `checkpoint`
- `tokenizer`
- `code_or_config`
- `interaction`

MRF-Bench v0.1 does not need all categories.

The taxonomy exists so benchmark expansion remains coherent.

---

# 12. Plausible nuisance requirement

A non-root candidate should not merely be random noise.

A benchmark nuisance should be a **plausible alternative explanation** that:

- is a real model-facing change;
- could reasonably be suspected from lineage/evidence;
- is comparable enough to the root to make diagnosis non-trivial;
- does not contain privileged root information;
- is prospectively defined;
- can be restored/intervened upon;
- does not intentionally sabotage the benchmark through unrelated global degradation.

The benchmark must document why each nuisance is scientifically plausible.

This requirement is directly motivated by Experiment 009's failed nuisance-rule v1.

Nuisance construction must not be relaxed post hoc until a desired number of candidates appears.

---

# 13. Single-root versus interaction tracks

## 13.1 Single-root track

Exactly one candidate change is the ground-truth cause.

This is the required first benchmark track.

## 13.2 Interaction track

A later track may define root sets where the regression emerges only from an interaction.

Example:

- restore (A) alone → no meaningful recovery;
- restore (B) alone → no meaningful recovery;
- restore (A+B) → recovery.

Interaction cases must not be evaluated using single-root scoring rules without modification.

---

# 14. Missing-root track

A future benchmark track may intentionally omit the true cause from the candidate list.

Its purpose would be to test whether a method can avoid falsely certifying an available candidate when the true cause is absent.

This is valuable for real-world realism but is **out of scope for v0.1**.

---

# 15. Reproducibility requirements

A publishable MRF benchmark case should pin or record, as applicable:

- dataset source/revision/hash;
- model/checkpoint revision;
- training configuration;
- code revision;
- software dependencies;
- trajectory/seeding logic;
- benchmark-case specification;
- intervention definitions;
- evaluation code;
- result artifact hashes;
- hardware/backend class where scientifically relevant.

Reproduction instructions should separate:

- lightweight verification;
- full retraining reproduction.

Large model checkpoints need not be committed when reproducibly obtainable.

---

# 16. Benchmark package schema — conceptual v0.1

A future library representation may resemble:

```python
RegressionCase(
    case_id="...",
    baseline_release=...,
    candidate_release=...,
    changes=[...],
    target_metric=...,
    protected_metrics=[...],
    trajectories=[...],
    access_policy=...,
    hidden_truth=...,
    interventions=...,
)
```

The benchmark loader must ensure `hidden_truth` is unavailable to diagnostic code.

Possible result type:

```python
DiagnosisResult(
    ranking=[...],
    predicted_root=...,
    confidence=...,
    selected_interventions=[...],
    decision="SUPPORTED" | "NOT_SUPPORTED" | "ABSTAIN",
)
```

These names are conceptual only and do not freeze the public API.

---

# 17. Exp009 mapping to this contract

Experiment 009 currently maps approximately as follows:

- baseline release: clean Banking77/DistilBERT classifier;
- candidate release: accepted (1/4) symmetric label-mapping corruption plus future legitimate nuisances;
- target metric: macro recall on the selected target pair;
- protected metric: macro recall over protected intents;
- candidate changes: root plus planned nuisance changes;
- trajectories: three paired stochastic trajectories in the development pilot;
- intervention: restore one candidate change while retaining the rest;
- hidden truth: the planted label-mapping fault;
- certification: not yet frozen;
- official confirmatory test: untouched.

The failed nuisance v1 construction is not a benchmark failure yet.

It demonstrates that the nuisance definition needs to be grounded in this contract before v2 is designed.

---

# 18. Exp009 nuisance-design requirements derived from the contract

Before nuisance rule v2 exists, the project must answer:

1. What makes a Banking77 change a plausible alternative explanation?
2. Must nuisances affect the same target slice, an adjacent semantic slice, or merely be model-facing?
3. How similar should nuisance magnitude be to root magnitude?
4. Must nuisance changes preserve class counts?
5. Must they preserve global/protected quality?
6. Is confusion evidence required?
7. Is semantic adjacency required?
8. What leakage would make the nuisance too easy?
9. What behavior would make the nuisance unrealistically destructive?
10. How many nuisances are scientifically necessary for the benchmark objective?

The answer should be driven by benchmark validity, not by the need to manufacture exactly four candidates.

---

# 19. Benchmark v0.1 minimum release criteria

MRF-Bench v0.1 should not be released until it contains:

- at least two complete benchmark cases;
- at least two distinct causal mechanisms;
- at least one natural-language case;
- repeated stochastic trajectories;
- exhaustive intervention truth for included candidates;
- at least three diagnosis baselines;
- at least one data-attribution baseline on a compatible case;
- documented failure/abstention semantics;
- reproducible benchmark loader/runner;
- frozen evaluation metrics;
- clear limitations.

A second dataset/model substrate is preferred before calling the benchmark broadly representative.

---

# 20. What this contract does NOT freeze

This document does not yet freeze:

- Exp009 nuisance v2;
- Exp009 confirmatory thresholds;
- exact statistical certification procedure;
- exact trajectory count for all future cases;
- the second benchmark substrate;
- Exp010 algorithm;
- final public Python API;
- publication venue;
- final benchmark name/version.

Those decisions require additional evidence.

---

# 21. Immediate next step

With the benchmark contract defined, the next task is an **Exp009 nuisance-rule v1 autopsy**.

That analysis should determine, without training any models:

- candidate-pair counts after each individual eligibility condition;
- candidate-pair counts after cumulative conjunctions;
- which rule(s) drive eligibility to zero;
- how many intent-disjoint pairs survive each stage;
- whether the failure arose from one overly restrictive criterion or from the joint geometry of the rule.

Only after that analysis should a scientifically justified nuisance v2 proposal be written.

No training should begin during the autopsy.
