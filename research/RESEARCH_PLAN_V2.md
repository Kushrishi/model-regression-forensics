# Model Regression Forensics — Research Plan v2

Status: **research architecture record; current sequencing is governed by `research/STATE.md` and `research/ROADMAP.md`**
Date: **2026-09-20**
Branch: `research-plan-v2`
Starting point: `73e444cddd62a4fb0991429010aeb402b1570cc6`

## 1. Purpose

Model Regression Forensics (MRF) is first and foremost an independent machine-learning research project and portfolio artifact.

Its success does **not** depend on becoming a startup.

The project should demonstrate the ability to:

- formulate an ambiguous ML failure as a precise research problem;
- build controlled training and evaluation pipelines;
- distinguish correlation/localization from causal evidence;
- reason about stochastic retraining;
- design reproducible benchmarks and interventions;
- compare methods against serious baselines;
- preserve negative results;
- turn experimental machinery into reusable research software;
- communicate the result as a paper-quality technical artifact.

The intended final outcome is:

> **a reproducible benchmark + causal-certification methodology + small open-source Python research library + paper/preprint-quality evaluation.**

Commercialization remains optional upside only.

---

## 2. Research question

The working research question is:

> **Given a known-good model release, a regressed model release, and a finite set of versioned training-system changes between them, can we identify and experimentally support the causal change or change set under stochastic retraining, while controlling intervention cost?**

This is deliberately narrower than generic training-data attribution, generic root-cause analysis, experiment tracking, or model diffing.

### 2.1 Formal objects

Let:

- (B) denote a known-good model release;
- (C) denote a regressed model release;
- (Delta = {delta_1,ldots,delta_n}) denote the finite set of versioned candidate changes between (B) and (C);
- (M(cdot)) denote a target evaluation metric;
- (P(cdot)) denote one or more protected/non-target metrics;
- (t) denote a stochastic training trajectory;
- (R_{j,t}) denote the model trained under trajectory (t) after restoring candidate change (delta_j);
- (R_{S,t}) denote the model trained after restoring a subset (S subseteq Delta).

For a paired trajectory:

[
G_t = M(B_t) - M(C_t)
]

is the observed regression magnitude, and:

[
mathrm{Recovery}_{j,t} = M(R_{j,t}) - M(C_t)
]

is the target recovery produced by restoring candidate (j).

A causal diagnosis should not be accepted merely because one restoration improves the target metric.

Evidence should also address:

- stochastic variability;
- nuisance restorations;
- protected-behavior stability;
- repeated trajectories;
- alternative explanations;
- eventually, interacting root sets.

### 2.2 Current single-root certification intuition

For a single-root benchmark, a candidate should be supportable only when its recovery distribution is materially positive and distinguishable from plausible non-root restoration effects and ordinary retraining variability.

A useful quantity is the root-vs-nuisance recovery margin:

[
D_{j,k,t} =
mathrm{Recovery}_{j,t} -
mathrm{Recovery}_{k,t}
]

for candidate (j) against nuisance (k).

Exact statistical criteria are **not frozen by this document**. They must be prospectively defined before confirmatory experiments.

---

## 3. Scope

### 3.1 In scope

MRF studies:

- post-hoc investigation after a meaningful regression has already been detected;
- known-good versus regressed releases;
- finite, versioned candidate changes;
- controlled counterfactual restoration/intervention;
- repeated stochastic retraining;
- causal support versus nuisance alternatives;
- intervention cost;
- benchmark construction for regression-diagnosis methods.

Candidate changes may eventually include:

- training-data changes;
- label/taxonomy changes;
- sampling changes;
- preprocessing changes;
- augmentation changes;
- hyperparameter changes;
- optimizer/scheduler changes;
- base-checkpoint changes;
- tokenizer changes;
- code/configuration changes;
- selected interactions.

### 3.2 Explicitly out of scope for the core project

MRF is **not** intended to become:

- a generic experiment tracker;
- a W&B/MLflow replacement;
- a model registry;
- a generic data-quality platform;
- a generic training-data-attribution library;
- a production monitoring SaaS;
- a model-hosting platform;
- an autonomous debugger for arbitrary unreproducible incidents;
- a replacement for evaluation validity.

If the original evaluation is invalid, MRF can only explain an invalid target.

---

## 4. Prior-art position

The broad problem statement is not novel.

Important adjacent work includes:

- influence functions and TracIn for training-example influence;
- Datamodels for predicting dataset counterfactuals;
- TRAK for scalable training-data attribution;
- Final-Model-Only Data Attribution;
- Distributional Training Data Attribution for stochastic training;
- DATE-LM for benchmarking data-attribution methods;
- Goodfire Predictive Data Debugging / Probe-Based Data Attribution;
- Anthropic model-diffing work;
- model-update regression / negative-flip literature;
- ML-pipeline root-cause and counterfactual-repair work;
- causal bandits, best-arm identification, sequential experimental design, and budgeted interventions;
- MLflow/W&B/DVC-style lineage and experiment tracking.

Therefore MRF must **not** claim novelty for any one of the following in isolation:

- training-data attribution;
- accounting for stochastic training;
- retraining to evaluate a counterfactual;
- behavior/model diffing;
- pipeline lineage;
- causal intervention;
- adaptive experiment selection;
- best-arm identification;
- regression measurement.

### 4.1 Working white-space hypothesis

The research gap to test is the combination:

> **post-hoc causal diagnosis over an explicit version-diff search space, with hidden known causal truth, matched stochastic retraining, root-vs-nuisance intervention comparison, and evaluation of diagnosis quality per unit intervention cost.**

This remains a **working hypothesis, not a novelty claim**.

If directly overlapping prior work is found, claims must narrow or pivot.

---

## 5. Research contribution hierarchy

MRF should not depend on one high-risk novelty claim.

The project will pursue contributions in the following order.

### Contribution A — MRF-Bench

A reproducible benchmark/evaluation protocol for causal diagnosis of model regressions.

A benchmark case should provide:

- healthy release specification;
- regressed release specification;
- observable candidate-change set;
- hidden causal truth;
- target metric;
- protected metrics;
- intervention definitions;
- repeated stochastic trajectories;
- reproducible training/evaluation environment;
- exhaustive ground-truth intervention outcomes where feasible.

The benchmark should let a method answer:

1. Which change is the cause?
2. How strong is the evidence?
3. Should the method abstain?
4. How many expensive interventions were required?
5. Did the diagnosis remain correct under retraining stochasticity?

This contribution may remain valuable even if no new search algorithm is discovered.

### Contribution B — MRF-Certify

A methodology and software layer for evaluating causal support from repeated counterfactual interventions.

MRF-Certify should distinguish:

- localization/ranking;
- restorative influence;
- causal specificity;
- protected-behavior stability;
- stochastic uncertainty;
- abstention when evidence is insufficient.

Experiment 008 already showed why restoration alone is insufficient: non-root restorations can also produce recovery.

Experiment 009 is the current ground-truth laboratory for developing this layer.

### Contribution C — MRF-Search

An optional cost-aware diagnosis method.

Given many candidate changes and costly retraining interventions, MRF-Search would select which intervention to run next and stop when enough evidence exists.

This should be treated as an **application/adaptation of sequential experimental design**, not as a claim to have invented adaptive experimentation.

The research question is:

> Can a diagnosis strategy reach the same causal conclusion as exhaustive intervention while using materially fewer training runs or less compute?

MRF-Search proceeds only after MRF-Bench and MRF-Certify are scientifically stable.

---

## 6. Experiment 009 role

> **Current-state note (2026-09-25):** this section records the architecture that led into Exp009. Nuisance-v2 construction, Stage A, and Stage-B authorization have since occurred. Use `research/STATE.md` and `research/ROADMAP.md` for current execution state; do not reinterpret the historical sequencing below as an uncompleted prerequisite list.


Experiment 009 is not the final product.

It is the first serious natural-language **ground-truth causal-regression case**.

Its purpose is to establish that a controlled regression can be:

- prospectively constructed;
- localized;
- protected from unrelated degradation;
- replicated across stochastic trajectories;
- compared against nuisance restorations;
- causally evaluated without post-hoc threshold rescue.

### 6.1 Frozen facts that remain valid

- Banking77 development substrate is frozen.
- Official Banking77 test remains untouched.
- DistilBERT model/revision is pinned.
- Clean tuning is closed.
- Target pair is fixed for the current pilot.
- (1/4) root corruption is accepted for the pilot.
- Three-trajectory root-regression replication passed the frozen Stage-2 rule.
- Nuisance-selection rule v1 is permanently recorded as infeasible.
- No confirmatory certification has been run.

### 6.2 What Exp009 must still establish

Before calling Exp009 complete:

1. Redesign nuisance construction **scientifically**, not by relaxing rules until four candidates appear.
2. Freeze nuisance rule v2 before nuisance-model outcomes are observed.
3. Freeze a confirmatory protocol.
4. Run result-bearing baseline/candidate/restoration siblings on one consistent backend regime.
5. Measure root and nuisance recovery distributions.
6. Apply prospectively frozen certification rules.
7. Preserve failure if unique causal certification does not hold.
8. Touch the official Banking77 test only according to the frozen confirmatory protocol.

### 6.3 What Exp009 should not be forced to do

Exp009 does **not** need to contain every future form of heterogeneous pipeline change.

It is acceptable for Exp009 to remain:

> one natural-language task, one planted root family, multiple legitimate nuisance changes, repeated stochastic trajectories, exhaustive ground-truth intervention.

Heterogeneous changes belong primarily in later benchmark expansion.

---

## 7. MRF-Bench v0.1 design

The first benchmark version should remain small enough to reproduce on commodity research hardware.

### 7.1 Minimum benchmark shape

MRF-Bench v0.1 should aim for:

- more than one regression case;
- more than one causal mechanism;
- at least one natural-language task;
- at least one second substrate or clearly distinct benchmark environment;
- single-root cases first;
- repeated trajectories;
- exhaustive intervention truth for every included case.

Potential regression families include:

- label/taxonomy mapping;
- sampling/reweighting;
- preprocessing transformation;
- training configuration;
- base checkpoint or initialization choice.

The exact second dataset/model is **not frozen yet**.

Selection criteria should include:

- manageable compute on the M3 Max;
- permissive/public licensing;
- deterministic reconstruction;
- sufficient label/slice structure for protected metrics;
- enough examples for repeated retraining;
- relevance to ordinary supervised/post-training workflows.

### 7.2 Interactions

Multi-change interactions are scientifically important but should be treated as a stretch after single-root evaluation is reliable.

A later benchmark may include cases where:

- (A) alone does not cause the regression;
- (B) alone does not cause the regression;
- (A+B) does.

The public abstraction should eventually support root sets even if v0.1 evaluates single-root cases first.

---

## 8. Baselines

MRF must compare against baselines that match the information available to the method.

At minimum:

### Diagnosis/ranking baselines

- random candidate ordering;
- simple change-size/provenance heuristics;
- semantic/representation similarity where applicable;
- task-aware heuristics already developed in earlier MRF experiments.

### Data-attribution baselines

On cases where the candidate unit is training data and assumptions are compatible:

- TracIn;
- TRAK;
- other tractable attribution methods justified by the final literature review.

These methods should **not** be forced onto non-data changes where their assumptions do not apply.

### Intervention baselines

- exhaustive one-at-a-time restoration;
- fixed uniform replication across candidate interventions;
- simple successive-elimination/sequential baselines if MRF-Search is studied.

### Oracle

Exhaustive matched counterfactual retraining remains the ground-truth reference where feasible.

---

## 9. Evaluation metrics

The benchmark should report more than top-1 accuracy.

Core metrics should include:

- root identification accuracy;
- top-k localization accuracy;
- false-certification rate;
- abstention rate;
- target recovery effect;
- protected-metric drift;
- root-vs-nuisance recovery margin;
- variability across trajectories;
- calibration/coverage of confidence claims where applicable;
- number of intervention runs;
- total compute cost or a normalized proxy;
- wall-clock cost as a secondary engineering metric;
- robustness across benchmark cases.

For MRF-Search specifically:

- identification accuracy versus exhaustive ground truth;
- intervention savings;
- compute savings;
- incorrect early-stop rate;
- false-certification rate.

Numerical success thresholds for Exp010 must be frozen **before** seeing Exp010 results.

---

## 10. Proposed Exp010

Working title:

> **Experiment 010 — Budgeted Causal Regression Search**

Exp010 should not begin until:

- the literature dossier is complete enough to rule out obvious duplication;
- MRF-Bench has at least one completed ground-truth case;
- MRF-Certify has a stable interface;
- evaluation metrics and baselines are frozen.

### 10.1 Question

> Can a sequential intervention policy recover the same causal conclusion as exhaustive counterfactual testing using fewer intervention runs?

### 10.2 Candidate approach

The first method should be deliberately simple and interpretable.

Possible ingredients:

- initial candidate priors from cheap evidence;
- paired recovery observations;
- confidence intervals/posteriors over candidate recovery effects;
- successive elimination;
- cost-aware intervention choice;
- explicit abstention;
- stopping when separation is sufficient.

Sophisticated Bayesian/bandit machinery should only be added if simple baselines are inadequate.

### 10.3 Important novelty rule

If standard best-arm identification or causal-bandit machinery solves the benchmark essentially unchanged, MRF should **use and cite it**, not rename it as a new algorithm.

The paper can still contribute the benchmark, formulation, empirical analysis, and failure modes.

---

## 11. Open-source library direction

The open-source library remains a good idea, but only as a **small research library derived from stable abstractions**.

The library is not the primary contribution and should not outrun the research.

### 11.1 Current repository strengths

The repository already has:

- a real `src/model_forensics` package;
- `pyproject.toml`;
- `uv.lock`;
- 196 passing tests at the current authoritative Exp009 state;
- reusable certification/diagnosis/evaluation/lineage modules;
- rigorous experiment records.

### 11.2 Current library gaps

Before calling the project a usable open-source library, it still needs:

- a stable public API;
- an OSI-approved license;
- CI;
- user-facing documentation;
- a minimal quickstart;
- versioned releases;
- packaging metadata;
- citation metadata;
- separation of experiment-specific code from public abstractions.

### 11.3 Target public abstractions

The eventual public API should revolve around concepts such as:

- `RegressionCase`;
- `Change`;
- `MetricSpec`;
- `Intervention`;
- `Trajectory`;
- `InterventionResult`;
- `Diagnosis`;
- `Certificate`;
- benchmark-case interfaces;
- training/evaluation runner protocols.

A future usage pattern might conceptually look like:

```python
case = RegressionCase(
    baseline=...,
    candidate=...,
    changes=...,
    target_metric=...,
    protected_metrics=[...],
)

evidence = run_interventions(case, ...)
certificate = certify(case, evidence)
```

The library should remain training-framework-light where practical: users provide training/evaluation callbacks or adapters rather than MRF becoming another training framework.

### 11.4 What remains experiment code

Files whose purpose is specifically to reproduce Exp007/008/009 should remain research artifacts and need not become public API.

The library should be **extracted from repeated patterns that survive the research**, not designed around one experiment.

---

## 12. Public research artifact

The final public MRF project should be understandable without reading ten experiment directories.

The public story should eventually be:

> **Model Regression Forensics is an open-source research toolkit and benchmark for investigating which versioned training change caused a model regression and testing that diagnosis through controlled stochastic counterfactual retraining.**

The project should expose:

1. a concise README;
2. a five-minute conceptual overview;
3. benchmark specification;
4. reproducible examples;
5. headline research results;
6. paper/technical report;
7. library documentation;
8. full experiment history for auditability.

The chronology remains available, but it becomes supporting evidence rather than the primary onboarding path.

---

## 13. Publication strategy

### Primary research output

Target a serious manuscript/preprint covering:

- problem formulation;
- related work;
- benchmark construction;
- certification methodology;
- baseline comparisons;
- stochasticity analysis;
- negative results/failure modes;
- limitations.

A rolling venue such as TMLR is a plausible fit because its acceptance criteria emphasize whether claims are supported by convincing evidence and whether the findings are useful to part of the ML community.

Future evaluation/benchmark venues are also conceptually aligned. NeurIPS 2026 explicitly treated evaluation methodology, benchmark analysis, evaluation tools, negative results, and stress testing as scientific contributions.

### Software publication

JOSS is a possible later target only after the library becomes genuine reusable research software with:

- an OSI-approved license;
- documentation;
- tests;
- CI;
- releases;
- substantive research use;
- preferably some external use/contribution.

JOSS should not drive the current research agenda.

---

## 14. Portfolio objective

The project should build a broader professional identity than “model regression specialist.”

Desired signal:

> **ML research engineer focused on reliable evaluation, model behavior, reproducible experimentation, and debugging complex training systems.**

MRF should visibly demonstrate:

- PyTorch/Hugging Face;
- model training;
- controlled experiments;
- eval design;
- statistical reasoning;
- reproducibility;
- causal reasoning;
- research software engineering;
- debugging;
- technical writing;
- preservation of negative results.

This maps directly onto contemporary research-engineering work involving model failures, evals, training pipelines, experiment reliability, and reusable research infrastructure.

---

## 15. Research integrity rules

The following remain mandatory:

1. Negative results stand.
2. No threshold is changed after observing the outcome it governs.
3. Technical harness fixes must not alter scientific configuration.
4. Development and confirmatory evidence remain separate.
5. Official test sets remain embargoed until the relevant protocol is frozen.
6. Do not silently replace failed benchmark rules.
7. Do not claim novelty until prior-art review supports the claim.
8. Paired result-bearing comparisons must use the same backend regime.
9. Every benchmark case must preserve provenance and reproducibility.
10. A method may abstain; unsupported certification is worse than no certification.
11. Baselines must receive comparable information.
12. Incompatible prior methods should not be forced into misleading comparisons.
13. No result is rescued post hoc.

---

## 16. Decision / kill criteria

MRF should pivot or stop a line of research if any of the following occurs.

### Novelty kill

Direct prior work already solves essentially the same:

- version-diff candidate space;
- stochastic intervention;
- root-vs-nuisance causal certification;
- cost-aware sequential search;

with no meaningful benchmark or empirical gap left for MRF.

### Benchmark kill

The benchmark can only be made solvable through artificial shortcuts or post-hoc tuning, or cannot create stable localized regressions under prospectively frozen rules.

### Certification kill

After one scientifically justified nuisance redesign, Exp009 still cannot produce interpretable root-vs-nuisance causal evidence and no useful methodological lesson remains.

### Search-method kill

Adaptive search provides no meaningful intervention/compute reduction over strong simple baselines at comparable correctness.

This would kill MRF-Search, **not necessarily MRF-Bench or MRF-Certify**.

### Library kill

The reusable API cannot be separated cleanly from experiment-specific code or offers no utility beyond reproducing the paper.

In that case, release reproducible research code without pretending it is a general library.

---

## 17. Definition of project success

MRF is successful if it reaches a coherent endpoint with most of the following:

- one defensible research/evaluation contribution;
- one reproducible benchmark with known causal truth;
- one rigorous causal-certification protocol;
- serious baseline comparisons;
- one paper-quality technical report/preprint or peer-reviewed submission;
- one installable, documented research package **if the abstractions prove reusable**;
- one clear public project page;
- a reproducible lightweight example;
- preserved experimental history and negative results.

A startup is **not** required.

A top-conference acceptance is **not** required.

GitHub stars are **not** a success criterion.

---

## 18. Repository strategy

The current authoritative scientific state remains:

`exp009-stochastic-counterfactual-certification`

at:

`73e444cddd62a4fb0991429010aeb402b1570cc6`

Planning work should occur off that exact state so Exp009 evidence is not casually disturbed.

The default `main` branch currently diverges from Exp009 and contains a README-only commit whose Exp009 status is now stale.

Do **not** blindly merge or cherry-pick that README commit.

After the Research Plan v2 is finalized and the next experimental direction is frozen:

1. reconcile branch history deliberately;
2. update the public README to the current scientific truth;
3. make the default branch the authoritative public state;
4. preserve experiment branches as historical research records.

---

## 19. Immediate next phase

No new model training should begin yet.

The next tasks are:

### A. Complete the Thesis Validation Dossier

Expand `research/RELATED_WORK.md` into a systematic matrix covering:

- training-data attribution;
- stochastic attribution;
- model diffing;
- model-update regression;
- ML-pipeline RCA;
- lineage/experiment tracking;
- active experimental design;
- causal bandits/best-arm identification;
- evaluation/benchmark design.

For each work record:

- input;
- output;
- diagnosis unit;
- access assumptions;
- gradients/checkpoints required;
- intervention type;
- stochastic treatment;
- version-awareness;
- multi-change support;
- cost-awareness;
- benchmark structure;
- overlap/threat to MRF.

### B. Freeze the benchmark contract

Define the exact schema and evaluation protocol for an MRF benchmark case.

### C. Diagnose nuisance-rule v1 failure

Determine which v1 constraints eliminated candidate nuisance pairs.

This is analysis only.

Do not implement v2 until the benchmark contract explains what a nuisance must represent.

### D. Decide Exp009 continuation protocol

Only after A–C should nuisance v2 and confirmatory Exp009 work resume.

### E. Freeze Exp010 only after Exp009 ground truth exists

No Exp010 implementation before then.

---

## 20. Current decision

**CONTINUE MRF as a research/portfolio project.**

Primary direction:

> **MRF-Bench + MRF-Certify**

Secondary/conditional direction:

> **MRF-Search**

Open-source direction:

> **small research library extracted from stable methodology**

Not a current objective:

> startup/product engineering.

The project now optimizes for:

> **research credibility × ML-research-engineering signal × reproducibility × reusable scientific value per unit effort.**
