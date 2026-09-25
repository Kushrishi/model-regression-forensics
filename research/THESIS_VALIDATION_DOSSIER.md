# Model Regression Forensics — Thesis Validation Dossier v0.1

Status: **adversarial prior-art review / research gate**
Date: **2026-09-20**
Branch: `research-plan-v2`
Starting scientific state: `73e444cddd62a4fb0991429010aeb402b1570cc6`

## Executive conclusion

The broad idea behind Model Regression Forensics is **not novel**:

- training-data attribution is a mature field;
- counterfactual retraining has long been used as a gold-standard notion of influence;
- stochastic training has explicitly been incorporated into modern data-attribution work;
- model-update regressions and negative flips are established research topics;
- behavioral model diffing exists;
- ML-pipeline root-cause analysis with counterfactual repair exists;
- experiment/data/model lineage is already handled by mature MLOps systems;
- adaptive intervention selection and best-arm identification are established optimization problems.

However, the reviewed literature does **not yet establish that MRF's narrower combined evaluation problem is saturated**.

The current working white-space hypothesis is:

> **Benchmarking and certifying post-hoc causal diagnosis of a model regression over an explicit finite version-diff search space, using hidden known causal truth, repeated matched stochastic retraining, root-vs-nuisance intervention comparison, abstention, and intervention-cost accounting.**

This is a **working gap**, not a novelty claim.

The strongest recommended project direction is therefore:

1. **MRF-Bench first** — make causal regression diagnosis a reproducible evaluation problem;
2. **MRF-Certify second** — formalize when intervention evidence is sufficient and when the system must abstain;
3. **MRF-Search third and conditionally** — study intervention-efficient search only if it adds value beyond standard sequential-design/best-arm methods.

This ordering protects the project from depending on a fragile claim that adaptive search itself is novel.

---

# 1. Exact problem MRF is trying to own

MRF should not ask:

> Which individual training examples influenced this model output?

That is training-data attribution.

MRF should not ask:

> What behaviors differ between model A and model B?

That is model diffing/evaluation.

MRF should not ask:

> What parameters, code, datasets, and artifacts differed between two runs?

That is lineage/experiment tracking.

MRF should not ask:

> Which hyperparameter would improve expected performance?

That is tuning / pipeline optimization / counterfactual repair.

The working MRF problem is:

> **A known-good release became a regressed release. A finite set of concrete versioned training-system changes occurred between them. Which change or change set actually caused the regression, how strong is the causal evidence under stochastic retraining, and how much intervention cost was required to reach that conclusion?**

A benchmark instance should therefore distinguish four layers:

1. **Detection** — the target behavior regressed.
2. **Lineage** — several concrete changes occurred.
3. **Localization** — evidence ranks one or more candidate changes as suspicious.
4. **Certification** — controlled interventions support or fail to support a causal explanation.

MRF's strongest methodological claim may ultimately be that layers 3 and 4 must be evaluated separately.

---

# 2. Capability matrix

Legend:

- **Yes** — central supported capability.
- **Partial** — relevant but not the main unit/problem.
- **No** — not a central capability in the reviewed work.
- **N/A** — not meaningful for that work.

| Work/system | Primary unit | Main input/access | Main output | Real intervention/retraining? | Explicit stochastic-training treatment? | Version-diff aware? | Heterogeneous change causes? | Intervention-cost aware? | Threat to MRF |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Koh & Liang, Influence Functions (ICML 2017) | Training example | Gradients + Hessian-vector products | Example influence on prediction/loss | Approximate leave-one-out, not full version RCA | No | No | No | Motivated by avoiding retraining | Foundational overlap on attribution |
| TracIn (NeurIPS 2020) | Training example | Gradients + saved checkpoints | Influence score | No full counterfactual rerun required | Training path sampled through checkpoints, but not distributional RCA | No | No | Computationally motivated | Attribution baseline |
| Datamodels (ICML 2022) | Training-set membership | Many subset-training outcomes | Predict outcome under dataset subsets | Uses many retrainings to fit datamodel | Repeated trials can reduce noise, but not version-RCA framing | No | Data only | Strong compute motivation | Serious counterfactual-data overlap |
| TRAK (ICML 2023) | Training example | Differentiable model + a handful of trained models | Scalable data attribution | Avoids exhaustive retraining | Not the central framing | No | No | **Yes, computational tractability central** | Strong baseline / cost threat |
| Final-Model-Only Data Attribution (2024) | Training example | Final model only | Attribution against further-training gold standard | **Yes, further training used as gold standard** | Averaging is part of the gold-standard construction | No | No | Partial | Overlap on retraining-as-ground-truth |
| Distributional TDA (NeurIPS 2025) | Training example / data distribution | Training algorithm + stochastic runs | Effect of data on output distribution | Framework targets distributional effect | **Yes, central contribution** | No | No | Partial | Kills stochasticity-as-novelty claim |
| DATE-LM (NeurIPS 2025) | Attribution method | Attribution methods + LLM tasks | Benchmark scores across applications | Depends on method/task | Partial | No | Data only | Evaluates method trade-offs | Strong precedent for benchmark contribution |
| What is Your Data Worth to GPT? / LoGra (NeurIPS 2025) | Training example/data value | Gradients; projected influence machinery | Scalable LLM data valuation | No full release-level intervention | No central stochastic-RCA treatment | No | No | **Yes, throughput/memory central** | Scalable attribution baseline |
| Li et al., Do Influence Functions Work on LLMs? (EMNLP Findings 2025) | Training example | Influence-function variants | Empirical failure analysis | Compares attribution behavior | Highlights convergence/behavior mismatch | No | No | Discusses scaling constraints | Important negative baseline evidence |
| Goodfire Probe-Based Data Attribution (2026) | Training datapoint | Activations/probes + post-training data | Datapoints linked to harmful behavior | **Yes — filter and retrain** | Not central | Version context only indirectly | Primarily data | **Yes — reports lower cost** | Very strong data-debugging threat |
| Goodfire Predictive Data Debugging (2026) | Preference-data cluster/example | Preference dataset + interpretability features | Predict learned behavioral effects; trace to data | **Yes — reshape data/training and validate** | Not central framing | Not generic version-diff RCA | Primarily preference data/training signal | **Yes — avoid train/guess loops** | Very strong post-training threat |
| DebugLM (2026) | Data source/stage | Model explicitly trained with provenance tags | Source provenance for behavior + targeted remediation | Provenance learned during training; remediation can be test-time | Not central | Multi-stage provenance | Data sources/stages, not generic config diff | Partial | Adjacent provenance approach |
| Anthropic model diff tool (2026) | Behavior/model representation | Two model versions | Automatically surfaced behavioral differences | Feature steering / analysis, not root-cause intervention over pipeline changes | No | **Yes — compares models** | No external pipeline root cause | Not central | Strong detection/diff adjacency |
| Xie et al., Regression Bugs (2021) | Prediction/example negative flip | Old + new model predictions | Regression metrics + mitigation | Training methods to reduce regression | Notes SGD/update variability but not causal RCA | **Yes — model update** | Update factors discussed, not diagnosed causally | No | Establishes regression problem prior art |
| Backward-Compatible Prediction Updates (NeurIPS 2021) | Prediction update | Old/new model predictions + limited reevaluation budget | Which predictions to update while avoiding negative flips | Selective reevaluation rather than root-cause retraining | Probabilistic treatment | **Yes** | No root-cause search | **Yes — explicit budget** | Cost-aware model-update adjacency |
| MUSCLE (EMNLP Findings 2024) | Model-update compatibility | Old/new LLM base models + downstream adapters | Reduce negative flips | **Yes — compatibility adapter training** | Not RCA-focused | **Yes** | Model/base changes, but not cause identification | No | Strong regression-mitigation adjacency |
| Dapaah & Grabowski, From diagnosis to repair (2026) | Pipeline descriptor / hyperparameter | Dataset complexity descriptors + configurations | Root-cause contributions + predicted counterfactual repair | **Yes — predicted intervention validated by reruns** | Not central | Partial | **Yes — dataset descriptors + hyperparameters** | Partial | **Closest system-level novelty threat** |
| MLflow Tracking | Run/model/artifact | Parameters, code versions, metrics, artifacts | Run comparison / lineage / reproducibility | No causal intervention | No | **Yes — tracks versions/runs** | **Yes — metadata can cover many changes** | No | Defines lineage boundary MRF should not duplicate |
| Comet Artifact Lineage / Model Registry | Data/model/run lineage | Versioned experiments/artifacts/models | Lineage, comparison, registry | No causal intervention | No | **Yes** | **Yes at metadata level** | No | Same boundary as MLflow |
| Cleanlab | Data issue/example/class | Predicted probabilities/embeddings/labels | Label/data issue detection and curation | Can retrain after cleaning, but not version-root certification | No | No | Primarily data quality | Partial | Strong data-debugging product/library adjacency |
| Sen et al., Best Interventions (ICML 2017) | Intervention/arm | Causal graph + intervention samples | Best intervention under budget | **Yes** | Stochastic outcomes | No ML-version framing | Generic causal intervention | **Yes, central** | Kills generic budgeted-intervention novelty |
| Zhang et al., Active Learning for Optimal Intervention Design (Nat. Mach. Intell. 2023) | Intervention | Causal model + sequential observations | Efficient intervention choice | **Yes** | Sequential uncertainty | No | Generic intervention space | **Yes** | Strong active-design adjacency |
| Li & Cheung, BAI with Resource Constraints (AISTATS 2024) | Arm/alternative | Stochastic arm outcomes + resource consumption | Best arm under resource constraints | Sampling interventions | **Yes** | No | Generic alternatives | **Yes, central** | Strong MRF-Search prior art |
| Komiyama et al., Anytime BAI (AISTATS 2026) | Arm/alternative | Sequential stochastic samples | Rate-optimal best-arm identification | Sampling | **Yes** | No | Generic | **Yes** | Current sequential-search baseline family |

---

# 3. Detailed findings by research neighborhood

## 3.1 Training-data attribution

### Influence Functions — Koh & Liang (ICML 2017)

Primary source:
https://proceedings.mlr.press/v70/koh17a.html

Contribution:

- traces a model prediction back to responsible training points;
- approximates the effect of upweighting/removing training examples;
- requires gradient/Hessian-vector-product machinery;
- explicitly motivates model debugging and dataset error detection.

MRF consequence:

> Attribution to training examples is old and cannot be claimed as an MRF contribution.

MRF distinction:

- candidate unit is intended to be a **versioned system change**, not necessarily an example;
- a candidate can eventually be a sampler, preprocessing transform, optimizer schedule, checkpoint, or code/config change;
- MRF evaluates a release-level causal hypothesis, not only example influence.

### TracIn — Pruthi et al. (NeurIPS 2020)

Primary source:
https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html

Contribution:

- traces influence through gradient-descent checkpoints;
- requires gradients, checkpoints, and loss functions;
- provides a scalable training-example influence estimate.

MRF consequence:

TracIn is a plausible **cheap diagnostic signal/baseline** on data-change cases.

It is not a general comparator for optimizer/code/config changes.

### Datamodels — Ilyas et al. (ICML 2022)

Primary source:
https://proceedings.mlr.press/v162/ilyas22a.html

Contribution:

- learns a function from training-set membership to model output;
- predicts effects of dataset counterfactuals;
- demonstrates that relatively simple models can approximate a complicated train-and-evaluate process.

Novelty threat:

This is important because it weakens any MRF claim like:

> “We are the first to predict what retraining under a data change would do.”

MRF distinction:

Datamodels' intervention space is subsets of a dataset. MRF's proposed search space is release/version changes that can be heterogeneous.

### TRAK — Park et al. (ICML 2023)

Primary source:
https://proceedings.mlr.press/v202/park23c

Contribution:

- scalable data attribution for differentiable models;
- explicitly targets the tension between accurate attribution and the huge cost of retraining thousands of models;
- uses only a handful of trained models to approximate expensive attribution approaches.

MRF consequence:

This is one of the most important baselines for any data-focused MRF case.

MRF must **not** claim:

> “We uniquely reduce retraining cost for causal/data diagnosis.”

A future MRF-Search result must show value on the version-diff diagnosis task, not just cheaper attribution.

### Final-Model-Only Data Attribution — Wei et al. (2024)

Primary source:
https://arxiv.org/abs/2412.03906

Contribution:

- defines a further-training procedure as a gold standard for attribution when only a final model is available;
- uses adjustment and averaging;
- evaluates gradient methods by how well they approximate this gold standard.

MRF consequence:

Using training/retraining as a gold-standard intervention is not novel by itself.

Potential MRF distinction:

MRF-Certify compares **candidate version-change interventions against each other and against stochastic variability**, rather than assigning influence to individual examples.

### Distributional Training Data Attribution — Mlodozeniec et al. (NeurIPS 2025)

Primary source:
https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html

Contribution:

- explicitly treats randomness in initialization and batching as fundamental;
- models how the distribution of model outputs over training runs changes with the dataset.

MRF consequence:

> “MRF is novel because it accounts for stochastic training” is false.

Potential MRF distinction:

Stochasticity is a **nuisance variable in causal certification over versioned changes**, not the attributed object itself.

### DATE-LM — Jiao et al. (NeurIPS 2025)

Primary source:
https://proceedings.neurips.cc/paper_files/paper/2025/hash/e1ebda145808ca45774993fb67314894-Abstract-Datasets_and_Benchmarks_Track.html

Contribution:

- a unified benchmark for data-attribution methods;
- evaluates methods across several practical tasks;
- finds no single method dominates and that evaluation design matters substantially.

MRF consequence:

This is favorable precedent for **MRF-Bench**.

A benchmark/evaluation contribution can be scientifically valuable even when the underlying component methods already exist.

MRF-Bench must still define a genuinely different unit of evaluation.

### LoGra / What is Your Data Worth to GPT? — Choe et al. (NeurIPS 2025)

Primary source:
https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6d26053b977f8c589669fd201615119-Abstract-Conference.html

Contribution:

- makes influence-function-based data valuation much more scalable for LLMs;
- reports large throughput and memory improvements;
- releases LogIX as reusable software.

MRF consequence:

Scalability of gradient attribution is an existing engineering area. MRF should consume these methods as candidate signals/baselines rather than compete on attribution throughput.

### Do Influence Functions Work on LLMs? — Li et al. (EMNLP Findings 2025)

Primary source:
https://aclanthology.org/2025.findings-emnlp.775/

Contribution:

- systematically evaluates influence functions on LLM tasks;
- finds poor performance in many settings;
- identifies approximation error, fine-tuning convergence, and parameter-change/behavior-change mismatch as problems.

MRF consequence:

This strengthens the case that **behavioral causal validation matters** and that an attribution ranking alone should not be accepted as a cause.

It does not by itself establish MRF's benchmark novelty.

---

## 3.2 Goodfire and modern post-training data debugging

### Probe-Based Data Attribution — Xiao & Aranguri (Goodfire, 2026)

Primary source:
https://www.goodfire.com/research/probe-based-data-attribution

Contribution:

- surfaces undesirable post-training behavior;
- links behavior to responsible datapoints using activation-space probes;
- filters identified datapoints and retrains;
- reports a substantial reduction in the harmful behavior;
- explicitly describes the retraining step as causal validation.

This is a **high-severity novelty threat** to any MRF framing based on:

> “Find the bad training data and prove it by removing the data and retraining.”

That is already done.

Potential MRF distinction:

- explicit release-diff candidate space;
- non-data causes;
- candidate-vs-nuisance certification;
- repeated stochastic intervention distributions;
- benchmark-level false-certification/abstention evaluation.

### Predictive Data Debugging — Goodfire (June 2026)

Primary source:
https://www.goodfire.com/research/predictive-data-debugging

Contribution:

- predicts which behaviors preference data will amplify/suppress before training;
- traces behaviors to responsible data;
- supports reshaping the dataset/training process;
- validates predictions against actual learned behavior.

This further closes the door on a generic “training-data debugging” identity for MRF.

MRF should be positioned as **versioned regression forensics**, not as a competitor to data-debugging systems.

---

## 3.3 Provenance-first systems

### DebugLM — Mo et al. (2026)

Primary source:
https://arxiv.org/abs/2603.17884

Contribution:

- trains LLMs to associate behavior with provenance tags;
- traces behaviors to training-data sources;
- supports targeted test-time remediation.

MRF consequence:

Provenance can be made a learned property of the model itself.

MRF's setting should remain:

> post-hoc investigation of ordinary training systems where causal provenance was **not deliberately encoded into the model during training**.

---

## 3.4 Model diffing and update regression

### Anthropic model diff tool (2026)

Primary source:
https://www.anthropic.com/research/diff-tool

Contribution:

- automatically surfaces behavioral differences between new and prior model versions;
- motivated directly by the software-diff analogy.

Threat:

The analogy “Git diff for models” is not ownable.

MRF distinction:

Anthropic's tool asks:

> What changed in model behavior?

MRF asks:

> Which external training-system change caused a detected regression?

Behavioral diffing may become an MRF input/signal.

### Regression Bugs Are In Your Model! — Xie et al. (2021)

Primary source:
https://arxiv.org/abs/2105.03048

Contribution:

- measures model-update regression using negative flips;
- shows regression is prevalent in NLP updates;
- proposes methods to reduce regression.

MRF consequence:

The phenomenon “a newer model regresses on previously correct behavior” is established prior art.

MRF's contribution cannot be regression measurement alone.

### Backward-Compatible Prediction Updates — Träuble et al. (NeurIPS 2021)

Primary source:
https://proceedings.neurips.cc/paper/2021/hash/012d9fe15b2493f21902cd55603382ec-Abstract.html

Contribution:

- asks which old predictions should be recomputed with a new model under a limited budget;
- tries to avoid negative flips probabilistically.

MRF consequence:

Even **cost-aware decisions around model updates** are not unique.

MRF-Search must remain tied to causal root-cause investigation rather than generic update selection.

### MUSCLE — Echterhoff et al. (EMNLP Findings 2024)

Primary source:
https://aclanthology.org/2024.findings-emnlp.430/

Contribution:

- demonstrates negative flips when pretrained LLM base models are updated;
- introduces compatibility metrics and a training strategy to reduce regression.

MRF consequence:

MRF is diagnosis, not primarily regression-prevention/compatibility training.

---

## 3.5 Pipeline-level RCA

### Dapaah & Grabowski — From diagnosis to repair (2026)

Primary source:
https://link.springer.com/article/10.1007/s11334-026-00642-8

This is currently the **single closest prior-art threat to MRF's broader system-level framing**.

Contribution:

- treats ML-pipeline performance diagnosis as a structured supervised/meta-learning problem;
- uses dataset-complexity descriptors and hyperparameter configurations;
- estimates interpretable contributors to success/failure;
- builds a structural causal model;
- evaluates counterfactual repair queries before retraining;
- validates counterfactual predictions with rerun experiments.

Overlap with MRF:

- pipeline-level rather than purely example-level;
- interpretable root-cause factors;
- heterogeneous descriptor/config space;
- counterfactual intervention;
- validation through reruns.

Important distinctions to test:

1. MRF begins from an explicit **known-good release → regressed release diff**, not a general population of pipeline runs.
2. MRF's candidate space is intended to be the concrete versioned changes between those releases.
3. MRF's ground truth is a hidden planted/known cause in a controlled benchmark.
4. MRF evaluates causal certification against **nuisance interventions**.
5. MRF explicitly repeats matched stochastic retraining trajectories.
6. MRF measures false certification / abstention, not only performance prediction or contributor attribution.
7. MRF may study intervention-count/compute efficiency as part of diagnosis.

These distinctions must survive a deeper full-paper read before publication claims are made.

---

## 3.6 Lineage / experiment tracking

### MLflow Tracking

Primary source:
https://mlflow.org/docs/latest/ml/tracking

MLflow records:

- parameters;
- code versions;
- metrics;
- output artifacts;
- run metadata.

MRF consequence:

MRF should **not build another experiment tracker**.

In a real deployment, MRF would ideally consume lineage from systems like MLflow rather than replace them.

Lineage answers:

> What changed?

MRF's proposed role begins after that:

> Which change caused the regression?

### Comet Artifact Lineage / Model Registry

Primary sources:
https://www.comet.com/docs/v2/guides/artifacts/artifacts-lineage/
https://www.comet.com/docs/v2/guides/model-registry/quickstart/

Comet tracks versioned datasets/models/experiments and exposes artifact lineage.

MRF consequence:

Again, version discovery and metadata provenance are existing infrastructure.

MRF's research should assume a finite candidate set is available rather than spending its core contribution on inventorying changes.

---

## 3.7 Data-quality tooling

### Cleanlab

Primary source:
https://docs.cleanlab.ai/

Cleanlab detects:

- label issues;
- outliers;
- near-duplicates;
- class/overlap problems;
- other data-quality issues.

MRF consequence:

MRF should not position itself as a data-quality scanner.

Cleanlab could plausibly become one **candidate-evidence source** when a release diff includes changed training data.

---

## 3.8 Active intervention selection / best-arm identification

### Sen et al. — Identifying Best Interventions (ICML 2017)

Primary source:
https://proceedings.mlr.press/v70/sen17a.html

Contribution:

- formalizes best intervention selection as a best-arm problem;
- includes fixed total sampling budgets and intervention costs;
- exploits information shared among intervention arms.

MRF consequence:

The abstract concept:

> “Choose interventions adaptively under a budget”

is old.

### Zhang et al. — Active Learning for Optimal Intervention Design (Nature Machine Intelligence 2023)

Primary source:
https://www.nature.com/articles/s42256-023-00719-0

Contribution:

- sequential experimental design over interventions;
- uses causal structure to identify desirable interventions efficiently when exhaustive search is infeasible.

MRF consequence:

The argument “exhaustive intervention is expensive, therefore actively choose experiments” is not novel.

### Li & Cheung — Best Arm Identification with Resource Constraints (AISTATS 2024)

Primary source:
https://proceedings.mlr.press/v238/li24c.html

Contribution:

- best-arm identification when different arms consume limited resources;
- develops successive-halving/resource-rationing algorithms.

MRF consequence:

Heterogeneous intervention cost has mature algorithmic analogues.

### Komiyama et al. — Rate-optimal Anytime Best Arm Identification (AISTATS 2026)

Primary source:
https://proceedings.mlr.press/v300/komiyama26a.html

Contribution:

- modern rate-optimal work on best-arm identification under limited sampling.

MRF consequence:

MRF-Search should begin with standard sequential-identification baselines rather than inventing a new heuristic and claiming novelty.

---

# 4. What is definitely NOT novel

The following claims should be prohibited in future MRF writing unless materially qualified:

> “MRF is the first system to trace model behavior back to training data.”

False/unsupported.

> “MRF is the first to validate attribution by retraining.”

False/unsupported.

> “MRF is the first to account for stochasticity in data attribution.”

False.

> “MRF is the first to compare model versions for regressions.”

False.

> “MRF is the first to use causal interventions for ML pipeline debugging.”

False/unsupported.

> “MRF is the first to choose debugging experiments adaptively to reduce cost.”

False as a general algorithmic claim.

> “MRF is a Git diff for machine learning.”

Too broad and already used in adjacent model-diff framing.

---

# 5. What may still be a defensible contribution

## 5.1 MRF-Bench: strongest current candidate

Potential contribution:

> **A benchmark whose unit is a model-release regression incident with an explicit finite version-diff candidate set and hidden known causal truth, where methods are evaluated not only on localization but on intervention-supported causal certification under stochastic retraining.**

What appears unusual in the reviewed set:

- explicit release pair (B ightarrow C);
- concrete debugger-visible versioned changes;
- hidden ground-truth root;
- root plus legitimate nuisance changes;
- exhaustive intervention truth;
- repeated matched training trajectories;
- target + protected metrics;
- false-certification and abstention evaluation;
- cost accounting.

This is currently the **safest research white-space hypothesis**.

It still requires a more exhaustive literature search before a paper-level novelty claim.

## 5.2 MRF-Certify: plausible methodological contribution

Potential contribution:

> **A prospective criterion for deciding when restoration evidence is sufficiently specific to support a causal diagnosis rather than merely demonstrating restorative influence.**

Exp008 provides a strong motivating failure mode:

- true-root restoration recovered the target;
- some non-root restorations also recovered the target;
- therefore recovery alone was insufficient.

Exp009 adds a second problem:

- identical planted change;
- materially different regression effect across stochastic trajectories.

A certification framework that jointly accounts for:

- root recovery;
- nuisance recovery;
- stochastic variation;
- protected behavior;
- abstention;

may be useful even if its statistical ingredients are standard.

Novelty may lie in the **evaluation formulation and empirical analysis**, not necessarily a new estimator.

## 5.3 MRF-Search: highest-risk contribution

Potential contribution:

> **Use cheap model/lineage evidence plus sequential counterfactual outcomes to reduce the number of expensive retraining interventions required for a correct causal diagnosis.**

This is compelling practically but has the weakest standalone novelty because:

- best-arm identification is mature;
- causal best-intervention selection is mature;
- resource-constrained experiment selection is mature;
- attribution methods such as TRAK already attack retraining cost.

MRF-Search should therefore be framed as:

> applying/adapting established sequential-design ideas to the MRF benchmark and measuring whether they actually work.

If a standard method performs well, that is a useful result.

A new algorithm should only be proposed if the benchmark exposes a concrete failure of standard methods.

---

# 6. Why the open-source library still makes sense

The library should package **MRF-Bench and MRF-Certify first**, not an unvalidated search algorithm.

Good library scope:

- regression-case schema;
- change/intervention schema;
- metric and protected-slice schema;
- trajectory pairing;
- evidence ingestion;
- recovery-distribution calculations;
- certification/abstention rules;
- benchmark runners;
- baseline interfaces;
- result reports.

Bad library scope:

- generic experiment tracking;
- generic data cleaning;
- cloud training orchestration;
- dashboards;
- model registry;
- universal MLOps adapters;
- generic attribution reimplementation.

The library can create portfolio value even if the paper contribution is mainly evaluation methodology because it demonstrates the translation from research protocol to reusable software.

---

# 7. Recommended research claims hierarchy

Future writing should separate claims into four levels.

## Level 1 — established by MRF experiments

Examples:

- naive lexical diagnostics can exploit benchmark shortcuts;
- localization does not imply causal influence;
- restorative influence does not imply unique causal specificity;
- the same planted change can yield materially different observed regression across stochastic trajectories;
- prospectively frozen construction rules can fail and should not be relaxed after outcome inspection.

These are claims about **our controlled experiments**, not universal facts.

## Level 2 — established by prior work

Examples:

- training-data influence/attribution is mature;
- stochasticity matters in attribution;
- negative flips occur during model updates;
- counterfactual retraining can be used as influence evidence;
- active intervention selection can reduce experimentation cost.

MRF must cite these rather than re-claim them.

## Level 3 — working MRF hypothesis

Example:

> Version-diff causal certification under stochastic retraining is a useful evaluation problem not well covered by existing benchmarks.

This remains under validation.

## Level 4 — future empirical claim

Example:

> Method X reaches the same causal conclusion as exhaustive intervention using 60% fewer training runs at a fixed false-certification rate.

This can only be claimed after frozen evaluation.

---

# 8. Immediate implications for Experiment 009

Do **not** abandon Exp009.

Its role is now clearer:

> Exp009 is an MRF-Certify / MRF-Bench ground-truth case, not a standalone proof that MRF invented stochastic causal attribution.

Before nuisance v2 is implemented, perform a constraint-level autopsy of v1:

- how many pairs survive the recall floor alone?
- how many survive eval/train count requirements?
- how many survive token-Jaccard overlap?
- how many survive one-way confusion?
- how many survive bidirectional confusion?
- which conjunction first drives eligibility to zero?
- how many disjoint pairs remain after each stage?

Then define what a nuisance is scientifically required to represent.

A nuisance should not merely be:

> “whatever rule gives us four pairs.”

The benchmark contract must answer:

> What makes a non-root change plausible enough that a debugger should have to distinguish it from the root?

Only then can nuisance v2 be frozen.

---

# 9. Immediate implications for Exp010

Do not implement Exp010 yet.

The correct dependency order is:

1. finish dossier;
2. freeze benchmark case schema;
3. finish Exp009 ground truth;
4. implement at least simple diagnosis baselines;
5. evaluate exhaustive intervention;
6. only then design budgeted search.

The first Exp010 baseline should likely be simpler than a new Bayesian method:

- random ordering;
- static heuristic ordering;
- successive elimination using observed recovery;
- standard best-arm-style allocation where assumptions fit.

Only propose a specialized MRF algorithm if these baselines expose a concrete deficiency.

---

# 10. Additional literature questions before a novelty claim

The following searches remain mandatory before publication:

1. Benchmarks specifically for **model regression root-cause diagnosis**.
2. Software-engineering literature on ML regression testing and change-impact analysis.
3. AutoML failure diagnosis beyond the Dapaah/Grabowski paper.
4. Dataset-diff and pipeline-provenance RCA systems from data management literature.
5. Causal debugging of data/ML pipelines in systems/database venues.
6. Multi-factor/interacting causes in ML debugging.
7. Sequential hypothesis testing with abstention under expensive interventions.
8. Counterfactual debugging/repair systems that use version-control diffs as candidate causes.
9. Any 2026 work published after the current reviewed set and before submission.

The literature review should be refreshed again immediately before any public novelty claim.

---

# 11. Research decision after dossier v0.1

## Continue

Yes — but continue under the revised objective.

### Primary project

**MRF-Bench + MRF-Certify**

### Conditional project

**MRF-Search**

### Open-source deliverable

A small research library exposing the stable benchmark/certification methodology.

### Not required

- startup;
- SaaS;
- universal MLOps platform;
- new attribution algorithm;
- new bandit theory.

---

# 12. Current risk assessment

## Low risk to project value

Even if no novel search algorithm emerges, MRF can still produce:

- a careful benchmark;
- empirical lessons about causal certification;
- reusable research software;
- a technical report/preprint;
- a strong ML research-engineering portfolio artifact.

## Medium risk

The benchmark contribution may prove too narrow if realistic heterogeneous cases cannot be constructed without artificial shortcuts.

Mitigation:

- require more than one causal mechanism;
- add a second substrate;
- later reconstruct or source realistic regression incidents where possible.

## High risk

Claiming broad algorithmic novelty for adaptive intervention search.

Mitigation:

- treat existing sequential-design methods as first-class baselines;
- make MRF-Search optional;
- do not invent complexity unless benchmark evidence demands it.

## Highest novelty threat

Dapaah & Grabowski (2026) because it already joins:

- pipeline-level diagnosis;
- structured descriptors;
- causal modeling;
- counterfactual repair;
- rerun validation.

A full claim-by-claim comparison with that paper should be completed before freezing a publication thesis.

---

# 13. Dossier verdict

Current verdict:

> **CONTINUE — EVALUATION/BENCHMARK-FIRST**

Confidence:

**Moderate**, not high.

Reason:

The reviewed literature strongly overlaps with every individual ingredient of MRF, but the exact combination of **version-diff incident structure + hidden root + nuisance interventions + paired stochastic retraining + causal-specificity/abstention evaluation** has not been identified in this review as an existing benchmark or system.

This is enough to justify the next research step.

It is **not** enough to make a public “first” claim.

---

# 14. Next gate

The next artifact should be:

> **MRF Benchmark Contract v0.1**

It must define, independent of Banking77:

- what counts as a regression case;
- what information a debugger is allowed to observe;
- what information remains hidden;
- what a candidate change is;
- what an intervention means;
- how trajectories are paired;
- target/protected metrics;
- ground-truth cause/root set;
- certification outcomes;
- abstention;
- compute/intervention accounting;
- baseline fairness rules.

Only after that contract exists should the Exp009 nuisance design be reconsidered.
