# Experiment 009 — Stochastic Counterfactual Certification

Status: **development protocol only — confirmatory protocol not frozen**.

Parent research state:

`7cc3b6fecc26c2c18f0476bcfa8ca718fb738465`

Experiment 009 begins only after Experiment 008 has been permanently closed.
Experiment 008 must not be modified, retuned, or reinterpreted to make Experiment
009 easier.

This document governs **development and pilot work only**. It is intentionally
not a confirmatory preregistration. Pilot measurements may inform the final
model, corruption dose, trajectory count, margins, and statistical procedure,
but those choices must be frozen in a separate confirmatory protocol before any
confirmatory result is observed.

## Research question

> When a versioned training-data update causes a localized regression on a
> realistic natural-language task, can the responsible update be localized and
> then causally certified by showing that reversing it produces a recovery
> effect that is reliably distinguishable from non-causal updates and ordinary
> retraining variability?

Experiment 009 is motivated directly by Experiment 008.

Experiment 008 established all of the following under its frozen synthetic
benchmark:

- the candidate regression could be localized;
- the planted root ranked first in both frozen worlds;
- restoring the planted root fully repaired the target behavior in both worlds;
- protected behavior remained stable under the root restoration;
- nevertheless, non-root restorations also sometimes produced material target
  recovery.

Experiment 008 therefore showed that one successful root restoration is not, by
itself, sufficient evidence of uniquely specific causality.

Experiment 009 does not attempt to rescue the Experiment 008 shape substrate.
It moves to a realistic natural-language task and treats counterfactual effects
as repeated paired measurements over training trajectories.

## Claim boundary

Experiment 009 is not designed to establish any of the following by itself:

- universal neural-network causal attribution;
- a general solution for arbitrary model regressions;
- a new theory of stochastic training;
- novelty of distributional training-data attribution;
- weights-only root-cause identification;
- production-scale reliability;
- causal identification across arbitrary model families or training stages.

Recent work already treats stochastic training outcomes distributionally,
including Distributional Training Data Attribution and further-training-based
attribution. Experiment 009 therefore does **not** claim that averaging or
comparing effects across stochastic runs is itself novel.

The narrower question is whether a post-hoc regression-forensics workflow can
combine:

1. a baseline and regressed model release;
2. explicit versioned training lineage;
3. several debugger-visible candidate changes;
4. a hidden known root cause;
5. blinded candidate ranking;
6. candidate-by-candidate counterfactual retraining;
7. paired repeated trajectories; and
8. a prospective causal-specificity criterion that distinguishes the root from
   nuisance interventions and ordinary retraining variation.

Novelty remains **not established** until the related-work review is complete.

## Substrate

The first development substrate is **Banking77**.

Banking77 contains 13,083 English banking-support queries labeled with 77
fine-grained intents. The public dataset currently exposes 10,003 training
examples and 3,080 test examples and is distributed under CC BY 4.0.

The exact dataset source, revision, files, and checksums used by Experiment 009
must be pinned in code and recorded before the first result-bearing pilot run.

The official Banking77 test split is **embargoed during development**.

Development code must not load, inspect, summarize, embed, score, search, or use
the official test examples for:

- model selection;
- target-intent selection;
- corruption-dose selection;
- nuisance construction;
- threshold selection;
- trajectory-count selection;
- diagnostic selection;
- statistical-procedure selection.

Only the official training split may be used for pilot development.

## Development partition

A deterministic stratified development partition must be created from the
official Banking77 training split.

The partitioning implementation must be independent of model results.

The intended structure is:

- `development_train`: model fitting and versioned-shard construction;
- `development_eval`: clean baseline evaluation, target-pair development,
  corruption calibration, nuisance calibration, and variance estimation.

The initial implementation should reserve 20 percent of each intent for
`development_eval` and the remainder for `development_train` using a stable,
content-derived ordering rather than relying on source-file row order.

A suitable deterministic key is SHA-256 over the canonical label and text.
The exact canonicalization and rounding rule must be tested and committed before
pilot results are used.

Once created, the development partition must remain stable for the lifetime of
Experiment 009 unless a defect in the partitioning implementation is discovered
and logged before confirmatory freeze.

The official test split remains untouched even after development pair selection.

## Model-development rule

Experiment 009 is permitted to change model architecture relative to
Experiments 004–008.

The initial primary candidate is a small pretrained encoder classifier suitable
for repeated full supervised fine-tuning on 77-way intent classification. A
model such as `distilbert-base-uncased` is the preferred first candidate.

A smaller encoder may be evaluated as a fallback if the preferred model is not
scientifically or computationally suitable.

Model selection must occur using **clean development runs only**.

Permitted model-selection evidence includes:

- clean development macro performance;
- per-intent reliability;
- variation across clean trajectories;
- training runtime;
- peak memory usage;
- reproducibility and implementation stability.

Model selection must not use:

- which architecture yields the strongest planted regression;
- which architecture makes the hidden root easiest to rank;
- which architecture produces the most favorable root-vs-nuisance
  certification result.

The model and exact pretrained revision must be fixed before confirmatory worlds
are generated.

## Training configuration

Experiment 009 should use ordinary supervised sequence classification rather
than reusing the historical causal-LM LoRA training path by force.

Historical `lora_sft` behavior must remain unchanged for Experiments 000–008.
Experiment 009 should introduce a new training path rather than changing the
meaning of the existing configuration schema.

Development may tune, using only development data:

- learning rate;
- number of epochs;
- batch size;
- weight decay;
- warmup schedule;
- maximum sequence length;
- classifier-head initialization policy;
- early-stopping policy, if any.

If early stopping is used, its monitored metric and stopping rule must be fixed
before confirmatory execution and may not use the official test split.

## Training trajectory

The unit of repeated stochastic training is a **trajectory**.

A trajectory ID must determine all controllable stochastic quantities required
to reproduce one paired family of sibling runs.

At minimum this includes:

- classifier-head initialization;
- Python RNG state;
- framework RNG state;
- per-epoch training-slot order;
- dropout/randomization seed family;
- any data-loader generator state.

For each trajectory `t`, the baseline, candidate, and every restoration sibling
must start from the same pretrained checkpoint and the same initial classifier
state.

The initial state should be materialized or hashed so that sibling equality can
be audited.

All siblings within one trajectory must use:

- the same optimizer and scheduler settings;
- the same number of epochs;
- the same batch size;
- the same maximum sequence length;
- the same training-slot schedule;
- the same device/backend regime;
- the same software environment.

Dataset versions may change the content occupying a slot, but they must not
silently change the slot schedule.

To strengthen pairing, tokenization should use a fixed maximum tensor shape for
all siblings rather than allowing content-dependent dynamic shapes to alter the
random-number consumption pattern unnecessarily.

A confirmatory trajectory may never be replaced merely because its result is
unfavorable.

If a trajectory fails for a technical reason, any retry must use the same
trajectory ID and follow a predeclared retry policy.

## Stable training slots

Prepared training examples should be assigned stable slot identifiers that are
independent of dataset version.

Conceptually:

```text
slot_000001
slot_000002
...
```

A baseline or candidate release changes the model-facing record occupying a
slot; it does not change the identity of the slot itself.

The per-epoch slot order must be generated from the trajectory definition rather
than from the incidental order of serialized examples.

This allows baseline, candidate, and restoration siblings to receive matched
batch structure wherever possible while differing only in their intended
versioned records.

## Versioned release structure

Each benchmark world must represent a baseline training release `V0` and a
candidate training release `V1`.

The intended structural form is:

```text
V0 = immutable_core + B1 + B2 + B3 + B4 + B5
V1 = immutable_core + C1 + C2 + C3 + C4 + C5
```

`Bj` is the baseline version of candidate change `j`.

`Cj` is the candidate-release version of the same training artifact.

There are exactly five debugger-visible candidate changes.

Exactly one candidate is the benchmark-private root cause.

A restoration of candidate `j` is:

```text
Rj = V1 with Cj replaced by Bj
```

Every candidate restoration must therefore differ from the candidate release in
exactly one declared versioned artifact.

All five restorations must be trained for every confirmatory trajectory.

The hidden root identity must not be used to decide which restoration runs to
train.

## Candidate-change matching

The benchmark must prevent trivial structural identification of the root.

Before confirmatory freeze, candidate construction must establish prospective
matching constraints for quantities such as:

- artifact kind;
- changed-record count;
- training weight;
- slot count;
- label-count changes;
- text-change count;
- target-surface overlap;
- serialized metadata fields visible to the debugger.

The exact matching rules are development variables and must be finalized after
pilot construction audits.

The root must not be identifiable solely because its artifact is larger, has a
different schema, contains more changed rows, or is the only candidate whose
visible metadata contains labels.

## Root-cause mechanism

The preferred realistic root mechanism is a **taxonomy/label-mapping fault**
between two fine-grained Banking77 intents.

For a target intent pair `A` and `B`, the planted change should introduce a
controlled semantic-label mismatch resembling a versioned taxonomy migration or
mapping defect.

A symmetric mapping fault is preferred when feasible because it can preserve
aggregate label mass:

```text
A content -> B label
B content -> A label
```

The corruption dose is a development variable.

The final dose must be selected using pilot worlds and development evaluation
only, then frozen before confirmatory worlds are scored.

The root must create a measurable held-out regression on the intended target
behavior without requiring direct corruption of protected intents.

## Nuisance changes

The four non-root changes must be legitimate model-facing training updates.

They must remain label-correct according to the Banking77 taxonomy.

They should be capable of perturbing the learned decision boundary or training
trajectory rather than being obvious no-ops.

Possible nuisance forms include matched data refreshes among semantically
adjacent intents, provided that:

- their labels remain correct;
- their construction is prospectively declared;
- their structural footprint is matched to the root where possible;
- they do not secretly implement the same semantic fault as the root.

The pilot phase may compare nuisance constructions.

The confirmatory protocol must select one fixed nuisance-construction rule and
apply it mechanically to all confirmatory worlds.

## Target-intent selection

Confirmatory target intents must not be hand-picked after inspecting the
official test set or after observing confirmatory candidate outcomes.

During development, a deterministic target-pair selection rule must be created
using `development_train` and `development_eval` only.

The rule may use prospectively defined criteria such as:

- sufficient clean per-intent performance;
- sufficient training examples;
- semantic proximity or observed development confusability;
- non-overlap between worlds;
- ability to construct five matched candidate changes.

The exact scoring and tie-breaking rule remains a development variable.

Before confirmatory execution, the algorithm must be frozen and then applied
without manual substitution of an inconvenient world.

If a mechanically selected confirmatory world later fails a frozen prerequisite,
that failure is part of the Experiment 009 result.

## Pilot worlds versus confirmatory worlds

Pilot worlds may be used freely for development within the constraints of this
document.

Pilot results are not confirmatory evidence.

Where practical, pilot target intents should be disjoint from the intents later
used by confirmatory worlds.

The final confirmatory protocol must state:

- the number of confirmatory worlds;
- the deterministic world-selection procedure;
- whether target intents may repeat across worlds;
- the exact root-construction rule;
- the exact nuisance-construction rule.

The current working preference is three non-overlapping confirmatory worlds, but
this number is **not frozen** by the development protocol.

## Behavioral metrics

The intended primary target metric is macro recall over the two target intents.

This avoids allowing one intent in the pair to dominate the target score because
of class-count imbalance.

The intended protected aggregate metric is macro recall over all non-target
intents.

The pilot should also investigate a predeclared worst-protected-intent drift
statistic so that a stable aggregate score cannot hide severe degradation of one
protected intent.

Exact metric definitions, treatment of abstentions, tie handling, and all
numerical margins must be frozen in the confirmatory protocol.

## Paired counterfactual effects

For trajectory `t`, define:

```text
B_t      clean baseline
C_t      regressed candidate
R_j,t    restoration of candidate j
```

For a target metric `M`, candidate regression is:

```text
G_t = M(B_t) - M(C_t)
```

Candidate-restoration effect is:

```text
Delta_j,t = M(R_j,t) - M(C_t)
```

If `r` is the hidden root, the primary causal-specificity contrasts are:

```text
D_j,t = Delta_r,t - Delta_j,t
```

for every non-root candidate `j`.

Experiment 009 therefore does not require every nuisance intervention to produce
exactly zero movement.

The stronger question is whether the root's localized recovery is reliably and
materially larger than the effects produced by non-causal changes under matched
training trajectories.

## Candidate regression gate

The confirmatory protocol must define a prospective candidate-regression gate
whose form includes:

1. clean learnability;
2. material target regression;
3. protected-behavior locality.

The pilot must estimate appropriate practical margins and stochastic variation.

No confirmatory world may be replaced or retuned because it fails this gate.

## Root-recovery gate

The confirmatory protocol must require the planted-root restoration to produce a
material positive target effect relative to the candidate condition.

The criterion must use the repeated paired effect distribution rather than a
single restoration run.

The exact effect-size floor and uncertainty rule are development variables.

## Causal-specificity gate

The primary new gate in Experiment 009 is root-vs-nuisance separation.

For each nuisance candidate, the confirmatory analysis must compare the paired
root effect with the paired nuisance effect across the same trajectory IDs.

The final rule must include both:

- a non-zero practical separation margin; and
- an uncertainty procedure that accounts for the four nuisance comparisons.

Candidate statistical procedures include paired bootstrap simultaneous
confidence intervals, adjusted paired tests, or another justified repeated-run
procedure selected before confirmatory execution.

The procedure must be selected because it matches the estimand and pilot
variance structure, not because it produces the most favorable confirmatory
result.

## Protected-behavior equivalence

Failure to detect a protected-behavior difference is not sufficient evidence of
locality.

The confirmatory protocol must specify an equivalence margin for protected
behavior.

Root restoration should be tested against this equivalence region using a
prospectively selected procedure.

The pilot should determine whether the aggregate protected metric alone is
adequate or whether a worst-intent statistic must also be part of the gate.

## Selecting the number of trajectories

The confirmatory trajectory count `K` must not be selected by convention alone.

Pilot repeated runs must estimate the variability of the paired contrasts that
matter for certification, especially:

```text
D_j,t = Delta_r,t - Delta_j,t
```

The final `K` should be selected from:

- observed pilot variability;
- the minimum practically meaningful root-vs-nuisance separation;
- desired power/precision;
- multiplicity across nuisance comparisons;
- feasible compute budget.

The confirmatory protocol must record the calculation or simulation used to
choose `K`.

`K` may not be increased, decreased, or selectively extended after confirmatory
outcomes are observed.

## Variability controls

### Repeated candidate trajectories

Repeated `C_t` runs are mandatory and directly measure the distribution of the
regressed training condition.

### Repeated baseline trajectories

Repeated `B_t` runs are mandatory so candidate regression itself is evaluated as
a paired stochastic effect rather than against one privileged clean model.

### Deterministic replay check

During development, rerunning the exact same trajectory and dataset version
should be used as an implementation check.

A replay difference is a harness/reproducibility issue and must not be confused
with scientific between-trajectory variability.

### Order-only control

The pilot should include a prospectively defined order-only control in which
semantic training content is fixed while the training-slot order changes.

This directly investigates one plausible mechanism suggested by Experiment 008:
training-order changes may alter apparent counterfactual recovery.

The order-only control is a secondary mechanistic analysis, not a rescue path
for failed primary certification.

## Diagnostic methods

Experiment 009 does not require a novel attribution algorithm to be useful.

The development comparison set should begin with a small number of meaningfully
different diagnostics, such as:

- seeded random ranking;
- lexical or TF-IDF similarity over changed records;
- frozen embedding similarity;
- one gradient-based baseline such as Grad-Dot or a TracIn-style method;
- one task/provenance-aware semantic diagnostic.

TRAK or other heavier methods may be added only if they materially improve the
scientific comparison and remain feasible on the available compute.

Diagnostic code must not receive hidden root identity or restoration outcomes.

The final diagnostic set must be frozen before confirmatory test evaluation.

## Truth isolation

The benchmark-private manifest may contain the hidden root identity.

The diagnostic manifest must remain structurally ground-truth-free, following
the separation already used by earlier experiments.

Human blindness is procedural rather than cryptographic unless stronger
isolation is implemented later.

At minimum:

- diagnostic commands must never read the private benchmark manifest;
- diagnostic artifacts must not contain the hidden root ID;
- restoration execution must not depend on root truth;
- all five restorations must be trained for every confirmatory trajectory.

## Confirmatory execution order

The intended anti-leak order is:

1. freeze dataset source and checksums;
2. freeze development-derived model and training settings;
3. freeze target-pair/world-selection algorithm;
4. freeze root and nuisance construction;
5. freeze trajectory IDs and `K`;
6. freeze metrics, margins, statistical procedure, diagnostics, and stopping rule;
7. generate confirmatory worlds and private/redacted lineage artifacts;
8. train all required baseline, candidate, and five-way restoration siblings for
   every trajectory without using root truth;
9. freeze model/run artifacts and hashes;
10. evaluate baseline and candidate behavior on the previously embargoed official
    Banking77 test split;
11. generate and freeze blinded diagnostic rankings from permitted failure
    evidence and redacted lineage;
12. evaluate every restoration on the official test split without revealing the
    root;
13. freeze all per-candidate behavioral result artifacts;
14. reveal hidden root identities;
15. score diagnostic rankings;
16. compute paired root, nuisance, and protected-behavior effects;
17. apply the frozen certification rule exactly once;
18. record success or failure without retuning.

Restoration test results must not be available to the diagnostic method before
its ranking artifacts are frozen.

## Confirmatory freeze requirements

No confirmatory run may begin until a separate frozen protocol records at least:

- dataset source/revision/checksums;
- development-partition hash;
- model name and exact revision;
- tokenizer revision;
- training hyperparameters;
- device/backend regime;
- software versions;
- fixed-length tokenization settings;
- trajectory generator and exact trajectory IDs;
- `K` and its justification;
- number of worlds;
- world-selection rule;
- root mechanism and dose;
- nuisance-construction rule;
- candidate matching constraints;
- target metric;
- protected metrics;
- clean-baseline gate;
- candidate-regression gate;
- root-recovery margin;
- root-vs-nuisance separation margin;
- protected-equivalence margins;
- multiplicity/uncertainty procedure;
- diagnostic methods;
- truth-isolation boundary;
- retry policy;
- stopping rule.

The frozen protocol must be committed before official-test confirmatory results
are observed.

## Prohibited post-freeze changes

After confirmatory freeze, do not change any of the following because of an
observed result:

- target intents;
- world identity;
- root candidate;
- corruption dose;
- nuisance allocation;
- model architecture;
- pretrained revision;
- learning rate;
- epochs;
- batch size;
- trajectory IDs;
- `K`;
- metric definition;
- effect-size margin;
- equivalence margin;
- statistical procedure;
- diagnostic method;
- candidate matching rule.

Do not replace a failed world.

Do not add trajectories because the result is almost significant.

Do not reinterpret a nuisance as harmless because its observed recovery is
inconvenient.

Do not use a secondary control to rescue a failed primary result.

A defect in code or data handling may be corrected only if the defect is
independent of the scientific outcome, is documented, and requires rerunning all
affected conditions under the same frozen scientific protocol.

## Development stopping rule

Pilot development may iterate until all of the following are simultaneously
credible on development-only worlds:

- clean intent classification is reliable;
- the planted mapping fault can generate a material target regression;
- protected behavior can remain acceptably local;
- nuisance construction is structurally matched and non-trivial;
- repeated trajectories expose a measurable variance distribution;
- root and nuisance interventions can be compared under paired training;
- compute cost permits the required confirmatory `K` and world count.

If Banking77 cannot satisfy these requirements without increasingly artificial
construction, the correct action is to change substrate or mechanism during the
development phase rather than force a confirmatory experiment.

Once confirmatory execution begins, the development stopping freedom ends.

## Relationship to reusable library code

Experiment 009 may introduce reusable internal abstractions where the experiment
actually requires them, especially for:

- stable training slots;
- trajectory IDs;
- paired run manifests;
- versioned candidate changes;
- repeated evaluation summaries;
- distributional certification results.

Do not refactor historical Experiment 000–008 code merely for aesthetic
consistency.

Do not publish a general-purpose library API until the repeated-regression
workflow stabilizes across at least one realistic substrate.

## Immediate implementation order

Development implementation should proceed in this order:

1. pin and audit Banking77 without loading the official test split;
2. implement deterministic development partitioning;
3. add a sequence-classification training path without changing historical LoRA
   behavior;
4. add stable slot IDs and paired trajectory generation;
5. validate deterministic replay on one clean development condition;
6. establish a reliable clean Banking77 baseline;
7. develop the target-pair selection procedure;
8. pilot the taxonomy-mapping root mechanism;
9. pilot matched nuisance updates;
10. run repeated paired pilot trajectories;
11. estimate variance and compute requirements;
12. choose the confirmatory statistical design and `K`;
13. freeze a separate confirmatory protocol;
14. only then execute result-bearing confirmatory worlds.

## Related methodological context

The final paper-level related-work review must include, at minimum, work on:

- classical influence functions;
- TracIn;
- TRAK and scalable data attribution;
- final-model-only/further-training attribution;
- Distributional Training Data Attribution;
- Goodfire predictive data debugging;
- model-diff methods;
- provenance-aware debugging methods;
- stochastic fine-tuning variability;
- counterfactual and pipeline-level ML root-cause analysis.

The broad idea of tracing model behavior to training data is not novel.
Experiment 009 must be evaluated against the strongest current overlap before a
novelty claim is made.
