# Research roadmap

**Updated:** 2026-09-25  
**Active program:** Experiment 009  
**Target:** publication-grade evidence for post-hoc causal certification of versioned model regressions

This roadmap defines project-level milestones. Frozen experiment protocols and
run requests remain authoritative for result-defining choices.

## M1 — Development restoration pilot

**State:** in progress

Goal: finish the already-authorized three-trajectory Stage-B pilot without
changing its design.

Exit requirements:

- all 18 planned Stage-B model trainings complete or a documented technical
  failure prevents completion;
- every root and nuisance restoration is retained;
- same-session composite subtraction is preserved;
- no Stage-B significance threshold is introduced post hoc;
- official Banking77 test remains untouched;
- a permanent Stage-B evidence/result record is audited and committed.

This milestone estimates restoration-effect structure and stochastic
variability. It is not confirmatory causal certification.

## M2 — Development decision gate

**State:** pending M1

Goal: decide whether the observed root-vs-nuisance effect structure justifies a
paper-grade benchmark.

Required review:

- root restoration effect by trajectory;
- all four nuisance effects by trajectory;
- root-minus-nuisance paired margins;
- protected-behavior drift;
- variability and worst competing nuisance;
- failure/invalid-run audit.

Possible outcomes are all valid:

1. continue to a stronger benchmark;
2. narrow the hypothesis;
3. stop the certification claim if the effect is not distinguishable enough.

No development threshold may be invented merely to force continuation.

## M3 — Structurally matched benchmark

**State:** not started

Goal: remove the current nuisance-v2 structural leakage.

Before result-bearing training, freeze:

- benchmark worlds and candidate-generation rule;
- root and nuisance changes with matched observable structure;
- opaque candidate representation;
- target behavior and protected behavior;
- trajectory count and stochastic pairing;
- benchmark exclusion/invalidity rules.

The current nuisance-v2 pilot may motivate this design but may not be presented
as a strong structurally matched localization benchmark.

## M4 — Competitive localization baselines

**State:** partially developed

Goal: compare target-compatible diagnostics without changing the scientific
target to suit a method.

Required baseline families:

- seeded random reference;
- simple lexical/change-overlap reference where applicable;
- target-faithful last-layer Grad-Dot;
- modern influence/data-attribution baselines only where their objective can be
  implemented faithfully for the frozen target.

TRAK remains feasibility evidence unless its scoring objective is made genuinely
commensurate with the benchmark target without redefining that target.

## M5 — Confirmatory certification protocol

**State:** not started

Before any confirmatory outcome is observed, freeze:

- trajectory count justified by development variability;
- primary restorative-effect statistic;
- causal-specificity/abstention rule;
- multiplicity treatment across nuisance contrasts;
- protected-behavior rule;
- technical-rerun policy;
- official-test access rule;
- all primary baselines.

Development evidence from M1-M4 must not be relabeled confirmatory.

## M6 — Cross-substrate stress test

**State:** contingent on M5

Goal: test whether the method survives at least one materially different model,
dataset, or regression substrate.

This milestone is required before any broad generalization claim.

## M7 — Paper and reproducibility release

**State:** contingent on evidence

Deliverables:

- final related-work audit and contribution boundary;
- frozen claims ledger;
- figures and tables generated from machine-readable evidence;
- negative and abstention cases;
- reproduction instructions and exact source/configuration identities;
- manuscript/preprint;
- tagged public release.

## Stop rules

The project should be narrowed or stopped rather than optimized around an
unfavorable result if:

- a structurally matched benchmark cannot be constructed without obvious
  leakage;
- root restoration is not reliably distinguishable from plausible nuisances;
- a competitive baseline makes the proposed certification framing redundant; or
- the literature closes the claimed contribution gap.

## Public narrative rule

Repository evidence is the source of truth. Website, GitHub profile, CV, and
LinkedIn wording may summarize only completed milestones and must not outrun
`research/CLAIMS.md`.
