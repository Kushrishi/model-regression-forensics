# Research roadmap

**Updated:** 2026-10-01  
**Active program:** Experiment 009  
**Target:** publication-grade evidence for post-hoc causal certification of versioned model regressions

This roadmap defines project-level milestones. Frozen experiment protocols and run requests remain authoritative for result-defining choices.

## M1 — Development restoration pilot

**State:** complete

The frozen three-trajectory Stage-B pilot completed all 18 planned trainings. Every root and nuisance restoration was retained, same-session composite subtraction was preserved, the official Banking77 test remained untouched, and no significance threshold was invented after seeing outcomes.

This milestone estimates restoration-effect structure and stochastic variability. It is not confirmatory causal certification.

## M2 — Development decision gate

**State:** complete — continued to M3

The development result showed consistent root-vs-nuisance restoration separation across all three paired trajectories and justified building a stronger benchmark. That decision did not convert the pilot into confirmatory evidence.

## M3 — Structurally matched benchmark

**State:** complete

Goal: remove the nuisance-v2 structural mismatch before further end-to-end benchmark work.

Completed design:

- all candidates use the same 66-slot symmetric label-swap mechanism;
- candidate text is unchanged and aggregate label mass is preserved;
- candidate-facing schema is structurally identical;
- candidate changed slots and touched intents are disjoint within/across worlds;
- opaque candidate representation and exact restoration audits are implemented;
- target behavior, protected behavior, exclusion rules, and clean-only eligibility/ranking are frozen;
- official Banking77 test remains embargoed.

The base protocol requested three five-candidate worlds. Clean-only graph-capacity evidence showed that only 13 globally disjoint eligible pairs were possible versus 15 required. Amendment 1 therefore reduced the benchmark to two complete worlds **before any matched-benchmark model training**, without weakening eligibility criteria.

The amended structural preflight passed for:

- 2 worlds;
- 5 candidates per world;
- 10 candidate pairs;
- 20 unique candidate-touched intents.

M3 establishes a prospectively constructed structurally matched candidate benchmark. It does not establish localization success or causal specificity.

## M4 — Competitive localization baselines

**State:** active

Goal: determine how much localization signal remains once structural mismatch is removed, without redefining the target to suit a method.

Required baseline families:

- seeded random reference;
- simple semantic/lexical or changed-record overlap where applicable;
- target-faithful last-layer Grad-Dot;
- modern influence/data-attribution baselines only where their objective can be implemented faithfully for the frozen target.

The known target behavior makes a simple semantic/change-overlap baseline particularly important. If that baseline solves localization, report the result rather than redesigning the benchmark after observing it.

TRAK remains feasibility evidence unless its scoring objective is made genuinely commensurate with the frozen target without redefining that target.

Before result-bearing matched-world training, M4 must also define the development execution matrix and exactly which localization outputs are compared. M4 evidence remains development evidence.

## M5 — Confirmatory certification protocol

**State:** not started

Before any confirmatory outcome is observed, freeze:

- trajectory count justified by development variability;
- primary restorative-effect statistic;
- causal-specificity/abstention rule;
- multiplicity treatment across nuisance contrasts;
- protected-behavior/equivalence rule;
- technical-rerun policy;
- official-test access rule;
- all primary baselines.

Development evidence from M1-M4 must not be relabeled confirmatory.

## M6 — Cross-substrate stress test

**State:** contingent on M5

Goal: test whether the method survives at least one materially different model, dataset, or regression substrate.

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

## Stop / narrow rules

The project should be narrowed or stopped rather than optimized around an unfavorable result if:

- the structurally matched benchmark yields no meaningful target-localized regression without post-hoc construction changes;
- root restoration is not reliably distinguishable from plausible non-root interventions;
- a simple or competitive baseline makes the proposed localization contribution trivial or redundant;
- a faithful attribution baseline cannot be compared without changing the scientific target; or
- contemporary literature closes the intended contribution gap.

A negative localization or certification result is valid evidence and must not trigger retrospective benchmark redesign.

## Public narrative rule

Repository evidence is the source of truth. Website, GitHub profile, CV, and LinkedIn wording may summarize only completed milestones and must not outrun `research/CLAIMS.md`.
