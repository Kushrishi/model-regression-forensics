# Research roadmap

**Updated:** 2026-10-03 UTC
**Completed study:** Experiment 009 matched localization  
**Current disposition:** preserve a reproducible technical case study; further certification research held

The current deliverable is a readable completed case study with linked history and artifact replay. M1-M4 are complete; M5-M6 below are contingent historical proposals, not an active execution queue. The [continuation gate](M4_CONTINUATION_GATE.md) governs any future certification work.

This roadmap records project-level milestones. Frozen experiment protocols and run requests remain authoritative for result-defining choices.

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

**State:** complete; M5 held at continuation gate

M4 completed on 2026-10-03 UTC in source-pinned run `37083633414`. All three clean and six composite trainings, six blind scoring records, complete blind aggregation, and separate truth scoring succeeded. Replay from the retained scoring records reproduced the blind aggregate and truth evaluation byte-for-byte.

Primary world-level root ranks:

- B0 deterministic random: **3, 3**;
- B1 target-label overlap: **1, 1**;
- B2 lexical Jaccard: **1, 1**;
- B3 final-checkpoint Grad-Dot: **1, 5**;
- B4 seven-checkpoint TracIn: **1, 5**.

Simple visible-change baselines localized the root in both constructed worlds. Model-based methods added no top-1 benefit in this design. These are descriptive development results from two worlds, not causal certification or a general method comparison. Exact records are in `research/M4_RESULT.json` and `research/M4_RESULT_ARTIFACTS/`.

The formal continuation gate in `research/M4_CONTINUATION_GATE.md` accepts M4 and holds M5 pending an independent scientific justification. No restoration or official-test access is authorized.

## M5 — Confirmatory certification protocol

**State:** not started; held pending the M4 continuation gate

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

**State:** technical report draft and artifact replay available; submission and a tagged release are not established

The [technical report](M4_TECHNICAL_REPORT.md), [experiment history](EXPERIMENT_HISTORY.md), and [replay guide](REPRODUCE_M4.md) present the completed evidence. This bounded reporting work does not depend on starting M5 or establish a publication.

Historical paper-level deliverables:

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
