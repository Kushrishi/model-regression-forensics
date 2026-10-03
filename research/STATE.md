# Current research state

**Updated:** 2026-10-03 UTC
**Active program:** Experiment 009 — stochastic counterfactual certification  
**Evidence class:** development only

This file is the canonical short-form statement of the project's current scientific state. Historical experiment documents remain authoritative for their own frozen protocols and outcomes.

## Research question

Given a known-good release, a regressed release, and a finite set of versioned training changes, can a suspected cause be supported by counterfactual restoration evidence that is distinguishable from plausible non-root interventions and ordinary retraining variability?

## Completed Exp009 development evidence

The Banking77 development substrate uses 8,001 training examples and 1,998 development-evaluation examples across 77 intents with a pinned `distilbert-base-uncased` classifier. The official Banking77 test split remains untouched.

### Stage A

Workflow run `36139384603` completed the authorized Stage-A training siblings.

The frozen behavioral gate passed across trajectories 0, 1, and 2:

- mean target regression: **0.129289**;
- minimum per-trajectory target regression: **0.091912**;
- mean protected regression: **0.002711**;
- maximum per-trajectory protected regression: **0.003602**.

The truth-isolated last-layer Grad-Dot baseline ranked the planted root first in all three trajectories. That result is development evidence only because the nuisance-v2 candidate changes were structurally distinguishable from the root.

### Stage B

Workflow run `36154597887` completed all 18 frozen Stage-B trainings.

Across the three paired trajectories:

- mean root target recovery: **+0.124387**;
- root recovery range: **+0.091912 to +0.163603**;
- mean root-minus-strongest-nuisance margin: **+0.119485**;
- minimum root-minus-strongest-nuisance margin: **+0.091912**;
- root restoration exceeded every nuisance restoration in **3 / 3** trajectories.

No confirmatory threshold, p-value, or multiplicity procedure was applied to Stage B.

## M3 structurally matched benchmark

**State: complete.**

M3 replaced the structurally mismatched nuisance-v2 construction with a prospectively frozen matched candidate benchmark.

The base protocol initially requested three worlds with five globally intent-disjoint candidates per world. Clean-only capacity preflight showed that three complete worlds were impossible under the frozen eligibility rules: the eligible graph had 81 edges over 27 vertices and a maximum matching of 13 disjoint pairs, below the 15 required.

The project did not relax the eligibility rules after observing that failure. Amendment 1 prospectively reduced M3 to **two complete worlds** before any matched-benchmark model training and without accessing the official test split.

The final benchmark contains:

- 2 worlds;
- 5 candidates per world;
- 10 candidate changes across 20 unique intents;
- exactly 66 changed stable slots per candidate;
- symmetric 33/33 label swaps;
- zero text changes;
- preserved aggregate label mass;
- identical debugger-facing candidate structure;
- pairwise-disjoint changed-slot sets and intent labels;
- deterministic opaque candidate IDs;
- exact standalone restoration to baseline;
- composite order independence.

The amended hosted structural preflight passed in workflow run `36211417215`. A final post-formatting rerun on the merged implementation also passed. No matched-benchmark model training was performed by M3, and the official Banking77 test split remained embargoed.

Passing M3 establishes **structurally matched candidate construction only**. It does not establish localization success or causal specificity.

## Current decision boundary

M1, M2, and M3 are complete.

The next active milestone is **M4 — competitive localization baselines**.

Before treating localization as scientifically interesting, M4 must compare diagnostics against simple target-compatible references. Because the target behavior is known, a semantic/change-overlap baseline may localize the responsible pair easily. That is a legitimate result and must not trigger post-hoc benchmark redesign.

Required baseline families include:

- seeded random reference;
- simple semantic/lexical or changed-record overlap where applicable;
- target-faithful last-layer Grad-Dot;
- modern influence/data-attribution approaches only where their objective can be implemented faithfully for the frozen target.

TRAK remains feasibility evidence unless its scoring objective becomes genuinely commensurate with the frozen target without redefining that target.

The M4 protocol and Amendment 1 freeze the two-world, three-trajectory localization matrix and B0 through B4 before matched-world training. Execution preparation now reconstructs all frozen M3 identities, trains exactly three clean models and six composite models, captures all seven composite checkpoints, and finalizes blind rankings before a separate truth-scoring job. The initial zero-rate warmup update is permitted; epoch checkpoints record the positive rate of the final producing update. B0 retains unsigned 64-bit integer precision throughout ranking.

M4 completed on 2026-10-03 UTC in source-pinned run `37083633414`. All three clean and six composite trainings, six blind scoring records, complete blind aggregation, and separate truth scoring succeeded. Replay from the retained scoring records reproduced the blind aggregate and truth evaluation byte-for-byte.

Primary world-level root ranks:

- B0 deterministic random: **3, 3**;
- B1 target-label overlap: **1, 1**;
- B2 lexical Jaccard: **1, 1**;
- B3 final-checkpoint Grad-Dot: **1, 5**;
- B4 seven-checkpoint TracIn: **1, 5**.

Simple visible-change baselines localized the root in both constructed worlds. Model-based methods added no top-1 benefit in this design. These are descriptive development results from two worlds, not causal certification or a general method comparison. Exact records are in `research/M4_RESULT.json` and `research/M4_RESULT_ARTIFACTS/`.

The formal continuation gate in `research/M4_CONTINUATION_GATE.md` accepts M4 and holds M5 pending an independent scientific justification. No restoration or official-test access is authorized.

`research/M4_TECHNICAL_REPORT.md` presents the accepted design, all six recorded target-margin changes, world-level rankings, and provenance as a bounded technical report draft. `research/M4_CONTRIBUTION_REVIEW.md` assesses adjacent literature and retains the M5 hold; it is an author-led assessment, not independent continuation approval.

## Not established

The project does not currently establish:

- localization success beyond the two evaluated M3 worlds;
- causal specificity for Exp009;
- confirmatory causal certification;
- general training-data root-cause identification;
- superiority to modern attribution or influence methods;
- cross-model or cross-dataset generalization; or
- a completed paper or accepted publication.

## Source-of-truth order

When documents disagree, use this priority:

1. frozen experiment protocol/result files for the experiment they govern;
2. `research/ATTRIBUTION_TARGET.md` for localization-target semantics;
3. this file for current project-level state;
4. `research/CLAIMS.md` for externally safe claim boundaries;
5. `research/DECISION_LOG.md` for historical decisions;
6. README and portfolio copy.

Website and LinkedIn text must never outrun this repository state.
