# Locating training-release changes with simple baselines

**A matched Banking77 case study**
**Status:** technical report draft; not a submitted or peer-reviewed paper
**Evidence:** matched label-swap localization study, October 3, 2026

Historical result files use the identifier `M4`; it names this study, not an
additional method or performance level.

## Abstract

We study a narrow release-debugging problem: given a known model behavior and five visible training-data changes, which change should be investigated first? Two constructed Banking77 worlds contain structurally matched label swaps, with model training repeated over three paired trajectories per world. Target-label overlap and lexical Jaccard rank the planted responsible change first in both worlds. Final-checkpoint Grad-Dot and seven-checkpoint TracIn rank it first in one world and last in the other. Blind ranking records were finalized before separate truth scoring, and replay reproduces the retained outputs byte-for-byte. The study demonstrates a reproducible debugging workflow and the strength of simple baselines under the available target information. It does not establish general attribution-method superiority, difficult root localization, or causal certification.

## Question and experimental design

The diagnostic task is to prioritize a release change for investigation, not to certify a unique cause. Each world exposes five opaque candidate IDs and a known target intent pair. Every candidate swaps labels on 66 training slots: 33 in each direction, no text changes, no aggregate label-count change, and two affected labels. Changed slots and intent pairs are disjoint within a world. Benchmark truth is withheld from candidate scoring.

This construction controls visible differences in change mechanism and size. It deliberately leaves semantic information available: because the target pair is known and candidate intent pairs are disjoint, direct label overlap can identify the relevant candidate. Structural matching therefore does not imply a hard attribution problem.

The model is `distilbert/distilbert-base-uncased`, pinned at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`. Training uses seven epochs, batch size 32, learning rate 2e-5, weight decay 0.01, warmup ratio 0.10, maximum sequence length 128, and maximum gradient norm 1.0. Three clean models are shared across the paired trajectories; each of the two composite worlds has three trained models. The study contains nine trainings, not six independent worlds. The official Banking77 test split is not accessed.

The target behavior is the mean pairwise logit margin over all development-evaluation examples belonging to the target pair. Candidate scores sum changed-slot suspiciousness. Model-based scores are averaged across the three trajectories to obtain the primary world-level ranking. Higher scores indicate greater suspicion; ascending opaque candidate ID resolves ties. These rules were frozen before rankings were inspected.

The complete source definitions are in [the matched-study protocol](M4_LOCALIZATION_BASELINE_PROTOCOL.md) and its [amendment](M4_LOCALIZATION_BASELINE_PROTOCOL_AMENDMENT_1.md). This report summarizes the accepted experiment rather than changing those definitions.

## Baselines and results

The recorded target-margin regression is the clean mean margin minus the composite mean margin. It is positive in every trajectory. World 00 has 52 target-slice examples; world 01 has 51. These are logit-margin changes, not accuracy percentages or independent world replications.

| World | Trajectory | Clean mean margin | Composite mean margin | Margin regression |
| --- | ---: | ---: | ---: | ---: |
| 00 | 0 | 4.015 | 0.607 | 3.408 |
| 00 | 1 | 3.330 | 0.451 | 2.879 |
| 00 | 2 | 4.654 | 0.626 | 4.028 |
| 01 | 0 | 6.221 | 0.964 | 5.257 |
| 01 | 1 | 6.654 | 0.909 | 5.745 |
| 01 | 2 | 5.056 | 0.853 | 4.202 |

Values above are rounded to three decimals from the six retained scoring records; the original full-precision values remain in the retained-results directory. This table describes the observed behavior change and introduces no new statistical inference.

The comparison includes a deterministic pseudo-random reference, target-label overlap, changed-text lexical Jaccard, final-checkpoint last-layer Grad-Dot, and last-layer checkpoint TracIn using all seven epoch checkpoints. The lexical baseline uses the maximum token-set Jaccard similarity to target-slice evaluation examples for each changed slot. Grad-Dot and TracIn use the prospectively fixed target/sign/layer conventions; neither is retuned after truth scoring.

| Frozen baseline | Root rank, world 00 | Root rank, world 01 | Worlds with root ranked first |
| --- | ---: | ---: | ---: |
| Deterministic random | 3 | 3 | 0 / 2 |
| Target-label overlap | 1 | 1 | 2 / 2 |
| Lexical Jaccard | 1 | 1 | 2 / 2 |
| Final-checkpoint Grad-Dot | 1 | 5 | 1 / 2 |
| Seven-checkpoint TracIn | 1 | 5 | 1 / 2 |

These are descriptive results from two constructed worlds. Both simple baselines localize the root under the debugger-visible information. The evaluated model-based methods add no top-1 benefit in this design. The result is retained even though it does not support a claim of improved localization.

## Provenance and reproducibility

Source-pinned workflow `37083633414` completed three clean trainings, six composite trainings, six blind scoring records, complete blind aggregation, and separate truth scoring. Execution source is `a5ec718e6081624893fed406eedb1dd5e1405892`; implementation source is `8cc9a3c3175a9e4133bd571b6a8fc140845a764c`. The complete blind aggregate was persisted before truth access. Local replay of the six records reproduced the hosted blind aggregate and truth evaluation byte-for-byte.

[M4_RESULT.json](M4_RESULT.json) records acceptance checks, exact archive identities, retained-file SHA-256 digests, and world-level outcomes. [M4_RESULT_ARTIFACTS](M4_RESULT_ARTIFACTS/) retains the complete scoring records and outputs. Protocols, negative evidence, and the [continuation requirements](M4_CONTINUATION_GATE.md) remain public.

## Interpretation and limitations

Success on this task does not establish causal responsibility. Ranking a planted change first prioritizes an investigation; it neither rules out other repair interventions nor demonstrates that a certify/abstain rule has controlled error. No matched-world restoration experiment or confirmatory certification outcome is reported here. Older Stage B development evidence belongs to a different candidate construction and is not pooled with M4.

The known target pair and disjoint candidate intent pairs create a semantic shortcut. There are only two constructed worlds, one model family, one dataset, and a restricted classifier-layer attribution scope. The three paired trajectories quantify repetition within those worlds rather than independent scenario coverage. The evidence cannot support a broad ranking of attribution methods, general root-cause identification, external validity, or statistical superiority across tasks.

The study provides a reproducible example of release debugging in which simple visible-change checks outperform the evaluated model-based diagnostics. Whether controlled repair evidence can add useful causal information remains an open question. Further matched-world certification training is paused pending the independent assessment described in the [continuation review](M4_CONTRIBUTION_REVIEW.md).

## Descriptive trajectory stability check, 9 October 2026

A post-outcome analysis of the accepted score records checks what the primary
average hides. This does not change the frozen rankings, select a new method,
retune signs or access new outcomes. Input hashes are checked against the acceptance
record before analysis. Run `python scripts/analyze_retained_rank_stability.py`;
the retained output is [retained-rank-stability.json](retained-rank-stability.json).

| Diagnostic | World 00: trajectory 0 / 1 / 2 | World 01: trajectory 0 / 1 / 2 |
| --- | --- | --- |
| Deterministic random reference | 3 / 3 / 3 | 3 / 3 / 3 |
| Target-label overlap | 1 / 1 / 1 | 1 / 1 / 1 |
| Lexical Jaccard | 1 / 1 / 1 | 1 / 1 / 1 |
| Final-checkpoint Grad-Dot | 1 / 1 / 1 | 5 / 5 / 5 |
| Seven-checkpoint TracIn | 1 / 1 / 1 | 1 / 5 / 5 |

Entries are planted-change ranks among five candidates. In world 01, averaging
TracIn scores over trajectory pairs (0,1), (0,2) and (1,2) gives root ranks 5, 1
and 5 respectively. Its primary three-trajectory average remains rank 5. The
simple baselines remain rank 1 for every pair in both worlds; their scores do not
depend on model training. These overlapping pair analyses are descriptive
sensitivity checks, not independent replications or significance tests. The two
constructed worlds share the dataset and clean-model trajectories; they are not
independent draws from a population of real incidents.

The saved records therefore show within-world variation for TracIn as well as
between-world variation. They do not identify its mechanism. The records declare
that model checkpoint bytes were not retained; hashes are not executable weights.
Fresh gradient/target/layer ablations cannot be reconstructed from these score
records alone. No such ablation or new training was performed.
