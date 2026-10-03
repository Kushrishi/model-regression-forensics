# Locating training-release changes with simple baselines

**A matched Banking77 case study**  
**Status:** technical report draft; not a submitted or peer-reviewed paper  
**Evidence:** accepted M4 development study, October 3, 2026

## Abstract

We study a narrow release-debugging problem: given a known model behavior and five visible training-data changes, which change should be investigated first? Two constructed Banking77 worlds contain structurally matched label swaps, with model training repeated over three paired trajectories per world. Target-label overlap and lexical Jaccard rank the planted responsible change first in both worlds. Final-checkpoint Grad-Dot and seven-checkpoint TracIn rank it first in one world and last in the other. Blind ranking artifacts were finalized before separate truth scoring, and replay reproduces the retained outputs byte-for-byte. The study demonstrates a reproducible debugging workflow and the strength of simple baselines under the available target information. It does not establish general attribution-method superiority, difficult root localization, or causal certification.

## Question and experimental design

The diagnostic task is to prioritize a release change for investigation, not to certify a unique cause. Each world exposes five opaque candidate IDs and a known target intent pair. Every candidate swaps labels on 66 training slots: 33 in each direction, no text changes, no aggregate label-count change, and two affected labels. Changed slots and intent pairs are disjoint within a world. Benchmark truth is withheld from candidate scoring.

This construction controls visible differences in change mechanism and size. It deliberately leaves semantic information available: because the target pair is known and candidate intent pairs are disjoint, direct label overlap can identify the relevant candidate. Structural matching therefore does not imply a hard attribution problem.

The model is `distilbert/distilbert-base-uncased`, pinned at revision `12040accade4e8a0f71eabdb258fecc2e7e948be`. Training uses seven epochs, batch size 32, learning rate 2e-5, weight decay 0.01, warmup ratio 0.10, maximum sequence length 128, and maximum gradient norm 1.0. Three clean models are shared across the paired trajectories; each of the two composite worlds has three trained models. The study contains nine trainings, not six independent worlds. The official Banking77 test split is not accessed.

The target behavior is the mean pairwise logit margin over all development-evaluation examples belonging to the target pair. Candidate scores sum changed-slot suspiciousness. Model-based scores are averaged across the three trajectories to obtain the primary world-level ranking. Higher scores indicate greater suspicion; ascending opaque candidate ID resolves ties. These rules were frozen before rankings were inspected.

The complete source definitions are in [the M4 protocol](M4_LOCALIZATION_BASELINE_PROTOCOL.md) and its [amendment](M4_LOCALIZATION_BASELINE_PROTOCOL_AMENDMENT_1.md). This report summarizes the accepted experiment rather than changing those definitions.

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

Values above are rounded to three decimals from the six retained scoring records; the original full-precision values remain in the artifact directory. This table describes the observed behavior change and introduces no new statistical inference.

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

[M4_RESULT.json](M4_RESULT.json) records acceptance checks, exact archive identities, retained-file SHA-256 digests, and world-level outcomes. [M4_RESULT_ARTIFACTS](M4_RESULT_ARTIFACTS/) retains the complete scoring records and outputs. Protocols, negative evidence, and the [continuation gate](M4_CONTINUATION_GATE.md) remain public.

## Interpretation and limitations

Success on this task does not establish causal responsibility. Ranking a planted change first prioritizes an investigation; it neither rules out other repair interventions nor demonstrates that a certify/abstain rule has controlled error. No matched-world restoration experiment or confirmatory certification outcome is reported here. Older Stage B development evidence belongs to a different candidate construction and is not pooled with M4.

The known target pair and disjoint candidate intent pairs create a semantic shortcut. There are only two constructed worlds, one model family, one dataset, and a restricted classifier-layer attribution scope. The three paired trajectories quantify repetition within those worlds rather than independent scenario coverage. The evidence cannot support a broad ranking of attribution methods, general root-cause identification, external validity, or statistical superiority across tasks.

The completed artifact is useful for release-debugging reproducibility and for demonstrating why credible baselines and separated blind analysis matter. Whether a distinct causal-certification contribution is worth further computation remains unresolved. The [contribution review](M4_CONTRIBUTION_REVIEW.md) retains the M5 hold; this report introduces no new training or evaluation authorization.
