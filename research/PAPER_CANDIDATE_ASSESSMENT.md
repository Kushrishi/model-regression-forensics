# Paper candidate assessment

**Status:** existing-evidence case study outline; not submitted or peer reviewed
**Decision:** retain the technical report and candidate outline; further certification remains paused
**Scope:** no new experiments, thresholds, scoring methods, or protocol changes

## Candidate contribution

**Working title:** Regression localization and repair specificity in controlled training change studies

The candidate is an empirical case study of how three debugging questions can give different answers: which visible training change a diagnostic prioritizes, whether restoring that change repairs target behavior, and whether competing restorations also repair it. A second theme is the need to audit simple semantic baselines even when candidate changes are structurally matched.

This is a proposed presentation of existing evidence, not a new attribution algorithm, theorem, or general claim that model regressions cannot be attributed. Its publication value depends on whether the specific experimental observations add enough to established counterfactual attribution research.

## Candidate abstract

Controlled training-change studies can evaluate diagnosis and repair separately. In a two-world synthetic task, a semantic diagnostic uniquely ranks the planted label-corruption change first, and restoring that change fully repairs the target without damaging protected behavior. However, nuisance restorations also improve target accuracy, causing both worlds to fail their prospectively declared unique-certification criteria. In a separate matched Banking77 study, label-overlap and lexical baselines rank the planted change first in both worlds, while evaluated last-layer Grad-Dot and checkpoint TracIn rank it first in one and last in the other. These development studies illustrate pitfalls in interpreting recovery alone and in treating structural matching as evidence of difficult localization. The cohorts are not pooled. Two worlds per study, limited intervention replication, semantic shortcuts, and restricted attribution scope prevent general claims about causal identification or method superiority. Retained matched-study records support byte-identical replay.

## Evidence to include

### Synthetic restoration study

[Experiment 008](../experiments/008_selective_causal_rca/RESULTS.md) is the central repair-specificity example. Both candidate worlds passed the localized-regression prerequisite; the semantic diagnostic uniquely ranked the planted root first in both.

The table below reproduces every primary restoration. Recovery is restoration minus candidate target accuracy. Protected drift is the original maximum protected-slice measure.

| World | Candidate suffix | Role | Target recovery | Maximum protected drift |
| --- | --- | --- | ---: | ---: |
| 00 | 01 | nuisance | 0.1875 | 0.0000 |
| 00 | 02 | nuisance | 0.0000 | 0.0000 |
| 00 | 03 | nuisance | 0.1875 | 0.0000 |
| 00 | 04 | nuisance | 0.1875 | 0.0000 |
| 00 | 05 | planted root | 1.0000 | 0.0000 |
| 01 | 01 | nuisance | 0.0000 | 0.0000 |
| 01 | 02 | planted root | 0.8125 | 0.0000 |
| 01 | 03 | nuisance | 0.0000 | 0.0000 |
| 01 | 04 | nuisance | 0.8125 | 0.3125 |
| 01 | 05 | nuisance | 0.0000 | 0.0000 |

World 00 candidate target accuracy was 0.0000; world 01 was 0.1875. The frozen rules required root recovery at least 0.15, nuisance recovery at most 0.05, protected drift at most 0.05, and exactly one restoration above the general recovery threshold. Both worlds failed certification.

The world 01 nuisance matches root recovery only on the target metric; it violates the protected-behavior requirement. The root remains the better repair under those two measures. World 00 also has a clearly stronger root repair, despite nuisance recovery exceeding the frozen ceiling. Failure of these particular certification rules does not prove that useful repair selection or attribution is impossible.

There is one primary restoration model per candidate per world. The observations cannot distinguish systematic intervention effects from ordinary retraining variability. The alternative-order control was reserved for a passing primary certification and was not run. Optimization-path explanations remain hypotheses.

### Banking77 matched localization study

The [accepted M4 technical report](M4_TECHNICAL_REPORT.md) supplies the second example.

| Method | World 00 root rank | World 01 root rank |
| --- | ---: | ---: |
| Deterministic random | 3 | 3 |
| Target-label overlap | 1 | 1 |
| Lexical Jaccard | 1 | 1 |
| Final-checkpoint Grad-Dot | 1 | 5 |
| Seven-checkpoint TracIn | 1 | 5 |

All five changes in each world use equal-sized symmetric label swaps, unchanged text, and preserved aggregate label counts. Yet known target labels and disjoint candidate label pairs make direct overlap informative. Structural matching removes one family of shortcuts; it does not remove semantic shortcuts.

M4 comprises two constructed worlds and three paired trajectories within each world. It includes localization only, with no matched-world restoration or certify/abstain result. The model-based methods operate on the classifier layer. The result cannot establish general inferiority of full-network attribution methods.

### Earlier development history

The [experiment history](EXPERIMENT_HISTORY.md) explains capability failures, shortcut exposure, unsuccessful regression constructions, and the transition to Banking77. Use this as context, not a sequence of independent replications.

The earlier Banking77 [Stage B](../experiments/009_stochastic_counterfactual_certification/STAGE_B_RESULT.md) showed root restoration exceeding nuisance restoration in three trajectories. Its candidate construction had a structural mismatch. Present it separately if needed to explain why matching was attempted; never pool it with M4 or treat it as matched causal certification.

## Manuscript outline

1. **Problem and scope.** Define the debugger-visible target, planted change, candidate restoration, protected behavior, and difference between ranking and intervention evidence.
2. **Related work.** Separate attribution estimation, training-subset behavior prediction, training randomness, and counterfactual specifications.
3. **Synthetic case study.** Explain frozen prerequisites and report all ten restorations. Discuss target-only equivalence versus protected-behavior differences.
4. **Matched localization case study.** Explain the two-world design and complete baseline comparison. Identify the semantic shortcut explicitly.
5. **Practical implications.** Require simple baselines, report competing repairs, specify the behavior and intervention being evaluated, and preserve failed gates.
6. **Limitations and reproducibility.** Separate cohorts, identify missing mechanism tests, distinguish replay of retained results from model retraining, and state the narrow external validity.

The complete restoration table and ranking table are sufficient for a first outline. A later manuscript could use plots generated directly from retained evidence, but cosmetic additions would not strengthen novelty.

## Closest literature and remaining distinction

| Prior work | Established direction | Candidate distinction and limitation |
| --- | --- | --- |
| [TracIn](https://papers.nips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html), NeurIPS 2020 | Attribution through training-checkpoint gradients | MRF evaluates a restricted baseline in a release-change task; this is not a new method. |
| [Datamodels](https://proceedings.mlr.press/v162/ilyas22a.html), ICML 2022 | Predicting behavior under training-subset changes | Visible versioned changes and repair criteria are a specific setting, not a new counterfactual principle. |
| [TRAK](https://arxiv.org/abs/2303.14186), ICML 2023 | Scalable training data attribution | The current study does not establish an improvement over scalable attribution systems. |
| [Distributional Training Data Attribution](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html), NeurIPS 2025 | Attribution across stochastic training outcomes | Merely measuring retraining variance is insufficient as a new contribution. |
| [Which Influence Are We Estimating](https://arxiv.org/abs/2609.31214), September 2026 preprint | Attribution depends on behavior, intervention, and counterfactual training specification | A concrete repair-specificity case study may be complementary; a broad specification-mismatch claim overlaps directly. |
| [Outputs of generative diffusion models are often unattributable](https://www.nature.com/articles/s41467-026-75667-5), Nature Communications 2026 | Limits of leave-one-out attribution for generated diffusion outputs | Different model, intervention, and task. It neither proves nor is refuted by MRF's restoration outcomes. |

This comparison identifies overlap rather than proving exhaustive novelty absence. The narrow candidate contribution is a transparent experimental example, with competing repair outcomes and shortcut auditing. Whether that is sufficiently distinct for a particular workshop requires an appropriate reviewer and venue scope.

## Decision

The existing evidence supports a useful technical report and an honest project explanation. It does not currently justify a general method paper, causal impossibility claim, or a new certification training campaign.

A workshop case study remains possible, but no venue or independent review has established suitability. A preprint could make the report accessible without implying peer review.

Before further training, a scientific reviewer needs to assess whether the proposed repair-specificity study would add a useful result beyond the closest work. The [continuation requirements](M4_CONTINUATION_GATE.md) remain in effect. The completed evidence and this outline are retained regardless of that decision.
