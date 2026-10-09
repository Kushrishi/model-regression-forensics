# Contribution review after M4

**Date:** 2026-10-03 UTC
**Conclusion:** the localization study is complete; a further certification study needs independent scientific assessment.
**Review status:** author-led assessment, not an independent approval of the continuation gate.

## What M4 establishes

The accepted source-pinned study contains two constructed Banking77 worlds and three paired training trajectories per world. Both target-label overlap and lexical Jaccard ranked the planted root first in both worlds. Final-checkpoint Grad-Dot and seven-checkpoint TracIn ranked it first in one world and last in the other. Complete blind outputs were persisted before separate truth scoring and independently replayed byte-for-byte; exact identities remain in `M4_RESULT.json`.

The engineering result is a reproducible release-debugging experiment with explicit provenance, competitive simple baselines, paired training controls, and separated blind analysis. The scientific result is narrower: the evaluated model-based diagnostics added no top-1 benefit in this design. Three trajectories within a world are not three independent benchmark worlds. These data neither establish general method inferiority nor causal certification.

## Related methods

| Primary source | Existing contribution relevant to MRF | Implication |
| --- | --- | --- |
| Pruthi et al., [Estimating Training Data Influence by Tracing Gradient Descent](https://papers.nips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html), NeurIPS 2020 | Estimates example influence using gradients and training checkpoints. | Checkpoint attribution is a comparator, not an MRF invention. |
| Mlodozeniec et al., [Distributional Training Data Attribution](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html), NeurIPS 2025 | Studies dataset effects on distributions of outputs across stochastic trainings. | Accounting for retraining randomness alone is not a distinct contribution. |
| Li et al., [Which Influence Are We Estimating?](https://arxiv.org/abs/2609.31214), September 2026 preprint | Separates counterfactual specification mismatch from estimation error, explicitly identifying behavior, intervention, and training process. | Explicit estimand specification alone is not a distinct contribution. This source is a preprint. |

This focused review does not establish exhaustive literature coverage or absence of every possible contribution. It identifies direct overlap with the proposed justification for M5. Ranking disagreement cannot by itself distinguish specification mismatch, approximation error, and historical responsibility.

## What certification would need to add

Repair and explanation are different questions. A rollback can improve the target metric without uniquely identifying the historical source of a regression. A useful certification contribution would have to address plausible alternatives that produce similar repairs, interactions between changes, incomplete candidate sets, or stochastic effects large enough to prevent unique attribution.

These are conceptual failure classes, not new benchmark definitions or authorization to redesign the accepted worlds. The current disjoint target/candidate label construction supplies a direct semantic shortcut. M4 does not demonstrate that those hard alternatives are present or that any certify/abstain procedure handles them.

The appropriate comparator for a proposed certification rule is a simple diagnostic followed by the same paired restoration evidence. A confidence label on that workflow is not automatically a new method. A continuation justification must specify the practical decision improved, the scope of any uniqueness claim, the error controlled, and why the additional compute can establish that benefit. Finite-candidate uniqueness must not be presented as exclusion of every possible cause.

## Completed report

The [technical report](M4_TECHNICAL_REPORT.md) presents the completed study, including the candidate construction, semantic shortcut, baseline definitions and complete rankings. Earlier restoration results are reported separately because their candidate design differed.

The current evidence does not demonstrate a new causal-identification method. A follow-up needs a specific decision that controlled repair evidence improves beyond the closest existing work. The [continuation requirements](M4_CONTINUATION_GATE.md) still require independent assessment before further matched-world certification training. No new experiment is approved by this literature review.

## Publication and continuation decision, 9 October 2026

**Keep the completed study as a bounded technical report and reproducible artifact.
Do not submit it as a general comparison of attribution methods or start additional
training solely to enlarge its result table.** The new retained-score stability
analysis adds a useful observation: the second world's TracIn root ranks are 1,
5 and 5 across trajectories, with a primary average rank of 5. It supplies no new
independent worlds and no explanation for that variation.

Negative results do not require a novel algorithm to be valuable. TMLR's current
[acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html) emphasize
supported claims, clear communication and something an audience can learn; they
explicitly do not require novelty of the studied method. The ICML 2024 position
paper [Embracing Negative Results](https://proceedings.mlr.press/v235/karl24a.html)
argues for preserving such evidence. Neither source implies this particular report
is submission-ready. Here, the known-label semantic shortcut and only two worlds
leave the broader practical lesson under-supported. The legitimate current lesson
is to include engineer-visible semantic baselines and inspect trajectory variation
before interpreting a composite ranking.

A software paper is not an automatic alternative. Current
[JOSS requirements](https://joss.readthedocs.io/en/latest/submitting.html) include
sustained public development, demonstrated research use and substantial maintained
software. The comparison utility's strict record handling is useful, but no broad
adoption or advantage over established evaluation/debugging tools is demonstrated.
This review does not claim that a venue would accept or reject the work.

### One concrete follow-up decision

The next eligible empirical question is: **does explicitly checking alternative
repairs change a real repair-acceptance or diagnosis decision beyond ordinary diff
inspection and targeted regression tests, at the same test budget?**

The named LFQ first-fix/follow-up sequence has already been evaluated against this
question. The incomplete first fix fails ordinary unmasked-branch tests; the
follow-up passes the declared fixed-tensor checks. It demonstrates useful repair
testing but supplies no extra MRF diagnosis benefit. Do not retrain a model to
manufacture ambiguity in this already-resolved decision.

Reopening requires a different documented workflow containing a known-good and
regressed task result, complete visible changes, pinned/licensed runnable artifacts,
and plausible interventions that leave a decision unresolved after the ordinary
checks. Before execution, specify the decision, independent incident unit, baseline
test budget and what observation would falsify added value. Do not hide visible
evidence or introduce distractors simply to defeat cheap baselines. Until a concrete
workflow meets those conditions, preserve the public report, accept grounded case
submissions, and allocate substantial experiment effort to TrueMargin. This closes
the present assessment; it does not assert that ML regression diagnosis is solved.
