# Contribution review after M4

**Date:** 2026-10-03 UTC  
**Disposition:** retain the M5 hold; prepare a bounded technical report from accepted evidence.  
**Review status:** author-led assessment, not an independent approval of the continuation gate.

## What M4 establishes

The accepted source-pinned study contains two constructed Banking77 worlds and three paired training trajectories per world. Both target-label overlap and lexical Jaccard ranked the planted root first in both worlds. Final-checkpoint Grad-Dot and seven-checkpoint TracIn ranked it first in one world and last in the other. Complete blind outputs were persisted before separate truth scoring and independently replayed byte-for-byte; exact identities remain in `M4_RESULT.json`.

The engineering result is a reproducible release-debugging experiment with explicit provenance, competitive simple baselines, paired training controls, and separated blind analysis. The scientific result is narrower: the evaluated model-based diagnostics added no top-1 benefit in this design. Three trajectories within a world are not three independent benchmark worlds. These data neither establish general method inferiority nor causal certification.

## Adjacent work limits the novelty claim

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

## Efficient next artifact

Prepare a technical report from the frozen accepted record: problem and intervention scope; candidate construction and its semantic shortcut; target-aligned baseline definitions; paired training and blind-finalization provenance; complete world-level rankings; limitations and retained negative evidence. Keep the old Stage B development evidence separate from the matched M4 study. Neither belongs in a confirmatory-certification table.

Do not choose a favorable score orientation, add a winning method, select certification margins from M4 outcomes, or claim a new causal result. Do not run restoration, new training, or official-test evaluation. M5 remains held under `M4_CONTINUATION_GATE.md`; this author-led assessment does not satisfy its independent-review requirement.

The present evidence supports a strong reproducibility and debugging artifact. It does not yet justify another result-bearing certification study as a novel research contribution. Preserve the project and its evidence rather than extending experiments merely to reach another milestone.
