# M4 continuation gate

**Date:** 2026-10-03 UTC  
**Decision:** accept M4; hold M5 execution and protocol development pending a focused contribution review.

## Accepted evidence

Source-pinned run `37083633414` completed three clean trainings, six composite trainings, all six blind scoring records, complete blind aggregation, and separate truth scoring. The blind aggregate was persisted before truth access. Local replay from the six retained records reproduced both the hosted blind aggregate and truth evaluation byte-for-byte. See `M4_RESULT.json` and `M4_RESULT_ARTIFACTS/` for exact identities and outputs.

| Frozen method | World 00 root rank | World 01 root rank | First-ranked root |
| --- | --- | --- | --- |
| B0 deterministic random | 3 | 3 | 0 / 2 worlds |
| B1 target-label overlap | 1 | 1 | 2 / 2 worlds |
| B2 lexical Jaccard | 1 | 1 | 2 / 2 worlds |
| B3 final-checkpoint Grad-Dot | 1 | 5 | 1 / 2 worlds |
| B4 seven-checkpoint TracIn | 1 | 5 | 1 / 2 worlds |

These are descriptive development results from two constructed worlds. The three paired training trajectories are repetitions within each world, not six independent benchmark worlds.

## Interpretation

Structural matching did not make localization difficult under the debugger-visible target information. B1 has direct access to the known target labels while candidate label pairs are disjoint. Its success is therefore an expected semantic shortcut in the frozen design. B2 independently ranks the root first in both worlds. The evaluated model-based methods add no top-1 benefit over these visible-change baselines and rank the root last in world 01.

This does not establish a general inferiority of Grad-Dot or TracIn. It does remove support for presenting this benchmark as evidence of a difficult localization problem or an improved localization method. Correct localization still does not establish causal specificity; M4 contains no restoration or confirmatory certify/abstain outcome.

## Continuation conditions

M5 should proceed only if an independent contribution review identifies a useful certification question that is not made trivial by the disjoint target/candidate construction. That review must explain what plausible alternatives can defeat unique attribution, how a prospective certify/abstain decision improves on a simple diagnostic plus paired retraining, and what new evidence would be worth the compute.

M4 outcomes must not be used to redesign these worlds, select a favorable scoring orientation, add winning methods, choose certification margins, or claim confirmatory evidence. No new training, restoration, official-test access, or cross-substrate experiment is authorized by this gate.

If that justification fails, close the project as a bounded reproducible technical report covering the distinction between localization and certification and the failure of model-based diagnostics to improve on simple baselines in this design. Preserve all negative evidence.
