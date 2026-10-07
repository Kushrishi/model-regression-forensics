# vNext retrospective software validation — 7 October 2026

Scope: software/metric validation only. No new training, model acquisition,
official Banking77 test access or confirmatory study. No future diagnosis or
restoration threshold was selected from these records.

- Fresh repository-locked dev environment: Python3.12.14, Pydantic2.13.5,
  pytest9.1.1, Ruff0.16.5. Full dev suite passed282 tests, with6 optional-dependency
  skips. Fifteen new ledger checks cover ambiguity, unjustified specificity among
  multiple alternatives, interaction/omitted truth, failed repeated attempts,
  incident-level denominators, identity/reference/budget/nonfinite failures,
  explicit truth separation, exact CLI replay and non-clobber output.
- The retained ambiguous-repair fixture was represented through the new ledger.
  Both existing comparison-policy repairs restore the eight predictions. The
  supplied decision is an ambiguity set, historical cause is not inferred, and
  evaluator truth is separate. This is a retrospective deterministic fixture,
  not an externally grounded release or evidence that a diagnostic algorithm works.
- CLI replay reproduced the example report byte-for-byte. Report SHA-256:
  `bcb6d15090f81a49ed66e219cd5d2e8c55f72dbb2ff6f52694d79e8878171853`.
- All12 accepted M4 artifact hashes matched. At original execution source
  `a5ec718e6081624893fed406eedb1dd5e1405892`, blind aggregation then separate truth
  scoring reproduced both accepted outputs byte-for-byte. This replays retained
  scores, not training or attribution computation. Original random3/3,
  label/lexical1/1 and Grad-Dot/TracIn1/5 ranks are unchanged. Three trajectories
  within each of two worlds remain repetitions, not six incidents.
- A pure Python list check confirmed the diffusers issue's stated eight-slot
  token-layout discrepancy. No Torch/model inference or training was performed;
  reported real connector correlations were not reproduced.

The new schema is explicitly experimental0.1. Supplied hashes and text are not
verified source execution, scientific adequacy or complete-diff certification.
Decision policies remain caller supplied. Local core tests intentionally use no
research dependencies; hosted Research CI remains the separate numerical check.
