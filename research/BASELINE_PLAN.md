# Baseline plan

**Updated:** 2026-09-24  
**Status:** prospective feasibility plan; exact confirmatory baseline set not yet frozen

The purpose of baselines is to separate two questions:

1. can a method **rank** version changes associated with the regression?
2. does intervention evidence **certify** that a ranked change is specifically
   responsible rather than merely influential?

No baseline may use the hidden planted-root identifier.

## Baseline families

### B0 — Random candidate ranking

Purpose: chance reference.

- deterministic seeded permutation;
- evaluated at the version-change level;
- no model or text access.

### B1 — Simple change-level heuristics

At minimum:

- changed-record count / change magnitude;
- whole-change lexical overlap with regressed evaluation examples;
- changed-record-only lexical overlap.

These are intentionally simple and help detect benchmark shortcuts.

### B2 — Modern example-level attribution

Select at least one method that:

- supports transformer classification;
- can score training examples for a defined target evaluation function;
- is available in reproducible code;
- can be executed within the project compute budget.

TRAK is the first method to evaluate for feasibility because it is a
well-established scalable attribution reference and has been demonstrated on
BERT-family language models.

If TRAK is technically incompatible with the frozen setup, document the reason
before choosing the replacement.

### B3 — Gradient / influence cross-check

If B2 alone represents only one attribution family, add a method from a
different family such as TracIn-style checkpoint-gradient attribution or an
appropriate influence-function approximation.

Do not add baselines solely to increase method count.

### B4 — Empirical intervention reference

Where computationally feasible, use retraining/further-training or exact
candidate restoration as the empirical reference signal.

This reference is not itself an efficient debugging method. It defines what the
approximate ranking methods are trying to predict.

### B5 — MRF certification

For each candidate intervention:

- restore the candidate under the frozen release construction;
- use matched stochastic trajectories;
- measure target recovery;
- measure protected-behavior movement;
- compare candidate recovery with nuisance/restoration alternatives;
- allow **abstention** when effects are not distinguishable.

## Attribution target and change-level aggregation

The development attribution target and primary candidate aggregation are now
frozen in [`ATTRIBUTION_TARGET.md`](ATTRIBUTION_TARGET.md).

Primary behavior target:

- all development-eval examples in the frozen target pair;
- correct-vs-other-target logit margin;
- equal evaluation-example weight.

Primary candidate aggregation:

- convert each method's native example score to the prospectively defined
  common suspiciousness orientation;
- sum suspiciousness over all composite-release slots changed by the candidate.

Normalized mean, absolute-score mass, and top-k mass may be reported only as
secondary diagnostics. They do not replace the primary sum because the
counterfactual intervention restores the complete candidate change.

Do not select a target subset, sign, or aggregation rule by checking which one
ranks the hidden root highest.

## Fairness requirements

All methods must use only information declared available to the debugger.

Record:

- model/checkpoint access;
- training-data access;
- gradients/checkpoints required;
- target evaluation examples;
- hidden-truth access;
- compute time;
- number of training/retraining runs.

A method with privileged information should be labeled an oracle/reference,
not compared as though it had the same observability.

## Success criteria for the baseline stage

The baseline stage is complete when:

1. at least one serious attribution baseline runs reproducibly;
2. change-level aggregation is frozen for confirmatory work;
3. shortcut baselines are recorded;
4. compute and information access are documented;
5. the certification layer can be evaluated separately from the ranking layer.

This plan may be narrowed after feasibility work, but not after confirmatory
outcomes are visible.
