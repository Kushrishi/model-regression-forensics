# Exp009 attribution target contract

**Version:** 2
**Frozen:** 2026-09-25
**Evidence class:** development
**Modern attribution rankings observed before freeze:** none

This is the single canonical attribution contract for Experiment 009.

It prospectively supersedes two earlier pre-result drafts that disagreed about
the model-output scalar, release differencing, and change aggregation. Neither
draft had been used to produce a Banking77 attribution ranking when this version
was frozen. Git history preserves both earlier designs.

The attribution layer generates diagnostic hypotheses. It is separate from MRF's
counterfactual certification layer.

## 1. Debugger-visible incident

The debugger may use:

- the known-good baseline release/model;
- the regressed composite release/model;
- the frozen development-evaluation partition;
- the known affected behavior slice:
  Refund_not_showing_up and request_refund;
- five opaque candidate version changes and their changed-slot sets;
- model parameters, gradients, checkpoints, and training data required by the
  baseline method.

The debugger may not use:

- planted-root identity;
- root/nuisance role labels;
- restoration outcomes;
- the official Banking77 test split;
- future confirmatory outcomes.

Localization code consumes only the opaque diagnostic candidate manifest.

## 2. Why version 2 is necessary

The frozen Stage-A incident endpoint is target macro recall across the two
affected intents.

One earlier attribution draft used only the logit margin between those two
intents. That is too narrow because a target example can fail by being predicted
as any of the other 75 intents.

Another draft used the standard multiclass correct-class margin but subtracted
baseline-model attribution from composite-model attribution. That difference is
not itself a clean counterfactual estimate of a version change because both the
trained model and, for changed slots, the model-facing record differ across
releases.

Version 2 keeps the multiclass target that matches the classification incident
and removes the unsupported attribution-difference interpretation.

## 3. Target evaluation slice

Use every development-eval example whose true label is one of the two incident
intents.

Within each intent every example receives equal weight. The two incident intents
then receive equal weight. This mirrors macro-recall weighting even if their
example counts differ.

Do not select negative flips, select examples by observed deterioration, or tune
example weights after attribution scores are visible.

## 4. Differentiable target scalar

For a target example (x, y), define

    f_theta(x, y)
      = logit_y
        - logsumexp(logit_c for every c != y)

For each incident intent y, M_y(theta) is the mean f_theta over that intent's
development-eval examples.

The class-balanced incident scalar is

    M(theta) = mean of M_y(theta) over the two incident intents.

Higher is better.

For incident characterization only,

    G_margin = M(theta_baseline) - M(theta_composite).

A positive G_margin means the regressed model has a worse differentiable margin
on the affected slice. G_margin is descriptive and does not replace the frozen
Stage-A macro-recall gate.

## 5. Why this target is primary

This scalar:

- responds when probability mass moves from the correct intent to any wrong
  intent;
- gives both affected intents equal weight;
- is differentiable;
- matches TRAK's standard multiclass classification model-output family;
- uses no hidden root truth;
- is frozen before candidate attribution.

It is a surrogate for target-slice classification quality, not a claim that
margin and recall are identical.

## 6. Primary attribution model state

The localization question is which current training contributions in the
regressed composite release detract from the affected behavior.

Therefore the primary candidate ranking uses attribution from the composite
model/release.

The baseline model remains necessary to establish the incident, pass Stage A,
and compute G_margin. Baseline attribution may be reported only as a secondary
diagnostic and is not subtracted from composite attribution in the primary
candidate score.

## 7. Slot-score orientation

TRAK uses higher scores for training examples that support the target model
output. Under this contract:

    slot_suspiciousness = - class_balanced_target_support.

Larger positive suspiciousness therefore means the current composite-release
training slot is predicted to detract more strongly from correct target-slice
behavior.

Other methods must freeze their sign mapping from their mathematical definition
before hidden truth is inspected.

## 8. Stochastic-trajectory aggregation

For each of the three composite trajectories:

1. compute attribution to every target evaluation example;
2. average scores within each incident intent for every stable training slot;
3. average the two intent means to obtain class-balanced slot support.

Then average slot support across all three composite trajectories. Only after
trajectory averaging is the sign converted to suspiciousness.

No trajectory may be discarded because it gives an unfavorable ranking.

## 9. Candidate aggregation

For opaque candidate C with changed-slot set S_C:

    score(C) = sum(slot_suspiciousness(s) for s in S_C).

Higher is more suspicious.

Sum is primary because the downstream intervention restores the whole candidate
change. In nuisance-v2 all five candidates currently change 66 slots, so mean
and sum have identical ordering. A normalized mean may be reported later only
as a secondary diagnostic.

## 10. Replacement-change limitation

Some nuisance-v2 changes replace both text and label whereas the planted root
changes labels while retaining text.

Composite-model attribution evaluates the model-facing record currently present.
It does not directly estimate the contribution of a baseline record that has
been replaced and is absent from composite training.

Do not manufacture pseudo-attribution for absent records. This limitation is
one reason nuisance-v2 remains a development benchmark rather than the final
publication-grade localization benchmark.

## 11. Shortcut diagnostic

Separately record simple candidate-diff features:

- changed-slot count;
- text-change count;
- label-change count.

If a trivial structural rule isolates a candidate, that is evidence of
benchmark leakage, not evidence for sophisticated localization.

## 12. Head-restricted TRAK development baseline

The first development TRAK baseline is explicitly head-restricted:

- pre_classifier.weight
- pre_classifier.bias
- classifier.weight
- classifier.bias
- projection dimension 512
- projection seed 0
- padded DistilBERT-compatible adapter
- standard multiclass target from Section 4.

This is not represented as full-model TRAK. A broader-gradient baseline can be
added only through a separate prospective decision, not in response to hidden
root ranking.

## 13. Required outputs

Every result-bearing attribution run must record:

- model/release identities and code revision;
- development-partition hash;
- target labels and target counts per intent;
- per-intent and class-balanced M for baseline and composite;
- G_margin;
- native attribution definition and sign conversion;
- all three trajectory IDs;
- opaque candidate-manifest hash;
- candidate changed-slot hashes;
- summed candidate scores and ranking;
- runtime/device and library version;
- confirmation that localization code did not load truth roles;
- confirmation that the official test split was not loaded.

## 14. Interpretation boundary

A top-ranked candidate is a diagnostic hypothesis, not causal certification.

The separate restoration layer still tests every candidate against nuisance
restorations under matched stochastic trajectories. Ranking can succeed while
certification abstains; that is a legitimate result.

## 15. Frozen execution order

1. merge this canonical contract;
2. verify dependency-free tests;
3. verify the frozen Stage-A release/training protocol is unchanged;
4. run Stage A only: three baseline plus three composite models;
5. apply the frozen Stage-A macro-recall/protected-behavior gate;
6. if Stage A passes, compute G_margin;
7. run head-restricted TRAK on the three composite checkpoints;
8. record ranking without changing target, sign, or aggregation;
9. only then decide whether to spend the additional 15 Stage-B restoration runs.

Official Banking77 test data remain embargoed.

EXP009_ATTRIBUTION_TARGET_V2=FROZEN
