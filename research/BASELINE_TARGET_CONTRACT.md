# Exp009 baseline target contract

**Frozen:** 2026-09-25
**Evidence class:** development
**Attribution outcomes observed before freeze:** none

This document fixes the behavior that attribution baselines are allowed to
explain and the rule that maps example-level attribution to debugger-visible
version changes.

The contract is method-agnostic. TRAK is the first planned serious attribution
baseline, but changing attribution libraries must not change the target or
aggregation rule after result-bearing outcomes are visible.

## 1. Incident observable

The development incident is:

- known-good release: baseline;
- regressed release: composite;
- affected behavior slice: Refund_not_showing_up and request_refund;
- candidate causes: the five debugger-visible version changes in the frozen
  composite release.

The affected behavior slice is treated as incident metadata. The debugger is
allowed to know what behavior regressed.

The debugger is not allowed to know which candidate change is the planted root.

For a future publication-level benchmark, the affected slice must itself be
derived from observable baseline-versus-regressed behavior rather than planted
truth. The current Exp009 v2 benchmark remains development-only.

## 2. Attribution target examples

Use only frozen Banking77 development-eval examples whose true label is one of
the two incident labels.

Do not use:

- official Banking77 test examples;
- hidden root identifiers;
- nuisance/root role labels;
- counterfactual restoration outcomes.

Target examples retain their true class labels.

## 3. Model output semantics

Every attribution implementation must expose a signed score with the convention:

positive score = the training slot supports the target example's correct-class
margin.

For multiclass classification, the preferred target quantity is the standard
correct-class log-odds margin:

    f(z; theta)
      = logit(correct)
        - logsumexp(logit(all incorrect classes))

This matches the classification model-output family used by TRAK.

A method with the opposite sign must be converted to the convention above before
change-level aggregation.

## 4. Class-balanced target support

Let a_r(i, z) be the signed attribution from stable training slot i to target
example z for release r.

For each incident class y, compute the mean attribution over target examples in
that class. Then average the two class means.

The two incident classes receive equal weight even if their development-eval
example counts differ.

## 5. Release-differential slot score

For every stable training slot:

    Delta(i)
      = A_composite(i, target) - A_baseline(i, target)

Interpretation:

- negative Delta(i): the slot became less supportive of correct target-slice
  behavior in the regressed release;
- positive Delta(i): the slot became more supportive.

The baseline and composite attribution computations must use the same target
examples and the same score orientation.

## 6. Version-change suspicion score

For candidate version change C with debugger-visible changed stable slots S_C:

    Suspicion(C)
      = - mean_i Delta(i), for i in S_C

Higher is more suspicious.

The current five v2 candidate changes each modify 66 stable slots, so mean and
sum give the same ordering. Mean is frozen because it remains interpretable if a
future benchmark contains differently sized changes.

The candidate grouping is obtained from version-diff metadata only. No
root/nuisance role metadata enters the score.

## 7. Protected-behavior diagnostic

Repeat the same class-balanced support calculation over the 75 non-target
intents as a secondary diagnostic.

Protected behavior is not combined with target harm into a tunable composite
ranking score during this development stage.

Report it separately to detect broad nonspecific effects.

## 8. Structural-leak diagnostic

Nuisance v2 is intentionally not considered a publication-grade localization
benchmark because its diff structure leaks candidate type:

- planted root: label changes, text unchanged;
- nuisances: label changes and text changes.

Record simple diff-structure features for every candidate:

- changed-slot count;
- label-change count;
- text-change count.

A trivial rule that can isolate the root from these features is evidence of
benchmark leakage, not evidence of a good localization method.

## 9. TRAK feasibility decision

TRAK is the first serious attribution baseline because:

- it has an established multiclass/text-classification formulation;
- it targets counterfactual training-example effects;
- its public implementation includes a BERT text-classification example;
- it is sufficiently different from MRF's exhaustive restoration layer to be a
  meaningful ranking baseline.

The stock TRAK text-classification adapter assumes token_type_ids. DistilBERT
does not use them, so Exp009 requires a small custom DistilBERT model-output
adapter with the same correct-class-margin semantics.

Do not modify the target contract to accommodate TRAK implementation details.

## 10. Checkpoint and trajectory use

The baseline ranking study should use the matched Stage-A model families:

- three baseline trajectories;
- three composite trajectories.

Attribution may be aggregated across the three matched trajectories, but the
aggregation rule must be fixed before attribution outcomes are inspected.

The preferred development rule is to average class-balanced slot support across
the three trajectories within each release, then apply the release-differential
and change-level equations above.

## 11. Execution order

1. freeze this contract;
2. implement and test method-agnostic aggregation;
3. implement a DistilBERT-compatible TRAK adapter and smoke-test it without
   Exp009 result-bearing scoring;
4. execute the six frozen Stage-A baseline/composite trajectories;
5. evaluate the Stage-A regression gate;
6. if Stage A passes, compute baseline attribution rankings from those six
   checkpoints;
7. only then decide whether to spend the additional 15 Stage-B restoration
   runs.

This order avoids paying for all 21 runs before knowing whether both the
regression substrate and the ranking baseline are scientifically usable.

## 12. Prohibited adaptation

After attribution results are observed, do not change:

- target intent pair;
- target example membership;
- class balancing;
- attribution sign convention;
- baseline-versus-composite differential;
- mean-over-changed-slots aggregation;
- candidate grouping;
- target/protected separation

to improve root ranking.

Any later publication benchmark may define a new prospective contract, but it
must not retroactively alter this development result.

## Frozen marker

EXP009_BASELINE_TARGET_CONTRACT=FROZEN
