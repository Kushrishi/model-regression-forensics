# Attribution target contract

**Updated:** 2026-09-25  
**Status:** frozen for Exp009 development-baseline feasibility before any modern
attribution score is used for candidate ranking

This contract defines what a training-data attribution baseline is allowed to
explain in Experiment 009.

It is intentionally separate from the counterfactual-certification protocol.

## 1. Debugger-visible inputs

A localization baseline may use:

- the known-good baseline model release;
- the regressed composite model release;
- the frozen development evaluation set;
- the five debugger-visible version changes and their changed-slot manifests;
- model parameters, gradients, checkpoints, and training data when required by
  the baseline method.

It may **not** use:

- the hidden planted-root identity;
- restoration outcomes;
- nuisance-vs-root labels;
- the official Banking77 test split;
- any result from the future confirmatory stage.

Candidate identities passed to localization code must come from the opaque
diagnostic manifest produced by `exp009_candidates.py`. Semantic roles such as
"root" or "nuisance" belong only in the separate truth manifest and must not be
loaded by the localization baseline.

The diagnostic manifest may expose neutral stable slot IDs, counts, and hashes.
It must not expose semantic role names or restoration outcomes.

The target behavior itself is not hidden. In the current development benchmark,
the frozen behavior slice is the intent pair:

- `Refund_not_showing_up`
- `request_refund`

The benchmark asks which visible release change best explains degradation of
that already-identified behavior.

## 2. Primary behavior scalar

For an evaluation example `(x, y)` whose true label is one member of the
target pair, let `y_other` be the other member.

Define the correct-vs-paired-target logit margin:

```text
m_theta(x, y) = z_theta(x)[y] - z_theta(x)[y_other]
```

where `z_theta(x)` is the 77-way classifier logit vector.

The primary target function is:

```text
M(theta) = mean(m_theta(x, y) over every development-eval example
                whose true label is in the frozen target pair)
```

Higher `M` is better.

The observed development regression is summarized by:

```text
G_margin = M(theta_baseline) - M(theta_composite)
```

A positive value means the composite release degraded the frozen target pair.

## 3. Why the target is unweighted

Every development-eval example in the frozen target pair contributes equally.

Do **not**:

- keep only baseline-correct/composite-wrong negative flips;
- weight examples by their observed clean-to-composite deterioration;
- choose a favorable subset after seeing attribution scores;
- tune the target function to make the planted root easier to rank.

Those alternatives are useful descriptive diagnostics but would let the
observed failure pattern select the attribution objective after the fact.

The unweighted pairwise margin is fixed before attribution results.

## 4. Common orientation for attribution methods

Every attribution baseline must ultimately produce a slot-level quantity with
the following semantic orientation:

> **Positive suspiciousness means the method predicts that removing/restoring
> this current training contribution would improve the frozen target behavior
> scalar `M`.**

Different methods use different native signs:

- a contribution-to-margin method may assign harmful records negative
  contribution;
- an influence-on-loss method may assign harmful records positive influence on
  loss.

Method adapters must convert the native quantity to the common suspiciousness
orientation from the method's mathematical definition.

The sign may not be chosen by checking which orientation ranks the hidden root
higher.

## 5. Candidate-change aggregation

For candidate release change `C`, let `S_C` be the debugger-visible set of
training slots changed between the clean baseline and the composite release.

The primary change-level score is:

```text
score(C) = sum(slot_suspiciousness(s) for s in S_C)
```

The highest score is the method's first-ranked candidate.

### Why sum is primary

The intervention under evaluation restores the complete candidate change, not
an average slot. Total predicted behavioral effect is therefore the relevant
quantity.

In nuisance-v2 all five candidates currently change the same number of slots,
so sum and mean ranking differ only through score distribution. Future
benchmarks may use unequal change sizes; in those cases a normalized mean can
be reported as a secondary diagnostic but does not replace the predeclared
primary sum.

## 6. Replacement-change limitation

Some version changes replace both text and label, while the current planted
root changes labels while retaining text.

Example-level attribution on the composite model scores the **current**
model-facing records. It does not automatically recover the influence of
baseline records that are absent from the composite training set.

This is another reason nuisance-v2 is only a development localization/shortcut
diagnostic and not a publication-level structurally matched benchmark.

Do not hide this limitation by inventing pseudo-attribution scores for absent
records.

## 7. Baseline-specific target mapping

### TRAK-family baseline

If technically feasible, use a custom model-output function corresponding to
the pairwise target margin in Section 2.

Aggregate target-example attribution across the complete frozen target slice
with equal target-example weight.

Convert per-training-slot contribution to suspiciousness by the method-defined
sign convention and then apply the primary candidate sum.

### TracIn-style baseline

Use the gradient of a target objective whose direction is mathematically
equivalent to improving the pairwise margin.

Checkpoint choice and learning-rate weighting must be prospectively specified.

The TracIn adapter must document its sign orientation before candidate scores
are inspected.

### Simple lexical baselines

Use the same complete target-slice texts. Do not select only failed examples.

These baselines are explicitly intended to reveal structural shortcuts.

## 8. Development vs confirmatory use

This contract is frozen for development feasibility.

Development may determine:

- whether TRAK is technically reproducible on the pinned classifier;
- projection dimension or other engineering settings using method-internal
  guidance and non-root-aware diagnostics;
- whether a second attribution family is needed;
- implementation details required to map slot scores to candidate scores.

Development may **not** select:

- the attribution target based on root ranking;
- the sign based on root ranking;
- a target-example subset based on root ranking;
- a candidate aggregation rule based on official-test outcomes.

Before confirmatory execution, the exact successful baseline implementation and
all remaining hyperparameters must be frozen.

## 9. Required outputs

Every baseline run must record:

- baseline/composite model identities;
- development partition hash;
- target labels;
- target-example count;
- `M(theta_baseline)`;
- `M(theta_composite)`;
- `G_margin`;
- native slot-score definition;
- common suspiciousness sign mapping;
- candidate changed-slot identities/hash;
- primary summed candidate scores;
- candidate ranking;
- runtime and device;
- method/library version and code revision;
- confirmation that the official test split was not loaded.

## 10. Interpretation boundary

A top-ranked candidate is a **diagnostic hypothesis**.

It is not a causal certification.

MRF's separate restoration layer must still test the candidate against nuisance
restorations and matched retraining variability. A ranking method may be
correct while certification abstains, and that disagreement is scientifically
meaningful.
