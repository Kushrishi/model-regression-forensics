# Experiment 009 — Pilot Nuisance Rule v2 Proposal

Status: **PROPOSED — NOT FROZEN**
Date: **2026-09-20**
Model training authorized by this document: **NO**

This proposal follows the completed read-only v1 autopsy.

It does not alter the permanent v1 failure record.

## Goal

Construct legitimate non-root training changes for the Exp009 **development certification pilot**.

The immediate purpose is not to prove that the root is difficult to localize from raw diff structure.

The immediate purpose is:

> compare the stochastic target-recovery effect of restoring the known root against the effects of restoring legitimate non-root changes.

## Proposed conceptual change

Retain all v1 quality and adjacency requirements, but replace:

> at least one observed clean error in **each direction**

with:

> at least one observed clean error in **either direction** across the pooled clean trajectories.

In notation, replace:

[
N(C\rightarrow D) > 0
\quad\land\quad
N(D\rightarrow C) > 0
]

with:

[
N(C\rightarrow D) + N(D\rightarrow C) > 0.
]

## Scientific rationale

The intended nuisance property is **plausible local confusability**, not symmetry of the classifier's mistakes.

Reciprocal confusion is not required for a pair to occupy an ambiguous or nearby decision region.

Asymmetry can arise from:

- different class geometry;
- unequal margins;
- sample-count differences;
- finite evaluation samples;
- calibration differences;
- annotation noise.

The v1 autopsy showed that reciprocal confusion was the sole terminal infeasibility point after the other quality/adjacency gates.

This proposal is therefore a targeted change to the criterion whose scientific interpretation was stronger than necessary.

## Important transparency requirement

The v1 autopsy also showed that this proposed rule yields four disjoint development pairs.

Therefore this v2 proposal is **not confirmatory evidence** and must never be described as prospectively independent of the v1 outcome.

Its correct interpretation is:

> a development rule designed after observing a benchmark-construction failure, but before observing any nuisance-model outcome.

Any later confirmatory world-selection algorithm must be frozen and applied prospectively to fresh worlds/candidates.

## Proposed eligibility rule

An unordered pair `(C, D)` is eligible if:

1. neither intent is a pilot target intent;
2. each has at least 66 development-train examples;
3. each has at least 20 development-eval examples;
4. each has minimum clean recall >= 0.90 across the three frozen clean trajectories;
5. their intent names have non-zero token Jaccard overlap;
6. pooled clean predictions contain at least one `C -> D` **or** `D -> C` error.

No nuisance-model outcome may enter selection.

## Proposed ranking

Keep the existing v1 ranking structure as closely as possible.

For an eligible pair:

1. descending symmetric pooled confusion rate

[
0.5\left(
\frac{N(C\rightarrow D)}{N(C)}
+
\frac{N(D\rightarrow C)}{N(D)}
\right)
]

even when one direction is zero;

2. descending pooled pair-confusion count

[
N(C\rightarrow D) + N(D\rightarrow C);
]

3. descending lexical Jaccard;
4. lexicographic pair order.

Then greedily select four intent-disjoint pairs.

This proposal changes **eligibility**, not the general ranking logic.

## Nuisance mechanism

The existing balanced, dataset-label-consistent cross-intent refresh/reweighting mechanism may be retained for the **development certification pilot** if its structural audit still passes.

Every nuisance should continue to match the root on:

- 66 changed stable slots;
- two-intent scope;
- 33 changes per direction;
- zero aggregate label-count delta;
- fixed training-slot count;
- deterministic construction;
- intervention/restoration semantics.

However, the mechanism's text-change difference from the root remains a known structural mismatch.

## Claim restriction

If v2 is frozen in this form, the resulting pilot may support statements about:

- root-vs-nuisance intervention-effect separation;
- stochastic certification behavior;
- nuisance restoration effects;
- protected-metric behavior.

It may **not by itself** support a strong claim that:

> MRF can non-trivially localize the hidden root from equally matched debugger-visible candidate changes.

That stronger claim requires a construction without structural leakage.

## Before freezing v2

Run one final **read-only selection audit** that prints:

- every eligible v2 pair;
- ranking statistics;
- greedy disjoint selection;
- count/intent disjointness;
- root-intent exclusion;
- structural footprint expected for each selected nuisance.

Then review the selected pairs for implementation defects only—not for favorable model behavior.

No nuisance model may be trained before v2 is explicitly marked **FROZEN**.
