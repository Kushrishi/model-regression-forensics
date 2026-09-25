# Experiment 009 — Pilot Nuisance-Change Construction v2

Status: **development pilot rule — FROZEN before the first nuisance-model training result**.
Frozen: **2026-09-20**

This document governs nuisance construction for the current Exp009 Banking77 **development certification pilot only**.

It does not alter the permanent v1 failure record and it is not the Experiment 009 confirmatory protocol.

## Provenance

The sequence leading to this rule is intentionally preserved:

1. nuisance rule v1 was frozen before nuisance-model outcomes;
2. v1 produced zero eligible pairs and stopped;
3. no nuisance model was trained;
4. a read-only v1 autopsy localized infeasibility to the reciprocal-confusion conjunction;
5. v2 was proposed explicitly after observing that development construction failure;
6. a final read-only v2 selection/structural audit was run;
7. only after that audit was this rule frozen.

Therefore v2 is a **development rule informed by v1 failure**.

It must never be described as confirmatory or as prospectively independent of the v1 outcome.

## Purpose

Experiment 008 showed that successful restoration of a planted root is not sufficient for unique causal certification because non-root restorations may also move the target metric.

The current Exp009 pilot therefore needs legitimate non-root changes whose restoration effects can be compared with the root under matched stochastic trajectories.

The primary pilot question is:

> Does restoration of the known planted root produce target recovery that is materially and reliably distinguishable from restoration of legitimate non-root changes and from ordinary retraining variability?

This v2 nuisance construction is frozen for that **certification** question.

It is not sufficient by itself for a strong blinded-localization benchmark claim because root and nuisance diffs remain structurally distinguishable.

## Pilot root

The accepted development root remains the frozen `1/4` symmetric label-mapping fault on:

- `Refund_not_showing_up`
- `request_refund`

with:

- 33 changed slots in each direction;
- 66 changed slots total;
- aggregate label mass preserved.

The accepted root dose and its three-trajectory regression evidence are unchanged.

## Eligibility rule v2

After excluding the two pilot target intents, an unordered intent pair `(C, D)` is eligible only if all of the following hold using the frozen Banking77 development partition and the three preserved clean seven-epoch trajectories:

1. both intents have at least 66 examples in `development_train`;
2. both intents have at least 20 examples in `development_eval`;
3. both intents have minimum per-intent recall >= 0.90 across all three clean trajectories;
4. their intent names have non-zero token Jaccard overlap after lowercasing and splitting on `_`;
5. pooled clean predictions contain at least one observed `C -> D` **or** `D -> C` error.

The v2 scientific concept is **plausible local confusability**, not symmetric confusion.

Reciprocal empirical confusion is not required because observed decision-boundary ambiguity can be asymmetric under finite evaluation samples, unequal margins, differing class geometry, calibration differences, or annotation noise.

## Difference from v1

v1 required:

[
N(C\rightarrow D) > 0
\quad\land\quad
N(D\rightarrow C) > 0.
]

v2 requires:

[
N(C\rightarrow D) + N(D\rightarrow C) > 0.
]

All other eligibility concepts remain unchanged.

## Ranking rule

Eligible pairs are ranked deterministically by:

1. descending symmetric pooled confusion rate

[
0.5\left(
\frac{N(C\rightarrow D)}{N(C)}
+
\frac{N(D\rightarrow C)}{N(D)}
\right);
]

2. descending pooled pair-confusion count

[
N(C\rightarrow D) + N(D\rightarrow C);
]

3. descending lexical Jaccard overlap;
4. lexicographic pair order.

Starting from the top, select the first four pairs greedily subject to intent disjointness.

Neither pilot target intent may appear in a nuisance pair.

## Frozen selected nuisance pairs

The final read-only audit produced exactly four eligible, intent-disjoint pairs:

1. `activate_my_card` ↔ `card_not_working`
   - symmetric confusion rate: 0.01562500
   - pooled pair-confusion count: 3
   - directional counts: 3 / 0
   - lexical Jaccard: 0.200000

2. `card_about_to_expire` ↔ `getting_spare_card`
   - symmetric confusion rate: 0.01282051
   - pooled pair-confusion count: 2
   - directional counts: 0 / 2
   - lexical Jaccard: 0.166667

3. `card_payment_wrong_exchange_rate` ↔ `exchange_charge`
   - symmetric confusion rate: 0.01010101
   - pooled pair-confusion count: 2
   - directional counts: 2 / 0
   - lexical Jaccard: 0.166667

4. `cash_withdrawal_charge` ↔ `cash_withdrawal_not_recognised`
   - symmetric confusion rate: 0.00520833
   - pooled pair-confusion count: 1
   - directional counts: 0 / 1
   - lexical Jaccard: 0.400000

These pairs were selected before any nuisance-model outcome was observed.

They may not be manually substituted because subsequent training results are inconvenient.

## Nuisance update mechanism

The v1 balanced cross-intent refresh/reweighting mechanism is retained for this development certification pilot.

For nuisance artifact `j` over pair `(C, D)`:

- select 33 recipient slots currently occupied by dataset-label `C`;
- select 33 recipient slots currently occupied by dataset-label `D`;
- select 33 distinct donor records labeled `D` outside the recipient set;
- select 33 distinct donor records labeled `C` outside the recipient set;
- replace each selected `C` recipient with one `D` donor record;
- replace each selected `D` recipient with one `C` donor record;
- keep the donor records in their original slots.

The resulting update:

- changes 66 stable slots;
- changes text in all 66 selected slots;
- preserves aggregate label counts;
- preserves total slot count;
- changes empirical example weighting through duplicated donor content;
- remains consistent with the frozen Banking77 dataset labels.

Because published work has reported potential label noise in Banking77, the project should use the term **dataset-label-consistent** rather than claiming all source labels are objectively semantically correct.

## Structural matching

Each nuisance matches the root on:

- changed-slot count: 66;
- two-intent scope;
- 33 changes per direction;
- zero aggregate label-count delta;
- fixed total slot count;
- deterministic construction;
- restoration semantics.

However, one known mismatch remains:

### Root

- label changes: yes;
- text changes: 0.

### Nuisances

- label changes: yes;
- text changes: 66 each.

Therefore:

> **STRUCTURAL MATCH FOR BLINDED DIAGNOSIS: FAIL**

A diagnostic with access to detailed changed-record structure could potentially identify the planted root using this mismatch.

## Claim boundary

The v2 pilot may support evidence about:

- root-vs-nuisance intervention-effect separation;
- stochastic restoration variability;
- protected-behavior effects;
- causal-certification criteria;
- failure/abstention behavior.

The v2 pilot may **not by itself** support a strong claim that:

> a debugger non-trivially localized the root among structurally matched candidate changes.

That stronger MRF-Bench claim requires a future construction that removes or masks the structural leakage.

## Required pre-training checks

Before the first nuisance model is trained:

1. implementation must preserve v1 behavior separately;
2. v2 pair selection must reproduce the frozen four pairs exactly;
3. nuisance construction must reproduce 66 changed slots per nuisance;
4. nuisance pairs must be intent-disjoint;
5. nuisance changed slots must be mutually disjoint and disjoint from root changed slots;
6. every nuisance must have zero aggregate label-count delta;
7. structural leakage must be reported, not hidden;
8. full tests/lint/format checks must pass;
9. official Banking77 test split must remain untouched.

No nuisance-model training is authorized until those implementation checks pass.

## Integrity rules

- v1 remains permanently recorded as infeasible;
- v2 was designed after v1 construction failure and is development-only;
- no nuisance outcome may alter pair selection;
- no pair may be manually swapped after training;
- the root dose remains unchanged;
- the official Banking77 test remains embargoed;
- development and confirmatory evidence remain separate;
- the confirmatory nuisance rule remains unfrozen;
- the final MRF-Bench diagnosis construction remains unfrozen.

## Frozen status

**PILOT_NUISANCE_RULE_V2=FROZEN**

**NUISANCE_MODEL_TRAINING_PERFORMED_AT_FREEZE=NO**

**OFFICIAL_BANKING77_TEST_LOADED_AT_FREEZE=NO**
