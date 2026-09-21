# Experiment 009 — Nuisance Rule v1 Autopsy

Status: **completed read-only development analysis**
Date: **2026-09-20**
Model training performed: **NO**
Official Banking77 test split loaded: **NO**

## Purpose

The frozen pilot nuisance-selection rule produced zero eligible pairs and therefore stopped before nuisance-model training.

This autopsy asks **why** the rule failed. It does not revise the rule, train models, or reinterpret the frozen v1 outcome.

The analysis uses only:

- the frozen Banking77 development partition;
- the three preserved clean seven-epoch development trajectories;
- the frozen v1 nuisance eligibility criteria.

## Frozen v1 eligibility rule

After excluding the two pilot target intents, an unordered pair had to satisfy:

1. both labels have at least 66 development-train examples;
2. both labels have at least 20 development-eval examples;
3. both labels have minimum clean recall >= 0.90 across all three clean trajectories;
4. intent-name token Jaccard overlap is > 0;
5. pooled clean predictions contain at least one error in **both** directions.

The final requirement was bidirectional empirical confusion.

## Autopsy result

Candidate labels after target exclusion: **75**

All unordered candidate pairs:

[
inom{75}{2} = 2775
]

### Individual gates

| Gate | Surviving pairs | Participating labels | Greedy disjoint capacity |
|---|---:|---:|---:|
| train >= 66 for both | 2485 | 71 | 35 |
| eval >= 20 for both | 2080 | 65 | 32 |
| minimum clean recall >= 0.90 for both | 435 | 30 | 15 |
| lexical Jaccard > 0 | 532 | 70 | 33 |
| any-direction confusion | 177 | 72 | 33 |
| bidirectional confusion | 38 | 46 | 20 |

No individual criterion is infeasible.

### Frozen cumulative gate order

| Cumulative stage | Surviving pairs | Participating labels | Greedy disjoint capacity |
|---|---:|---:|---:|
| train count | 2485 | 71 | 35 |
| + eval count | 2080 | 65 | 32 |
| + clean recall | 406 | 29 | 14 |
| + lexical overlap | 63 | 23 | 10 |
| + bidirectional confusion | **0** | **0** | **0** |

The rule becomes infeasible **only at the final bidirectional-confusion conjunction**.

## Diagnostic intersections

Among the 406 pairs that satisfy count + recall requirements:

- 63 also have lexical overlap;
- 14 show confusion in at least one direction;
- 2 show bidirectional confusion.

Among the 63 pairs that also have lexical overlap:

- **4** show confusion in at least one direction;
- **0** show bidirectional confusion.

Those four any-direction-confused pairs are intent-disjoint.

Therefore the v1 failure is not a general lack of high-quality, semantically adjacent nuisance candidates.

It is specifically the conjunction:

> high clean quality + sufficient counts + lexical adjacency + **reciprocal** clean confusion.

## Four v1 near-miss pairs

The read-only autopsy identifies four pairs that satisfy every frozen v1 requirement except reciprocal confusion:

1. `activate_my_card` ↔ `card_not_working`
   - lexical Jaccard: 0.20
   - pooled directional errors: 3 / 0

2. `card_about_to_expire` ↔ `getting_spare_card`
   - lexical Jaccard: 1/6 ≈ 0.1667
   - pooled directional errors: 0 / 2

3. `card_payment_wrong_exchange_rate` ↔ `exchange_charge`
   - lexical Jaccard: 1/6 ≈ 0.1667
   - pooled directional errors: 2 / 0

4. `cash_withdrawal_charge` ↔ `cash_withdrawal_not_recognised`
   - lexical Jaccard: 0.40
   - pooled directional errors: 0 / 1

These are **development observations**, not confirmatory selections.

## Interpretation

The scientific concept v1 was trying to enforce was:

> nuisance pairs should represent legitimate, non-root training changes over intent pairs that are sufficiently learnable and plausibly adjacent/confusable.

Requiring at least one observed mistake in **both directions** is stronger than that concept requires.

Decision-boundary ambiguity can be asymmetric.

A classifier may systematically map examples from class A toward B without making the reverse mistake, especially with:

- unequal class geometry;
- unequal sample counts;
- asymmetric decision margins;
- calibration differences;
- annotation noise;
- finite evaluation samples.

Therefore:

> absence of an observed B→A error does not establish that A and B are not a plausible confusable pair.

The autopsy supports treating **empirical confusability** and **reciprocal confusion** as distinct properties.

## External validity note: Banking77

Banking77 is intentionally fine-grained and has partially overlapping intent categories.

Published work has also reported substantial **potential** label noise in Banking77. Therefore future MRF writing should avoid treating the dataset annotation as perfect semantic ground truth.

For nuisance construction, the safer term is:

> **dataset-label-consistent**

rather than claiming that every source record is objectively taxonomy-correct.

## Important separate issue: structural leakage

The pair-selection failure is not the only nuisance-design concern.

The current pilot root:

- keeps text fixed;
- changes labels.

The current nuisance mechanism:

- changes text;
- changes labels coherently according to the frozen dataset annotation.

Therefore the root and nuisance changes are structurally distinguishable if a diagnostic receives detailed changed-record metadata.

The Exp009 development protocol already warned about this issue.

### Consequence

The existing nuisance mechanism may remain usable for a **certification-focused pilot**, where the question is whether root intervention effects separate from non-root intervention effects.

It should **not automatically become the final MRF-Bench diagnosis construction** because root identity may be too easy to infer structurally.

A future confirmatory/full-benchmark construction must either:

1. match root and nuisance observable structure more closely; or
2. define an access track that hides the leaking fields; or
3. explicitly restrict the claim to causal certification rather than localization.

## Conclusion

Frozen nuisance rule v1 remains:

**FROZEN_PILOT_RULE_INFEASIBLE**

The autopsy localizes the failure to the final reciprocal-confusion condition.

No scientific result from v1 is changed.

A development v2 may now be proposed, but it must:

- explain why reciprocal confusion is scientifically necessary or unnecessary;
- be frozen before nuisance-model outcomes;
- remain explicitly development-only;
- separately address structural root/nuisance leakage before any confirmatory diagnosis claim.
