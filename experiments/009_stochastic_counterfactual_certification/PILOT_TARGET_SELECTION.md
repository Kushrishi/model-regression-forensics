# Experiment 009 — Pilot Target-Pair Selection

Status: **development pilot rule — not confirmatory world selection**.

This file records the first mechanically selected Banking77 target pair for
Experiment 009 pilot corruption development. The selection uses only the frozen
development partition and the three completed clean seven-epoch trajectories.
The official Banking77 test split remains embargoed.

## Clean evidence used

The selected clean training configuration is the pinned DistilBERT sequence
classifier trained for seven epochs with batch size 32, learning rate `2e-5`,
weight decay `0.01`, warmup ratio `0.10`, and maximum sequence length 128.

Three independent clean trajectories were evaluated on the same frozen
`development_eval` membership. Their aggregate performance was approximately
89.6% mean accuracy and 88.4% mean macro recall with low between-trajectory
variation.

No planted regression, restoration outcome, or official test example was used
for this selection.

## Deterministic pilot eligibility rule

For an unordered pair of Banking77 intents `(A, B)` to be eligible for the first
pilot target pair, all of the following must hold using only the three clean
seven-epoch development trajectories:

1. `A` and `B` each have minimum per-intent recall >= 0.90 across the three
   clean trajectories.
2. `A` and `B` each have at least 20 examples in `development_eval`.
3. Their intent names have non-zero token Jaccard overlap after lowercasing and
   splitting on `_`. This is a simple prospective proxy for taxonomy adjacency,
   not a claim of semantic equivalence.
4. Clean pooled predictions show at least one `A -> B` error and at least one
   `B -> A` error across the three trajectories. This requires evidence of
   bidirectional natural confusability before any corruption is introduced.

Eligible pairs are ranked deterministically by:

1. descending mean bidirectional confusion rate;
2. descending pooled mutual-confusion count;
3. descending lexical Jaccard overlap;
4. lexicographic order of the pair labels as the final tie-break.

The rule is fixed before any corrupted-model outcome is observed.

## Selected pilot pair

Applying the rule to the completed clean-development audit selects:

- `Refund_not_showing_up`
- `request_refund`

Clean audit properties for this pair:

- worst clean recall across both intents and all three trajectories: at least
  `0.912`;
- pooled clean mutual-confusion count: `4`;
- clean direction counts: `1` in one direction and `3` in the other;
- pooled bidirectional confusion rate: approximately `0.0199`;
- lexical Jaccard overlap: `0.200`.

This pair is therefore suitable for the **first development pilot** of the
preferred symmetric taxonomy/label-mapping fault. It is not reserved for a
confirmatory world.

## Guardrails

- The selected pair must not be replaced merely because corruption calibration
  is inconvenient or produces an unfavorable pilot result.
- Corruption dose remains a development variable and is not selected by this
  file.
- Nuisance-construction rules remain undeclared by this file.
- Confirmatory target-world selection remains unfrozen and should, where
  practical, avoid reusing these two pilot intents.
- The official Banking77 test split remains untouched.
- Pilot outcomes are not confirmatory evidence.

## Current status

- clean model configuration selected: **yes**;
- first pilot target pair selected: **yes**;
- corruption dose selected: **no**;
- nuisance changes constructed: **no**;
- confirmatory target-world rule frozen: **no**;
- official test split loaded: **no**;
- confirmatory result observed: **no**.
