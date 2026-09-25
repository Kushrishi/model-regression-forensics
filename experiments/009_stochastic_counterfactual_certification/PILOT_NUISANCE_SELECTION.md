# Experiment 009 — Pilot Nuisance-Change Construction

Status: **development pilot rule — frozen before the first nuisance-model training result**.

This document governs nuisance construction for the current Banking77 development pilot only. It is not the Experiment 009 confirmatory protocol.

## Purpose

Experiment 008 showed that a planted-root restoration can recover the target while non-root restorations also move behavior materially. Experiment 009 therefore requires nuisance changes that are real model-facing updates rather than obvious no-ops, while remaining label-correct and prospectively constructed.

The current pilot root is the accepted `1/4` symmetric mapping fault on:

- `Refund_not_showing_up`
- `request_refund`

with `33` changed slots in each direction, `66` total.

The nuisance construction below is fixed before any nuisance candidate or restoration model is trained.

## Nuisance-pair eligibility

Nuisance pairs are selected only from the frozen Banking77 development partition and the three completed clean seven-epoch trajectories. The official Banking77 test split remains embargoed.

For an unordered protected-intent pair `(C, D)` to be eligible:

1. neither intent may be one of the pilot target intents;
2. both intents must have minimum per-intent recall `>= 0.90` across the three clean trajectories;
3. both intents must have at least `20` examples in `development_eval`;
4. both intents must have at least `66` examples in `development_train`, allowing `33` recipient slots and `33` distinct donor slots per intent;
5. their intent names must have non-zero token Jaccard overlap after lowercasing and splitting on `_`;
6. pooled clean predictions must contain at least one `C -> D` error and at least one `D -> C` error across the three trajectories.

Eligible pairs are ranked by the same clean-only ordering used for the pilot target-pair audit:

1. descending mean bidirectional confusion rate;
2. descending pooled mutual-confusion count;
3. descending lexical Jaccard overlap;
4. lexicographic pair order as the final tie-break.

Starting from the top of that ranking, select the first four pairs greedily subject to **intent disjointness**. No intent may appear in more than one nuisance pair, and neither pilot target intent may appear in a nuisance pair.

If fewer than four disjoint pairs satisfy this rule, nuisance construction stops and that outcome is recorded before changing the rule.

## Nuisance update mechanism

Each nuisance is a balanced, taxonomy-correct **cross-intent refresh/reweighting update** over one selected adjacent intent pair `(C, D)`.

For nuisance artifact `j`:

- select `33` recipient slots whose clean label is `C`;
- select `33` recipient slots whose clean label is `D`;
- select `33` distinct donor records labeled `D` that are not recipient slots;
- select `33` distinct donor records labeled `C` that are not recipient slots;
- replace each selected `C` recipient's model-facing text and label with one selected `D` donor's text and correct label `D`;
- replace each selected `D` recipient's model-facing text and label with one selected `C` donor's text and correct label `C`.

Recipient and donor selection must be deterministic from a versioned nuisance namespace, nuisance index, unordered intent pair, stable slot ID, and source content ID. Pairing of ranked recipients to ranked opposite-label donors must also be deterministic.

The source identity attached to each stable slot remains unchanged; only the model-facing record occupying that slot changes.

The donor records remain in their original slots. Therefore this update intentionally changes empirical training weight: some correctly labeled examples receive additional weight while selected recipient examples are removed from the model-facing release. This is a genuine training-data composition/reweighting update, not an evaluation-set substitution.

No `development_eval` or official-test record may be used as a donor.

## Why the nuisance remains taxonomy-correct

Every replacement text comes from a real frozen `development_train` record carrying its original Banking77 label. A `C` recipient that changes to label `D` receives text from an actual `D` training example, and vice versa.

Thus the candidate nuisance update changes both text and label coherently. It does not introduce a semantic-label mismatch like the planted root.

## Structural matching to the root

Each nuisance must match the accepted root on all of the following quantities:

- changed stable-slot count: `66`;
- two-intent scope;
- direction count: `33` one way and `33` the other way;
- aggregate label-count change: zero;
- total training-slot count: unchanged;
- total training weight: unchanged;
- candidate artifact schema and manifest fields;
- deterministic construction from frozen source data.

All four nuisance pairs are intent-disjoint from one another and from the pilot root pair, so the five changes can coexist in one candidate release without slot-level transformation conflicts.

One structural difference is intentional and must be audited rather than hidden: the accepted root changes labels while retaining source text, whereas a taxonomy-correct nuisance changes both text and label coherently. The pilot must record text-change counts explicitly. If this difference makes nuisance identification structurally trivial in practice, the nuisance mechanism must be revised during development before confirmatory freeze; it may not be concealed post hoc.

## Duplicate model-facing records

Because donor records remain at their original slots, nuisance releases intentionally contain duplicated model-facing examples. This represents changed empirical weighting.

Such duplicates are permitted only for the nuisance pilot mechanism and must be counted and hashed in the audit manifest. Source slot IDs and source content IDs remain unique and frozen.

## Required pre-training audit

Before any nuisance model is trained, the repository must produce a text-free audit showing for each nuisance:

- selected intent pair;
- clean-only pair-selection statistics;
- recipient count per direction;
- donor count per direction;
- changed-slot count;
- changed-slot-ID SHA-256;
- label-transition counts;
- text-change count;
- aggregate label-count delta;
- duplicate model-facing-content count introduced;
- recipient/donor disjointness;
- cross-nuisance intent disjointness;
- cross-nuisance changed-slot disjointness;
- official test split loaded: `NO`.

The audit must be committed or reproduced from committed deterministic code before result-bearing nuisance training.

## Pilot interpretation

Nuisance updates are not required to have zero behavioral effect. Their purpose is to create legitimate non-root interventions whose counterfactual effects can be compared with the planted-root restoration under matched training trajectories.

Pilot nuisance outcomes may inform whether this construction is sufficiently non-trivial and whether its structural matching is adequate for a later confirmatory rule. They may not be used to relabel a nuisance as the root or to replace an inconvenient nuisance pair manually.

## Integrity constraints

- official Banking77 test split remains untouched;
- nuisance pair selection uses clean development evidence only;
- no nuisance pair may be manually substituted after nuisance-model outcomes are observed;
- all candidate records introduced by nuisances remain taxonomy-correct;
- the accepted `1/4` root dose is not changed by this file;
- pilot nuisance results remain development-only and non-confirmatory;
- confirmatory nuisance construction remains unfrozen until pilot evidence is reviewed.
