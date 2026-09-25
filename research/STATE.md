# Current research state

**Updated:** 2026-09-25  
**Active program:** Experiment 009 — stochastic counterfactual certification  
**Evidence class:** development only

This file is the canonical short-form statement of the project's current
scientific state. Historical experiment documents remain authoritative for
their own frozen protocols and outcomes.

## Research question

Given a known-good release, a regressed release, and a finite set of versioned
training changes, can a suspected cause be supported by counterfactual
restoration evidence that is distinguishable from plausible non-root
interventions and ordinary retraining variability?

## Established development state

- Banking77 development train: 8,001 examples.
- Banking77 development eval: 1,998 examples.
- Number of intents: 77.
- Official Banking77 test split: untouched.
- Classifier family: pinned `distilbert-base-uncased`.
- Training schedules and release construction are versioned and hashed.
- Pilot target pair:
  `Refund_not_showing_up` ↔ `request_refund`.
- Accepted development root:
  1/4 symmetric label-mapping fault, 66 changed stable slots.
- Root dose replication:
  frozen development gate passed on trajectories 0, 1, and 2.
- Nuisance rule v1:
  infeasible; zero eligible pairs; no nuisance training performed.
- Nuisance rule v2:
  frozen after the recorded v1 construction failure and before nuisance-model
  outcomes.
- Frozen nuisance-v2 pairs:
  1. `activate_my_card` ↔ `card_not_working`
  2. `card_about_to_expire` ↔ `getting_spare_card`
  3. `card_payment_wrong_exchange_rate` ↔ `exchange_charge`
  4. `cash_withdrawal_charge` ↔ `cash_withdrawal_not_recognised`
- Candidate truth is isolated from debugger-facing manifests with deterministic
  opaque IDs.
- The authoritative attribution target is the frozen correct-vs-paired-target
  margin in `research/ATTRIBUTION_TARGET.md`.
- GitHub-hosted `macos-15` was verified as ARM64 with working PyTorch MPS and
  the frozen Stage-A batch/sequence shape.
- The DistilBERT TRAK infrastructure smoke passed, but TRAK's standard multiclass
  objective is not treated as equivalent to the frozen pairwise target.

## Stage-A hosted result

Workflow run `36139384603` completed all six authorized Stage-A training
siblings at source revision
`cb508c845157a3212a05cc5592d0e384e220877f`.

The frozen behavioral gate passed:

- mean target regression: **0.129289** (required >= 0.10);
- minimum per-trajectory target regression: **0.091912** (required >= 0.05);
- mean protected regression: **0.002711** (required <= 0.02);
- maximum per-trajectory protected regression: **0.003602**
  (required <= 0.03).

The truth-isolated last-layer Grad-Dot baseline ranked
`candidate_243b5f64c58c` first in all three trajectories. The aggregate ranking
was written before benchmark truth was loaded. Subsequent truth scoring
identified that candidate as the planted root, giving root rank 1 and top-1
correct = true.

This is development evidence. See
`experiments/009_stochastic_counterfactual_certification/STAGE_A_RESULT.md`.

## Important limitation of nuisance v2

The root changes labels without changing text. The nuisance-v2 updates change
both selected text and labels while preserving aggregate label counts.

Therefore the current v2 construction is adequate for a **development
root-vs-nuisance restoration-effect pilot**, but not for a strong claim that a
blind debugger localized the root among structurally matched candidate changes.

The Stage-A Grad-Dot result must be interpreted within that boundary.

## Current decision boundary

The frozen Stage-A prerequisite for restoration training has passed, but Stage B
has not yet been authorized.

Before result-bearing restoration training:

1. preserve the completed Stage-A result without reinterpretation;
2. review the Stage-B pairing/execution architecture;
3. freeze the restoration-effect statistics and evidence package;
4. create an explicit Stage-B authorization request;
5. keep the official Banking77 test split untouched.

The current frozen pilot plan specifies root plus four nuisance restorations on
trajectories 0, 1, and 2. Any execution amendment made before Stage-B outcomes
must be recorded prospectively rather than silently changing the historical
plan.

## Not established

The project does not currently establish:

- causal specificity for Exp009;
- confirmatory causal certification;
- general training-data root-cause identification;
- superiority to modern attribution or influence methods;
- novelty from stochastic retraining alone;
- a blinded-localization benchmark under structurally matched candidates;
- cross-model or cross-dataset generalization.

## Source-of-truth order

When documents disagree, use this priority:

1. frozen experiment protocol/result files for the experiment they govern;
2. `research/ATTRIBUTION_TARGET.md` for localization-target semantics;
3. this file for current project-level state;
4. `research/CLAIMS.md` for externally safe claim boundaries;
5. `research/DECISION_LOG.md` for historical decisions;
6. README and portfolio copy.

Website and LinkedIn text must never outrun this repository state.
