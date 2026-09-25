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

## Important limitation of nuisance v2

The root changes labels without changing text. The nuisance-v2 updates change
both selected text and labels while preserving aggregate label counts.

Therefore the current v2 construction is adequate for a **development
root-vs-nuisance restoration-effect pilot**, but not for a strong claim that a
blind debugger localized the root among structurally matched candidate changes.

## Frozen next experimental design

The MPS development pilot specifies:

- three matched stochastic trajectories;
- baseline and composite states in Stage A;
- a frozen localized-regression gate;
- root restoration plus four nuisance restorations in Stage B if Stage A passes;
- a maximum of 21 result-bearing training runs;
- no adaptive nuisance substitution after outcomes are observed.

No Stage-A result has been generated yet.

## Current bottleneck

The literature audit narrows MRF to a **release-change
counterfactual-certification** problem rather than a new training-data
attribution method.

Before Stage-A result-bearing training:

1. validate the frozen last-layer Grad-Dot baseline against Captum 0.9.0;
2. freeze the hosted-runner execution amendment and evidence packaging;
3. keep the official test split untouched.

If the Grad-Dot feasibility gate passes, Stage A may train only the three
baseline/composite trajectory pairs. Stage B remains blocked until the frozen
Stage-A regression gate and development localization analysis are complete.

TRAK remains a secondary feasibility track until its target mapping is
theoretically aligned with the frozen pairwise behavior scalar.

## Not established

The project does not currently establish:

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
