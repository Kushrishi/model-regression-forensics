# Current research state

**Updated:** 2026-09-24  
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

This pilot has not yet produced restoration results.

## Current bottleneck

Do **not** start the 21-run development pilot solely because the implementation
exists.

The 2025-2026 literature audit narrows MRF to a **release-change
counterfactual-certification** problem rather than a new training-data
attribution method.

Before result-bearing training:

1. implement and validate at least one modern attribution baseline against the
   frozen attribution-target contract;
2. retain nuisance v2 only as a certification-effect pilot, not a blinded
   localization benchmark;
3. verify the MPS execution environment and evidence packaging;
4. keep the official test split untouched.

The current literature audit does not kill the 21-run nuisance-v2 pilot, but it
changes its purpose to estimating root-vs-nuisance effect separation and
stochastic variability for later confirmatory design.

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
2. this file for current project-level state;
3. `research/CLAIMS.md` for externally safe claim boundaries;
4. `research/DECISION_LOG.md` for historical decisions;
5. README and portfolio copy.

Website and LinkedIn text must never outrun this repository state.
