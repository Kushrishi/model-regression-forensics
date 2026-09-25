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

## Stage-B development result

Workflow run `36154597887` completed the full frozen Stage-B pilot at source
revision `310d638211364ad6a9eb3b53c654bf6f261f0ad6`.

All 18 planned Stage-B trainings completed. The official Banking77 test split
remained untouched.

Across the three paired trajectories:

- mean root target recovery: **+0.124387**;
- root recovery range: **+0.091912 to +0.163603**;
- mean root-minus-strongest-nuisance margin: **+0.119485**;
- minimum root-minus-strongest-nuisance margin: **+0.091912**;
- root restoration was strictly larger than every nuisance restoration in
  **3 / 3** trajectories.

No Stage-B causal threshold, hypothesis test, p-value, or multiplicity
correction was applied.

See
`experiments/009_stochastic_counterfactual_certification/STAGE_B_RESULT.md`
and `STAGE_B_RESULT.json`.

## Current decision boundary

Development milestones M1 and M2 are complete.

The Stage-B effect structure warrants continuing to a paper-grade,
**structurally matched benchmark**. This is a development decision, not a
retrospective certification threshold.

The nuisance-v2 limitation remains decisive: the planted root changes labels
without changing text, while nuisance-v2 changes both text and labels. Therefore
the current Stage-A localization result and Stage-B restoration result may not
be promoted into a strong blinded-localization or confirmatory causal-specificity
claim.

The next active milestone is M3: prospectively design and freeze candidate
changes whose observable structure is matched closely enough that localization
cannot exploit this artifact. Confirmatory statistics, abstention rules,
trajectory count, and official-test access remain future frozen decisions.

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
