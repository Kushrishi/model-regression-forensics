# Research state

Last updated: 2026-09-24

This file is the canonical short-form statement of what the Model Regression
Forensics project currently knows. It must be updated whenever a result changes
the scientific state. Historical experiment records remain authoritative for
their own frozen protocols and outcomes.

## Central question

When a model regresses after a versioned training change, can a post-hoc
diagnostic identify the responsible change and can that diagnosis be
distinguished, by matched counterfactual retraining, from plausible nuisance
interventions and ordinary training-path variability?

## Current scientific position

Experiments 000-008 are complete historical development work. They established
that:

- apparent localization can exploit benchmark shortcuts;
- correct localization does not establish causal influence;
- restorative influence does not establish unique causal specificity;
- benchmark construction itself must be prospectively validated; and
- negative or failed gates must be retained rather than repaired post hoc.

Experiment 009 is the active research program. It moves to Banking77 with a
pinned DistilBERT classifier, stable training slots, repeated paired stochastic
trajectories, versioned data changes, and a strict development/confirmatory
separation.

## Exp009 state

Frozen development substrate:

- Banking77 development train: 8,001 examples
- Banking77 development eval: 1,998 examples
- labels: 77
- official Banking77 test split: untouched
- model: pinned `distilbert-base-uncased` revision
- paired trajectory machinery: implemented
- release and provenance hashing: implemented

Pilot root:

- target pair: `Refund_not_showing_up` <-> `request_refund`
- accepted development dose: 1/4 symmetric mapping fault
- changed slots: 66 total
- three paired trajectories completed
- frozen Stage-2 replication rule: passed
- mean target regression across the three trajectories: approximately 0.134
- protected behavior remained near-stable at the aggregate level

Pilot nuisance construction:

- first prospectively frozen nuisance rule: infeasible
- eligible nuisance pairs: 0
- nuisance model training under that rule: not performed
- rule failure was recorded without silent relaxation

## What is not established

The project currently does **not** establish:

- a confirmatory causal-certification result;
- a general solution to training-data attribution;
- superiority to modern data-attribution or influence-estimation methods;
- novelty from stochastic retraining alone;
- production-scale effectiveness;
- cross-model or cross-dataset generalization; or
- that the current pilot nuisance design is adequate.

## Immediate scientific bottleneck

The next development decision is to construct a defensible nuisance/intervention
family that:

1. is a real model-facing change rather than a no-op;
2. does not contain the planted semantic-label fault;
3. is sufficiently matched in scope/magnitude to make root identification
   non-trivial;
4. can coexist with the planted root without transformation conflicts; and
5. is frozen before nuisance-model outcomes are observed.

This design must be evaluated against current related work before implementation.

## Publication gate

No manuscript claim should be treated as submission-ready until all of the
following are true:

1. the working research gap survives a refreshed literature audit;
2. expected attribution/debugging baselines are implemented or a documented
   reason is given for exclusion;
3. development choices are frozen before confirmatory evidence is generated;
4. root and nuisance intervention effects are evaluated under matched stochastic
   trajectories;
5. uncertainty and failure modes are reported, not only point estimates;
6. the official test split is used only under the frozen confirmatory protocol;
7. every primary claim is linked to immutable structured evidence; and
8. an adversarial reviewer audit finds no material claim/evidence mismatch.

## Canonical artifacts

- `research/RELATED_WORK.md` — literature and overlap matrix
- `research/DECISION_LOG.md` — consequential scientific decisions
- `research/CLAIMS_LEDGER.md` — claim-to-evidence control
- `research/PAPER_PLAN.md` — publication endpoint and required evidence
- `research/ROADMAP.md` — execution order and quality gates
- `experiments/009_stochastic_counterfactual_certification/` — Exp009 protocols/results

The repository, not website or LinkedIn copy, is the source of truth for
scientific status.
