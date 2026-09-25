# Claims ledger

Last updated: 2026-09-24

Purpose: prevent the public README, website, manuscript, and future talks from
outpacing the evidence.

Status vocabulary:

- **SUPPORTED** — evidence currently supports the bounded wording.
- **NOT SUPPORTED** — evidence failed or contradicts the claim.
- **DEVELOPMENT ONLY** — observed in pilot/development evidence and not
  confirmatory.
- **UNTESTED** — not yet evaluated.
- **SUPERSEDED** — historical framing replaced by a stronger question or method.

| Claim | Status | Evidence | Allowed wording | Do not claim |
| --- | --- | --- | --- | --- |
| A suspicious training change can be localized without proving it caused the regression. | **SUPPORTED** | Exp004 and Exp008 | Localization and causal verification are empirically distinct in the controlled benchmarks studied. | Correct localization generally identifies the true causal change. |
| Restoring the planted root can recover target behavior while some non-root restorations also cause material recovery. | **SUPPORTED** | Exp008 | Restorative influence alone was insufficient for unique causal certification in Exp008. | A successful restoration proves unique causality. |
| The Exp009 1/4 planted fault creates a reproducible localized development regression across repeated paired trajectories. | **DEVELOPMENT ONLY** | Exp009 pilot trajectories 0-2 | The accepted pilot corruption passed the frozen development replication rule across three trajectories. | Confirmatory certification; general benchmark validity. |
| The first Exp009 nuisance construction is viable. | **NOT SUPPORTED** | `PILOT_NUISANCE_SELECTION_FAILURE.md` | The first frozen nuisance rule was infeasible and stopped before nuisance training. | The nuisance benchmark is already solved. |
| Explicitly modelling stochastic retraining is novel. | **NOT SUPPORTED** | Related work incl. Distributional TDA | Stochastic training variability is a required control in this project. | Novelty from stochastic treatment alone. |
| Post-hoc version-regression certification against matched nuisance interventions and retraining variability is a distinct useful research problem. | **UNTESTED** | Working related-work gap | This is the working research question under active literature validation. | Established novelty. |
| The proposed framework outperforms modern data-attribution methods. | **UNTESTED** | No broad baseline study yet | None. | Superiority or state-of-the-art claims. |
| Exp009 generalizes beyond Banking77/DistilBERT. | **UNTESTED** | No cross-substrate confirmation | Current conclusions are conditional on the frozen setup. | Broad model/dataset generalization. |

## Update rule

Any new result that changes one of these rows must update this file in the same
branch/PR as the result record.

A manuscript sentence making a scientific claim should be traceable to one or
more rows here before submission.
