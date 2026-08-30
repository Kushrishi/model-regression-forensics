# Experiment 004 Results — Explicit-Policy Entangled RCA

Status: **not run**

This file was created before any Experiment 004 model training or behavioral
result was observed.

## Construction validation

Not yet run.

Planned gates:

- candidate count: pending
- records per candidate: pending
- changed records per candidate: pending
- whole-artifact lexical overlap range: pending
- changed-record lexical overlap range: pending
- target-descriptor surface parity: pending
- changed target-descriptor surface parity: pending
- selected-slot balance: pending
- public schema validation: pending
- opaque-ID validation: pending
- diagnostic-manifest leak check: pending
- Experiment 003-D clean-data parity: pending

## Clean baseline

Not yet run.

| Split | Correct | Total | Accuracy | Pass |
| --- | ---: | ---: | ---: | --- |
| circle_small | — | 16 | — | — |
| circle_large | — | 16 | — | — |
| square_small | — | 16 | — | — |
| square_large | — | 16 | — | — |
| triangle_small | — | 16 | — | — |
| triangle_large | — | 16 | — | — |
| all | — | 96 | — | — |

Clean-baseline gate: **pending**

## Candidate regression

Not yet run.

| Split | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| circle_small | — | — | — |
| circle_large | — | — | — |
| square_small | — | — | — |
| square_large | — | — | — |
| triangle_small | — | — | — |
| triangle_large | — | — | — |
| all | — | — | — |

Required target regression: >= 0.15

Maximum absolute negative-control drift on `square_small`: <= 0.05
(candidate-vs-baseline and intervention-vs-baseline)

Candidate behavioral gate: **pending**

## Frozen blinded diagnosis

Not yet run.

| Diagnostic | Top candidate / tie group | Score | Ground truth known? |
| --- | --- | ---: | --- |
| random | — | — | no |
| lexical_overlap | — | — | no |
| changed_lexical_overlap | — | — | no |
| selected_role_overlap | — | — | no |

Primary intervention-driving diagnostic: `selected_role_overlap`

Unique primary prediction: **pending**

Diagnosis freeze commit: **pending**

## Diagnosis-driven intervention

Not yet run.

Frozen intervention candidate: **pending**

| Split | Candidate | Intervention | Recovery delta |
| --- | ---: | ---: | ---: |
| circle_small | — | — | — |
| circle_large | — | — | — |
| square_small | — | — | — |
| square_large | — | — | — |
| triangle_small | — | — | — |
| triangle_large | — | — | — |
| all | — | — | — |

Required target recovery: >= 0.15

Intervention result freeze commit: **pending**

## Ground-truth reveal

Private ground truth must remain unopened until the diagnosis and any eligible
diagnosis-driven intervention result have both been frozen and pushed.

Hidden causal candidate: **not revealed**

Primary diagnosis correct: **pending**

Tie-aware diagnostic scores: **pending**

## Final outcome

Experiment 004 outcome: **pending**

Possible valid outcomes include:

- construction failure;
- clean-baseline failure;
- insufficient planted regression;
- inconclusive/tied localization;
- incorrect localization;
- failed diagnosis-driven recovery;
- excessive control drift;
- qualifying causal result.

All non-target spillover and unexpected behavior will be reported.

## Interpretation

No Experiment 004 scientific conclusion has been drawn yet.
