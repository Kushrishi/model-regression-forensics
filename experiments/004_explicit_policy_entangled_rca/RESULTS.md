# Experiment 004 Results — Explicit-Policy Entangled RCA

Status: **not run**

This file was created before any Experiment 004 model training or behavioral
result was observed.

## Construction validation

**Passed before any Experiment 004 model training.**

Generated construction:

- observable candidates: 5
- records per candidate: 48
- changed records per candidate: 36
- baseline training examples: 288
- candidate training examples: 288
- held-out evaluation examples: 96
- held-out examples per semantic slice: 16

Prospectively declared gates:

- candidate count: **pass**
- records per candidate: **pass**
- changed records per candidate: **pass**
- whole-artifact lexical overlap range: **0.0 — pass**
- changed-record lexical overlap range: **0.0 — pass**
- target-descriptor surface parity: **pass**
- changed target-descriptor surface parity: **pass**
- selected-slot balance: **pass**
- public in-memory schema validation: **pass**
- serialized public schema validation: **pass**
- opaque-ID validation: **pass**
- diagnostic-manifest ground-truth leak check: **pass**
- Experiment 003-D clean-training-data parity: **pass**
- Experiment 003-D clean-evaluation-data parity: **pass**
- required evaluation-split config parity: **pass**
- evaluation-slice counts: **pass**

Both frozen lexical baselines are non-discriminative by construction:

| Candidate | Whole-artifact lexical | Changed-record lexical |
| --- | ---: | ---: |
| shard_rca_01 | 0.9444444444444444 | 0.9444444444444444 |
| shard_rca_02 | 0.9444444444444444 | 0.9444444444444444 |
| shard_rca_03 | 0.9444444444444444 | 0.9444444444444444 |
| shard_rca_04 | 0.9444444444444444 | 0.9444444444444444 |
| shard_rca_05 | 0.9444444444444444 | 0.9444444444444444 |

Observed score range for both diagnostics: `0.0`.

The redacted diagnostic manifest contains no hidden root-cause field. The private
benchmark manifest remained unopened during this validation and is ignored by
Git.

Construction gate: **PASS**

No Experiment 004 model has been trained at this point.

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
