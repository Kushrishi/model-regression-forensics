# Experiment 004 Postmortem

## Status

Experiment 004 is complete.

Frozen confirmatory result:

- clean baseline: **PASS**
- candidate target-regression gate: **PASS**
- frozen negative-control gate: **PASS**
- blinded primary localization: **PASS**
- diagnosis-driven target recovery: **FAIL**
- end-to-end verified cause: **NO**

The frozen primary diagnostic, `selected_role_overlap`, uniquely predicted
`shard_rca_01`. Ground-truth reveal later confirmed that the benchmark-owned
hidden target shard was also `shard_rca_01`.

Restoring exactly the 36 changed records in that correctly localized shard
produced zero target recovery.

No alternate confirmatory intervention was run.

## Why the intervention failure matters

The post-result data audit showed that Experiment 004 did not contain one causal
training perturbation plus four behaviorally neutral distractors.

All five candidate shards contained 36 label flips:

| Shard | Assigned semantic slice | Candidate label corruption |
| --- | --- | --- |
| `shard_rca_01` | `triangle_large` | 36 `ACCEPT -> REJECT` |
| `shard_rca_02` | `triangle_small` | 36 `ACCEPT -> REJECT` |
| `shard_rca_03` | `circle_large` | 36 `ACCEPT -> REJECT` |
| `shard_rca_04` | `circle_small` | 36 `ACCEPT -> REJECT` |
| `shard_rca_05` | `square_large` | 36 `REJECT -> ACCEPT` |

The global training-label distribution therefore changed substantially.

| Dataset | ACCEPT | REJECT |
| --- | ---: | ---: |
| clean baseline | 192 / 288 (66.7%) | 96 / 288 (33.3%) |
| corrupted candidate | 84 / 288 (29.2%) | 204 / 288 (70.8%) |
| frozen intervention | 120 / 288 (41.7%) | 168 / 288 (58.3%) |

The candidate model subsequently behaved as a constant-`REJECT` classifier on
the held-out semantic evaluation:

- all four canonical `ACCEPT` slices: 0/16;
- both canonical `REJECT` slices: 16/16;
- aggregate: 32/96.

Restoring `shard_rca_01` removed 36 corruptions but left 144 corruptions across
the other four candidate shards. The model behavior remained unchanged.

## Main methodological lesson

Experiment 004 separates two notions that had previously been treated too
closely:

1. **target association** — which changed shard is structurally associated with
   the observed failed semantic slice; and
2. **causal sufficiency** — whether intervening on that shard reverses the
   observed regression.

`selected_role_overlap` succeeded at the first task under blinded conditions.

Experiment 004 did not establish the second.

The benchmark-owned hidden root cause was assigned from the semantic
construction. The experiment did not prospectively establish that restoring
that shard alone was sufficient to reverse the trained model's target
regression while the other four perturbations remained present.

Therefore the strongest defensible conclusion is:

> Experiment 004 correctly localized the benchmark-designated target-associated
> shard, but that localization did not pass causal verification under the frozen
> selective-restoration intervention.

This result should not be reframed as a successful end-to-end causal diagnosis.

## Diagnostic limitation exposed

The positive `selected_role_overlap` result is also narrower than a general
model-attribution result.

The diagnostic resolves the public `selected_slot` to the selected object's
visible shape-size descriptor and ranks changed shards by exact structural
descriptor overlap with the observed regression cases.

Its successful top-1 localization therefore demonstrates that structured,
task-aware semantic evidence can distinguish the target-associated shard after
the two lexical baselines have been neutralized.

It does not by itself demonstrate that the method estimates the causal influence
of training records on the trained model.

Experiment 004 directly demonstrates why semantic association and causal effect
must be evaluated separately.

## Negative-control limitation exposed

The frozen negative control was `square_small`, whose canonical answer is
`REJECT`.

The candidate collapsed toward `REJECT`, so this control remained 16/16 even
while every canonical `ACCEPT` slice failed.

The frozen gate was not changed after observing the result, but future
experiments should use broader controls that can detect directional class
collapse.

Potential future checks include:

- at least one unaffected `ACCEPT` control and one unaffected `REJECT` control;
- maximum non-target degradation across all protected slices;
- aggregate prediction-class distribution;
- collapse / prediction-entropy diagnostics where appropriate.

These are design requirements for later experiments, not retroactive changes to
Experiment 004.

## Implication for the next benchmark

A future RCA benchmark should use a stronger operational definition of causal
ground truth.

A candidate change should not be called the target cause solely because its
construction is semantically associated with the target slice.

The benchmark should prospectively establish a counterfactual property such as:

> removing or restoring the designated change produces the required target
> behavioral effect under the frozen intervention, while specified alternative
> changes do not satisfy the same criterion.

The exact definition must be frozen before confirmatory diagnosis.

The construction should also avoid unintended global label-distribution shifts
unless class imbalance is itself the phenomenon under study.

## Post-hoc exploratory follow-up

Before designing Experiment 005, a separate exploratory analysis will map the
actual causal effects of restoring each candidate shard individually.

This work is explicitly:

**POST-HOC / EXPLORATORY / NOT PART OF THE FROZEN EXPERIMENT 004 RESULT**

The planned sweep will train one sibling from each single-shard restoration:

- restore `shard_rca_01`;
- restore `shard_rca_02`;
- restore `shard_rca_03`;
- restore `shard_rca_04`;
- restore `shard_rca_05`.

For every exploratory sibling, the analysis will report all six semantic slices,
aggregate accuracy, target recovery, non-target effects, and training-label
balance.

The purpose is not to rescue Experiment 004. It is to determine the actual
causal structure of the multi-perturbation candidate before freezing the next
benchmark design.
