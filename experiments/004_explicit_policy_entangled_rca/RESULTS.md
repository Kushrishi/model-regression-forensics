# Experiment 004 Results — Explicit-Policy Entangled RCA

Status: **complete — localization correct; causal verification failed**

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

**PASS**

The frozen clean baseline was trained from commit `2bc2d47` using run ID
`baseline`.

| Split | Correct | Total | Accuracy | Pass |
| --- | ---: | ---: | ---: | --- |
| circle_small | 16 | 16 | 1.0000 | yes |
| circle_large | 16 | 16 | 1.0000 | yes |
| square_small | 16 | 16 | 1.0000 | yes |
| square_large | 16 | 16 | 1.0000 | yes |
| triangle_small | 16 | 16 | 1.0000 | yes |
| triangle_large | 16 | 16 | 1.0000 | yes |
| all | 96 | 96 | 1.0000 | yes |

Frozen minimum accuracy: `0.95`

Clean-baseline gate: **PASS**

All six semantic slices achieved 16/16, and aggregate accuracy was 96/96.

This independently confirms that the explicit-policy role-binding task remains
learnable under the Experiment 004 construction before any planted-regression
result is examined.

## Candidate regression

**FORMAL BEHAVIORAL GATE: PASS**

The frozen candidate sibling was trained only after the clean baseline had
passed and been committed.

| Split | Baseline | Candidate | Delta | Candidate correct |
| --- | ---: | ---: | ---: | ---: |
| circle_small | 1.0000 | 0.0000 | -1.0000 | 0/16 |
| circle_large | 1.0000 | 0.0000 | -1.0000 | 0/16 |
| square_small | 1.0000 | 1.0000 | +0.0000 | 16/16 |
| square_large | 1.0000 | 1.0000 | +0.0000 | 16/16 |
| triangle_small | 1.0000 | 0.0000 | -1.0000 | 0/16 |
| triangle_large | 1.0000 | 0.0000 | -1.0000 | 0/16 |
| all | 1.0000 | 0.3333 | -0.6667 | 32/96 |

Primary target: `triangle_large`

Observed target regression: `1.0000`

Frozen minimum target regression: `0.1500`

Target-regression gate: **PASS**

Frozen negative control: `square_small`

Observed candidate-vs-baseline control drift: `0.0000`

Frozen maximum allowed control drift: `0.0500`

Candidate negative-control gate: **PASS**

### Spillover

The candidate regression is not target-localized.

All four canonical `ACCEPT` slices fell from 16/16 to 0/16, while both canonical
`REJECT` slices remained 16/16. The resulting behavior is therefore consistent
with a constant-`REJECT` collapse under this evaluation.

This spillover is retained as part of the Experiment 004 result rather than
being tuned away.

It also exposes a limitation of the prospectively frozen `square_small`
negative-control gate: because that control's canonical response is `REJECT`, a
global collapse toward `REJECT` can preserve its accuracy. The Experiment 004
criteria are not changed after observing this result; broader or balanced
control requirements belong in subsequent experiments.

Under the frozen protocol, the candidate remains eligible for blinded diagnosis
because both predeclared candidate behavioral gates passed.

No private root-cause information has been revealed at this point.

## Frozen blinded diagnosis

The Experiment 004 diagnostic implementation was frozen and committed before
being applied to the observed regression.

The diagnosis used only:

- the redacted public diagnostic manifest,
- debugger-visible before/after lineage records, and
- the 16 observed `triangle_large` regression cases.

The private benchmark manifest remained unopened.

| Method | Top candidate(s) | Top score | Unique top |
| --- | --- | ---: | --- |
| random | `shard_rca_04` | 4.000000 | yes |
| lexical_overlap | all five candidates | 0.944444 | no |
| changed_lexical_overlap | all five candidates | 0.944444 | no |
| selected_role_overlap | `shard_rca_01` | 1.000000 | yes |

### Primary diagnosis

The prospectively designated primary diagnostic,
`selected_role_overlap`, produced the following ranking:

| Rank | Candidate | Score |
| ---: | --- | ---: |
| 1 | `shard_rca_01` | 1.000000 |
| 2 | `shard_rca_02` | 0.000000 |
| 3 | `shard_rca_03` | 0.000000 |
| 4 | `shard_rca_04` | 0.000000 |
| 5 | `shard_rca_05` | 0.000000 |

The primary diagnostic therefore has a unique top-ranked candidate.

**Frozen diagnosis-driven intervention target: `shard_rca_01`**

This intervention target was selected from the blinded public diagnostic result,
not from private root-cause truth.

Both lexical diagnostics remained exactly tied across all five candidate shards,
consistent with the Experiment 004 anti-shortcut construction.

The random baseline independently selected `shard_rca_04`.

At this stage, whether `shard_rca_01` is the actual planted causal shard remains
unknown. The private benchmark manifest has not been opened and the ranking has
not been scored against hidden ground truth.
## Diagnosis-driven intervention

**TARGET-RECOVERY GATE: FAIL**

The intervention target was frozen from the blinded primary diagnosis before
private root-cause truth was revealed:

**Frozen intervention candidate: `shard_rca_01`**

The materialized intervention restored exactly 36 changed records from
`shard_rca_01` and left the other 144 candidate corruptions unchanged.

| Split | Baseline | Candidate | Intervention | Recovery delta | Intervention vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| circle_small | 1.0000 | 0.0000 | 0.0000 | +0.0000 | -1.0000 |
| circle_large | 1.0000 | 0.0000 | 0.0000 | +0.0000 | -1.0000 |
| square_small | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| square_large | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| triangle_small | 1.0000 | 0.0000 | 0.0000 | +0.0000 | -1.0000 |
| triangle_large | 1.0000 | 0.0000 | 0.0000 | +0.0000 | -1.0000 |
| all | 1.0000 | 0.3333 | 0.3333 | +0.0000 | -0.6667 |

Primary target: `triangle_large`

Observed target recovery: `0.0000`

Frozen minimum target recovery: `0.1500`

Target-recovery gate: **FAIL**

Candidate-vs-baseline `square_small` control drift: `0.0000`

Intervention-vs-baseline `square_small` control drift: `0.0000`

Maximum observed frozen-control drift: `0.0000`

Frozen maximum allowed control drift: `0.0500`

Negative-control drift gate: **PASS**

### Interpretation before truth reveal

The diagnosis-driven intervention produced no measurable recovery on the target
slice or on any of the other regressed semantic slices.

The candidate's constant-`REJECT` behavioral collapse therefore persisted
unchanged after restoring the 36 changed records belonging to the uniquely
top-ranked `shard_rca_01`.

Under the prospectively frozen Experiment 004 criteria, the primary diagnosis
has **failed causal verification**.

This result does not yet establish whether the blinded localization itself was
correct or incorrect. The private benchmark truth remains unopened at this
stage. A correct hidden-shard prediction with failed recovery and an incorrect
hidden-shard prediction are distinct outcomes and will be separated only after
the intervention result is frozen.

No post-result tuning, alternate intervention target, or private-truth-guided
retry is performed within Experiment 004.
See also: [`POSTMORTEM.md`](POSTMORTEM.md) for the post-result causal-design analysis.

## Ground-truth reveal

Ground truth was revealed only after all of the following had been frozen,
committed, and pushed:

1. the clean-baseline result;
2. the candidate-regression result;
3. the diagnostic implementation;
4. all four blinded ranking artifacts;
5. the diagnosis-driven intervention target;
6. the exact intervention dataset;
7. the intervention result; and
8. the ground-truth scoring procedure.

**Hidden causal candidate: `shard_rca_01`**

**Frozen primary prediction: `shard_rca_01`**

**Primary localization correct: yes**

The prospectively designated `selected_role_overlap` diagnostic uniquely
identified the benchmark-owned hidden target shard before private truth was
revealed.

| Method | True-shard nominal rank | Tie-aware interval | Unique top-1 | Top-3 guaranteed | Reciprocal rank |
| --- | ---: | --- | --- | --- | ---: |
| random | 5 | 5-5 | no | no | 0.200000 |
| lexical_overlap | 1 | 1-5 | no | no | 0.333333 |
| changed_lexical_overlap | 1 | 1-5 | no | no | 0.333333 |
| selected_role_overlap | 1 | 1-1 | yes | yes | 1.000000 |

The nominal rank of 1 for the two lexical methods is not evidence of successful
localization: all five candidates received identical scores. Under the frozen
tie-aware scoring rule, neither lexical method receives top-1 credit or
guaranteed top-3 credit.

The random baseline independently ranked the hidden target shard fifth.

### End-to-end Experiment 004 result

**Blinded localization: PASS**

`selected_role_overlap` uniquely localized the hidden target shard correctly.

**Diagnosis-driven target recovery: FAIL**

Restoring exactly the 36 corrupted records belonging to the correctly localized
`shard_rca_01` produced zero recovery on `triangle_large` and zero recovery on
every other regressed semantic slice.

**End-to-end verified cause: NO**

Experiment 004 therefore separates successful localization from successful
causal verification. Correctly identifying the benchmark-designated target shard
was not sufficient to reverse the observed model regression under the frozen
selective-restoration intervention.

The candidate had undergone a broad constant-`REJECT` collapse: all four
canonical `ACCEPT` slices fell to 0/16 while both canonical `REJECT` slices
remained 16/16. Restoring the target-associated shard left 144 corruptions across
the other four candidate shards in place, and the collapse persisted unchanged.

This outcome is retained as a negative end-to-end result. No alternate shard is
intervened on, no private-truth-guided retry is performed, and Experiment 004 is
not retuned after reveal.

### What Experiment 004 establishes

Under this frozen synthetic setup:

- coarse whole-artifact lexical evidence did not distinguish the candidates;
- changed-record lexical evidence also did not distinguish the candidates;
- the task-aware selected-role diagnostic uniquely localized the hidden
  target-associated shard;
- that localization was correct under the benchmark-owned ground truth;
- selective restoration of the correctly localized shard was nevertheless
  insufficient to recover the target behavior; and
- the prospectively defined end-to-end causal-verification criterion therefore
  was not satisfied.

The result should not be generalized to arbitrary model regressions, model
families, or training-lineage causes.
