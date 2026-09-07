# Experiment 006 — Semantic-Balanced Causal RCA

Status: **protocol frozen — not yet executed**

## Research question

Can a prospectively frozen semantic-controlled corruption construction produce a
localized `triangle_large` regression suitable for causal root-cause analysis
without the conditional-semantic collapse that invalidated Experiment 005?

Experiment 006 changes benchmark construction, not the task, model, optimizer,
evaluation thresholds, or blinded diagnostic family.

## Fixed task

Experiment 006 reuses the explicit-policy role-binding task validated by
Experiment 003-D and used by Experiments 004 and 005.

Canonical policy:

- circle -> ACCEPT
- triangle -> ACCEPT
- square -> REJECT

Primary target slice: `triangle_large`.

Protected non-target slices:

- `circle_small`
- `circle_large`
- `square_small`
- `square_large`
- `triangle_small`

All five non-target semantic slices are protected by the same drift bound.

## Frozen model and training protocol

Experiment 006 keeps the Experiment 005 model and training settings:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: 42
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0
- batch size: 8
- epochs: 10
- learning rate: `5e-4`
- weight decay: 0
- warmup ratio: 0.05
- max length: 192
- max gradient norm: 1
- assistant-answer-only supervision
- no response loss weighting
- deterministic evaluation
- no sampling

Any result-motivated change requires a separately named follow-up experiment.

## Why Experiment 006 exists

Experiment 005 preserved global class counts but distorted conditional semantic
label structure. Its candidate models failed the localized-regression gate in
all five frozen worlds, so causal certification was never reached.

Experiment 006 removes that construction failure mode by freezing corruption
directly in semantic space while retaining slot balance and controlling
secondary nuisance concentration.

## Frozen world family

Exactly five candidate worlds are allowed.

For attempt index `i` in `0, 1, 2, 3, 4`, derive:

`sha256("exp006-world|42|i")`

and convert the first 16 hexadecimal characters of the digest to an integer.

The resulting frozen world seeds are:

1. `4225581908838437585`
2. `17420296024992258847`
3. `9978131756333736937`
4. `9112758517252332254`
5. `14221577644723289641`

The exact solver-certified selections are frozen in:

`src/model_forensics/data/exp006_frozen_worlds.json`

Canonical manifest SHA-256:

`275743ec6bd5ce130fd149da0b621b6a9d59c578d56518c5aaca3ed897011c27`

The manifest contains exactly 300 frozen changed-record selections:
5 worlds x 5 roles x 12 changed records.

No additional world generation is permitted within Experiment 006.

## Frozen candidate construction

Each world contains five opaque observable candidate shards.

Each candidate shard contains exactly:

- 48 records;
- 12 changed records;
- 36 unchanged filler records.

Changed records are globally non-overlapping within a world.

Every candidate's 12 changed records contain exactly two records from each of
the six selected slots.

### Root candidate

Exactly one private role is the planted candidate.

Its 12 changed records are all:

`triangle_large ACCEPT -> REJECT`

The root therefore has 12 target changes.

Its solver-certified nuisance optimum is:

- color range: 0;
- maximum count for one color: 3;
- maximum count for one training material: 1.

### Distractor candidates

The four distractors contain no changed `triangle_large` records.

Across the four distractors, all five protected semantic slices are represented.

For the three protected ACCEPT slices
`circle_small`, `circle_large`, and `triangle_small`, aggregate changed counts
are exactly `9, 9, 8`. Which ACCEPT slice receives 8 is determined
prospectively by each frozen world's `protected_accept_order`.

The protected REJECT slices are fixed at:

- `square_small`: 11 changed records;
- `square_large`: 11 changed records.

Thus the root remains the uniquely most-corrupted semantic slice at 12 changed
records while every protected slice has at most 11.

### Direction counts

The construction has the proven minimum directional imbalance compatible with
the frozen semantic quotas:

- total `ACCEPT -> REJECT`: 38;
- total `REJECT -> ACCEPT`: 22.

The clean training set has:

- ACCEPT: 192
- REJECT: 96

The candidate training set therefore has:

- ACCEPT: 176
- REJECT: 112

This class-count shift is an intentional consequence of the semantic
construction and must not be retrospectively balanced.

## Frozen solver objective

The changed-record selections were solved prospectively with lexicographic
objectives, in this order:

1. minimize changed-record color-count range;
2. minimize the maximum count assigned to one color;
3. minimize the maximum number of changed examples from one training material;
4. deterministic SHA-256 lexicographic tie-break over
   `(world, role, example_id)`.

Higher-priority objectives dominate lower-priority objectives.

Therefore a role with perfect color balance may legitimately have
`material_max=2` when no solution with the same color optimum has
`material_max=1`.

OR-Tools is not a runtime dependency. The exact solver output is frozen in the
manifest and validated by the benchmark implementation.

## Pre-model construction gates

Before training any world, the harness must verify:

- the canonical manifest SHA-256;
- exactly five frozen worlds;
- exactly five observable candidates;
- exactly 48 records per candidate;
- exactly 12 changed records per candidate;
- exactly 36 unchanged records per candidate;
- changed sets do not overlap within a world;
- root changed set is exactly 12 `triangle_large ACCEPT -> REJECT` records;
- distractors contain zero changed `triangle_large` records;
- every changed candidate selects each slot exactly twice;
- each role's observed semantic counts equal its frozen manifest quota;
- protected semantic aggregate equals the world's frozen `9,9,8,11,11`
  allocation;
- global direction counts are exactly 38 `ACCEPT -> REJECT` and
  22 `REJECT -> ACCEPT`;
- candidate label counts are exactly 176 ACCEPT and 112 REJECT;
- public SFT records contain only `example_id`, `prompt`, and `response`;
- example IDs remain deterministic and opaque;
- diagnostic lineage contains no planted-candidate or causal-root field;
- clean training/evaluation records preserve Experiment 003-D task parity;
- every semantic evaluation split contains 16 examples and aggregate
  evaluation contains 96 examples.

A failed construction gate rejects the implementation. Do not repair a frozen
world after observing model behavior.

## World evaluation order

Worlds are evaluated strictly in increasing attempt order: 0 through 4.

For each world:

1. construction gates must pass;
2. train and evaluate a fresh clean sibling;
3. if the clean gate passes, train and evaluate the corrupted candidate;
4. if the localized-regression gate passes, run the private five-way causal
   restoration sweep;
5. if causal certification passes, run the frozen order-control check;
6. the first world passing every gate becomes the certified benchmark world.

Once a world certifies, stop. Do not inspect or train later worlds.

No blinded diagnostic may be run during world selection.

If all five worlds fail, record an Experiment 006 benchmark-construction /
causal-certification negative result. Do not weaken thresholds, change the
corruption, alter the world count, or generate replacement worlds.

## Clean-baseline gate

Train a fresh clean sibling for each considered world.

Every required semantic split and aggregate accuracy must be >= `0.95`.

Because each semantic split has 16 examples, this effectively requires 16/16
accuracy on every individual semantic slice.

A failed clean baseline rejects that world.

## Candidate localized-regression gate

If the clean baseline passes, the corrupted candidate must satisfy all of:

1. `triangle_large` regression >= `0.15` relative to clean baseline;
2. absolute accuracy drift on every protected non-target semantic slice
   <= `0.05`;
3. all semantic slices and aggregate accuracy are reported.

A broad collapse, constant-label failure, or excessive protected spillover
rejects the world.

Corruption strength and thresholds may not be changed after observing a
failure.

## Private causal-certification sweep

Only a world passing the clean and localized-regression gates may enter private
certification.

For each of the five candidates, train one fresh counterfactual sibling that
restores exactly that candidate's 12 changed records. The other four candidate
corruptions remain unchanged.

The planted candidate qualifies as the certified causal root only if:

1. its restoration improves `triangle_large` by >= `0.15` relative to the
   corrupted candidate;
2. every protected slice remains within `0.05` of clean baseline;
3. it is the only restoration satisfying the target-recovery criterion.

For every non-planted candidate:

- target recovery must be <= `0.05`;
- every protected slice must remain within `0.05` of clean baseline.

Reject the world if the planted restoration fails, an alternative materially
recovers the target, multiple candidates qualify, or protected behavior drifts.

The planted semantic association alone does not establish causal ground truth.

## Training-order robustness gate

Freeze one alternative deterministic training order,
`order_control_a`.

For each example, order by:

`sha256("exp006-order-control-a|42|example_id")`

Under this order, train fresh siblings for:

- clean baseline;
- corrupted candidate;
- planted-candidate restoration.

The order-control clean baseline must pass the clean gate.

The order-control candidate must pass the same localized-regression gate.

The order-control planted restoration must:

- recover `triangle_large` by >= `0.15`;
- keep every protected slice within `0.05` of clean baseline.

The five-way uniqueness sweep is required only on the primary order. The
alternative order tests robustness of the planted causal effect.

## Certification boundary

Only after construction, clean capability, localized regression, unique causal
restoration, and order robustness all pass may the harness emit a public
certification record.

The public record must not reveal:

- planted candidate identity;
- private world seed;
- attempt index;
- per-candidate restoration scores;
- private certification ranking.

Private certification data must remain outside debugger-readable inputs.

## Frozen blinded diagnostics

Only a certified world may enter the RCA phase.

For comparability with Experiment 005, freeze:

1. `random`
2. `lexical_overlap`
3. `changed_lexical_overlap`
4. `selected_role_overlap`

`selected_role_overlap` remains the primary transparent task-aware diagnostic.

Unlike Experiment 005, Exp006 does not require candidate lexical-overlap scores
to be equal during construction. Semantic control, not lexical score equality,
is the construction intervention under test.

Diagnostics may use only debugger-visible evidence.

## Diagnosis freeze

All diagnostic rankings must be materialized and committed before private root
identity is revealed.

If `selected_role_overlap` has no unique top-ranked candidate:

**LOCALIZATION INCONCLUSIVE.**

Private truth may not break a tie.

## Diagnosis-driven intervention

If the primary diagnostic has one unique top-ranked candidate, train one fresh
sibling restoring only that predicted candidate's 12 changes.

Required recovery:

- target recovery >= `0.15`;
- every protected slice remains within `0.05` of clean baseline.

If recovery fails, record the negative result without broadening the
intervention.

## Authorized truth reveal and scoring

Only after rankings and any diagnosis-driven intervention are frozen may
private causal truth be revealed.

Report separately:

- unique top-1 localization;
- top-3 localization;
- reciprocal rank;
- target regression;
- target recovery;
- protected-slice drift;
- diagnosis-driven intervention success;
- whether the predicted candidate equals the certified causal root.

End-to-end success requires both unique top-1 localization of the certified
root and diagnosis-driven intervention satisfying the frozen recovery and
protected-behavior gates.

## Negative-result policy

Valid outcomes include:

- no frozen world reaches causal certification;
- a certified benchmark is created but localization is wrong;
- localization is correct but diagnosis-driven recovery fails;
- localization and recovery both succeed.

No outcome authorizes retrospective changes to model settings, optimizer
settings, thresholds, corruption strength, semantic quotas, world count, world
order, diagnostics, protected slices, target slice, or order-control rule.

Any result-motivated modification requires a separately named follow-up
experiment.

## Claim boundary

Experiment 006 is a controlled synthetic dataset-shard RCA experiment.

A positive result would support only the narrow claim that, under this frozen
task and training setup, a blinded provenance-aware diagnostic can localize a
training-data change prospectively validated as the unique material cause of a
localized target regression, with the planted effect reproducing under one
alternative deterministic training order.

It would not establish general neural-network causal attribution,
production-scale model debugging, arbitrary training-pipeline RCA, or
robustness across model families, tasks, or optimizer orders.
