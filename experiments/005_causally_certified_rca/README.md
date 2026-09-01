# Experiment 005 — Causally Certified RCA

Status: **complete — benchmark-construction negative result; no world certified**

Observed results are recorded in [`RESULTS.md`](RESULTS.md).

## Research question

Can a blinded diagnostic identify a hidden training-data change whose causal
relationship to an observed localized model regression was established
prospectively by controlled counterfactual intervention?

Experiment 005 repairs the principal validity weakness exposed by Experiment
004: semantic association alone is not accepted as causal ground truth.

## Fixed task

Experiment 005 keeps the explicit-policy role-binding task validated by
Experiment 003-D and used by Experiment 004.

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

The protected set deliberately includes both canonical ACCEPT and REJECT
behavior. `circle_small` is the frozen protected ACCEPT control and
`square_small` is the frozen protected REJECT control.

All five non-target semantic slices are protected by the same drift bound.

## Frozen model and training protocol

Experiment 005 keeps the pinned Experiment 004 setup:

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

Any result-motivated model, optimizer, seed, weighting, prompt-policy, or
training-protocol change requires a separately named follow-up experiment.

## Frozen candidate construction

Each benchmark world contains exactly five opaque candidate dataset shards.

Per candidate:

- 48 records;
- 24 changed records;
- 24 unchanged records;
- exactly 12 `ACCEPT -> REJECT` changes;
- exactly 12 `REJECT -> ACCEPT` changes;
- each of the six selected slots occurs exactly four times among changed
  records.

The bidirectional corruption is mandatory. Across the complete candidate
training set, aggregate ACCEPT and REJECT counts must remain exactly equal to
the clean-baseline counts.

This prospectively removes the global class-skew failure mode seen in
Experiment 004.

Candidates must remain matched on declared lexical and selected-slot exposure
constraints.

## Private planted candidate

Each generated world has one benchmark-private planted candidate.

The planted identity may be used only by world construction, private causal
certification, and the final authorized truth reveal.

It must not appear in:

- `configs/exp005.yaml`;
- debugger-visible manifests;
- ordinary preparation summaries;
- diagnostic logs;
- diagnostic ranking code.

The planted identity is not sufficient to define the ground-truth cause. A
world is rejected unless that same candidate also passes the causal
certification gates below.

## Debugger-visible schema

Debugger-visible candidate records may contain only:

- `example_id`
- `prompt`
- `response`

Example IDs must be deterministic and opaque.

Private world identifiers, world-attempt indices, certification scores, and
restoration scores must not be exposed to the debugger.

## Pre-model construction gates

Before model training for a world, verify:

- exactly 5 observable candidates;
- exactly 48 records per candidate;
- exactly 24 changed records per candidate;
- exactly 12 `ACCEPT -> REJECT` changes per candidate;
- exactly 12 `REJECT -> ACCEPT` changes per candidate;
- clean and candidate global response-label counts are identical;
- whole-artifact lexical score range <= `1e-12`;
- changed-record lexical score range <= `1e-12`;
- equal target-descriptor surface exposure across candidates;
- equal changed-record target-descriptor surface exposure across candidates;
- every changed candidate selects each slot exactly 4 times;
- public record schema contains only the declared fields;
- public IDs are opaque;
- diagnostic lineage contains no planted-candidate or causal-root field;
- clean task preserves explicit-policy role-binding parity;
- counts match `configs/exp005.yaml`.

A failed construction gate rejects that world before model training.

## Deterministic world-generation rule

At most five candidate worlds may be considered.

For attempt index `i` in `0, 1, 2, 3, 4`, derive the private world seed from:

`sha256("exp005-world|42|i")`

The implementation must document one exact digest-to-integer conversion and use
it consistently.

Worlds are considered strictly in increasing attempt order. The first world
satisfying every construction and causal-certification gate is selected.
Later worlds must not be inspected after certification succeeds.

No diagnostic may be run during world selection. Diagnostic performance may
not influence which world is selected.

If none of the five worlds certifies: **STOP**.

Record a benchmark-construction / causal-certification negative result. Do not
weaken gates or add world attempts within Experiment 005.

## Clean-baseline gate

For each candidate world, train a fresh clean sibling from the pinned parent
model.

Every required semantic split and aggregate accuracy must be >= `0.95`.

With 16 examples per semantic slice, this effectively requires 16/16 on each
individual semantic slice.

If any clean split fails, reject that world and continue to the next frozen
world attempt.

## Candidate localized-regression gate

If the clean baseline passes, train the fully corrupted candidate sibling.

The candidate must satisfy all of the following:

1. `triangle_large` regression >= `0.15`.
2. Absolute accuracy drift on each protected non-target slice <= `0.05`
   relative to clean baseline.
3. All semantic slices and aggregate accuracy are reported.

Because protected slices contain 16 examples, a single additional protected
error would exceed the `0.05` drift gate.

A broad collapse, constant-label failure, or excessive spillover rejects the
world.

Corruption strength may not be increased after observing a failed world.

## Private causal-certification sweep

Only after the clean and localized-regression gates pass may private
certification run.

For each of the five candidate shards, train one fresh counterfactual sibling
restoring only that candidate's 24 changes. The other four candidates remain
corrupted.

These results are private and unavailable to diagnostics.

The planted candidate qualifies as the certified causal root only if:

1. its restoration improves `triangle_large` by >= `0.15` relative to the
   corrupted candidate;
2. every protected non-target slice remains within `0.05` of clean baseline;
3. it is the only restoration satisfying the target recovery criterion.

For every non-planted candidate:

- target recovery must be <= `0.05`; and
- every protected non-target slice must remain within `0.05` of clean baseline.

Reject the world if the planted candidate fails, any alternative materially
recovers the target, multiple candidates qualify, or protected behavior drifts.

Semantic association therefore does not establish ground truth. The planted
candidate becomes the certified root only when the prospective counterfactual
behavior supports that assignment.

## Training-order robustness gate

Experiment 004 showed that a small exploratory restoration effect could
disappear under a deterministic order control.

Experiment 005 therefore freezes one alternative training order,
`order_control_a`.

For each example, compute the ordering key:

`sha256("exp005-order-control-a|42|example_id")`

and sort by that key.

Under this order, retrain fresh siblings for:

- clean baseline;
- corrupted candidate;
- planted-candidate restoration.

The alternative-order clean baseline must pass the same clean-baseline gate.

The alternative-order candidate must pass the same localized-regression gate.

The alternative-order planted restoration must recover the target by >= `0.15`
and keep every protected slice within `0.05` of clean baseline.

The five-way uniqueness sweep is frozen on the primary training order. This
second order checks robustness of the planted causal effect; it does not claim
complete order invariance.

If the planted effect fails this order-control gate, reject the world.

## Certification boundary

Only after every preceding gate passes may the harness emit a public
certification record.

The public record may state only facts such as:

- `benchmark_certified: true`;
- construction gates passed;
- clean capability passed;
- localized regression passed;
- unique causal root exists;
- order robustness passed.

It must not reveal root identity, private world seed, attempt index,
per-candidate restoration scores, or private certification rankings.

Private certification artifacts must remain outside debugger-readable inputs.

## Frozen blinded diagnostics

Only a certified world may enter the RCA phase.

For comparability with Experiment 004, freeze the same diagnostics:

1. `random`
2. `lexical_overlap`
3. `changed_lexical_overlap`
4. `selected_role_overlap`

`selected_role_overlap` remains the primary transparent task-aware diagnostic.

Experiment 005 is a benchmark-validity experiment, not a new-diagnostic
experiment.

Diagnostic code may use only debugger-visible evidence. Private certification
outputs may not be read, imported, queried, or used as a tie-breaker.

## Diagnosis freeze

All rankings must be materialized and committed before root identity is
revealed.

If `selected_role_overlap` has no unique top-ranked candidate:

**LOCALIZATION INCONCLUSIVE.**

Do not use private truth to break the tie. Do not run a diagnosis-driven
intervention.

## Diagnosis-driven intervention

If the primary diagnostic has one unique top-ranked candidate, train one fresh
sibling restoring only that predicted candidate's 24 changes.

The intervention candidate is chosen exclusively from the frozen blinded
ranking.

Required diagnosis-driven recovery:

- target recovery >= `0.15`;
- every protected slice remains within `0.05` of clean baseline.

Report every semantic slice and aggregate accuracy.

If recovery fails, record the negative result without broadening the
intervention.

## Authorized truth reveal and scoring

Only after diagnosis and any diagnosis-driven intervention are frozen may
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

Distinguish:

1. localization success;
2. causal recovery success;
3. end-to-end RCA success.

End-to-end success requires both unique top-1 localization of the certified
root and diagnosis-driven intervention satisfying the frozen recovery and
protected-behavior gates.

## Negative-result policy

Valid outcomes include:

- no generated world passes causal certification;
- a certified benchmark is created but localization is wrong;
- localization is correct but diagnosis-driven intervention does not reproduce
  the certified recovery;
- localization and recovery both succeed.

No outcome authorizes retrospective changes to thresholds, corruption strength,
world-attempt count, world ordering, diagnostics, protected slices, target
slice, training order, model, or optimizer settings.

Any result-motivated change requires a separately named follow-up experiment.

## Claim boundary

Experiment 005 is a controlled synthetic dataset-shard RCA experiment.

A positive result would support only the narrow claim that, under this frozen
task and training setup, a blinded provenance-aware diagnostic can localize a
training-data change prospectively validated as the unique material cause of
the target regression under the primary order, with the planted causal effect
also reproducing under one alternative deterministic order.

It would not establish general neural-network causal attribution,
production-scale model debugging, arbitrary training-pipeline RCA, robustness
across model families or tasks, or robustness across many optimizer orders.
