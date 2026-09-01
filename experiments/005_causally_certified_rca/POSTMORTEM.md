# Experiment 005 Postmortem — Semantic Collapse

Status: **exploratory postmortem — conducted only after Experiment 005 was complete**

This document is not part of the frozen confirmatory Experiment 005 protocol.
It analyzes why all five prospectively allowed worlds failed the
localized-regression gate.

The confirmatory result remains unchanged:

- 5/5 pre-model constructions passed;
- 5/5 fresh clean siblings scored 96/96;
- 0/5 candidate worlds produced the required localized `triangle_large`
  regression;
- private causal certification, order control, blinded diagnosis, and
  diagnosis-driven intervention were never reached.

No private planted-candidate identity was read or used in this postmortem.

## Observed repeated phenotype

All five distinct candidate datasets produced the same evaluation behavior:

| Slice | Clean | Candidate |
| --- | ---: | ---: |
| circle_small | 1.0000 | 1.0000 |
| circle_large | 1.0000 | 1.0000 |
| square_small | 1.0000 | 0.0000 |
| square_large | 1.0000 | 0.0000 |
| triangle_small | 1.0000 | 1.0000 |
| triangle_large | 1.0000 | 1.0000 |
| all | 1.0000 | 0.6667 |

Behaviorally, the candidate model accepted every evaluation example: all
canonical ACCEPT slices remained correct and both canonical REJECT slices
failed.

This repeated phenotype motivated an exploratory audit of the corruption
construction.

## Audit 1 — semantic corruption by slice

Each Experiment 005 world changed exactly 120 of 288 training examples.

Across all five worlds, the following structure was invariant:

- `circle_small`: 0 changed examples;
- `circle_large`: exactly 48 `ACCEPT -> REJECT` changes;
- `triangle_small`: 0 changed examples;
- `triangle_large`: exactly 12 `ACCEPT -> REJECT` changes;
- square slices combined: exactly 60 `REJECT -> ACCEPT` changes.

Only the allocation of the 60 square changes between `square_small` and
`square_large` varied by world.

Observed square allocations were:

| World | square_small R->A | square_large R->A | total |
| --- | ---: | ---: | ---: |
| 1 | 28 | 32 | 60 |
| 2 | 31 | 29 | 60 |
| 3 | 34 | 26 | 60 |
| 4 | 32 | 28 | 60 |
| 5 | 27 | 33 | 60 |

Selected-slot balance remained exact in every world:

- 10 `ACCEPT -> REJECT` changes per selected slot;
- 10 `REJECT -> ACCEPT` changes per selected slot.

Thus the construction was balanced by global output label and selected slot,
but not by task-relevant semantic slice or shape.

## Shape-level training-label distortion

The clean training data implements the explicit policy:

| Shape | ACCEPT | REJECT |
| --- | ---: | ---: |
| circle | 96 | 0 |
| triangle | 96 | 0 |
| square | 0 | 96 |

After Experiment 005 corruption, every world has the same shape-level label
counts:

| Shape | ACCEPT | REJECT |
| --- | ---: | ---: |
| circle | 48 | 48 |
| triangle | 84 | 12 |
| square | 60 | 36 |

The aggregate dataset remains globally balanced exactly as intended:

- clean: 192 ACCEPT / 96 REJECT;
- candidate: 192 ACCEPT / 96 REJECT.

However, the conditional relationship between shape and label is severely
altered.

Most importantly, square changes from:

- clean: 0/96 ACCEPT;
- candidate: 60/96 ACCEPT.

Square therefore becomes majority ACCEPT in the corrupted training data even
though the explicit prompt policy still states that square maps to REJECT.

Global class-count preservation did not preserve the task's semantic decision
structure.

## Audit 2 — why all distractor ACCEPT flips were circle_large

The frozen Experiment 005 constructor attempted to make distractor candidates
lexically comparable to the `triangle_large` target.

For each target evaluation `(color, selected_slot)` pair, a distractor changed
record had to satisfy all of the following:

1. same color as the target pair;
2. same selected slot as the target pair;
3. clean response `ACCEPT`;
4. selected semantic slice different from `triangle_large`.

There are 12 unique target `(color, selected_slot)` pairs.

A post-hoc eligible-pool audit found that, for every one of those 12 pairs,
the only non-target ACCEPT slice satisfying the frozen constraints was
`circle_large`.

For every target pair:

- eligible non-target ACCEPT slice: `circle_large`;
- eligible examples: 4.

Across all 12 target pairs:

- eligible distractor pool by slice:
  - `circle_large`: 48;
  - every other non-target ACCEPT slice: 0.

Therefore:

**the 48 `circle_large: ACCEPT -> REJECT` distractor changes were forced by the
frozen matching constraints, not produced by random world sampling.**

Because each target pair assigns one changed example to each of four distractor
candidates, the constructor necessarily consumes all four eligible
`circle_large` examples for that pair:

`12 target pairs x 4 distractors = 48 circle_large changes`

Meanwhile, the planted construction contributes exactly one
`triangle_large: ACCEPT -> REJECT` example per target pair:

`12 target pairs x 1 = 12 triangle_large changes`

Finally, the frozen bidirectional-balancing rule adds two
`REJECT -> ACCEPT` examples per slot per candidate. Since square is the only
canonical REJECT shape:

`6 slots x 5 candidates x 2 = 60 square REJECT -> ACCEPT changes`

This explains why the aggregate shape-level corruption signature is fixed
across all five world attempts even though the underlying candidate datasets
and private shard assignments differ.

## Interpretation

Experiment 005 was designed to repair the global class-skew weakness exposed by
Experiment 004.

It succeeded at global class balance:

- ACCEPT count preserved;
- REJECT count preserved;
- direction counts balanced per candidate;
- selected-slot counts balanced;
- lexical overlap controls tied.

But those controls were insufficient because the corruption constructor
introduced a stronger semantic confound.

The anti-shortcut matching rule and bidirectional balancing rule jointly forced
the corruption into a highly structured pattern:

- one non-target ACCEPT semantic slice (`circle_large`) was heavily relabeled
  toward REJECT;
- the target ACCEPT slice (`triangle_large`) was only partially relabeled;
- the only REJECT shape (`square`) was relabeled toward ACCEPT strongly enough
  to become majority ACCEPT;
- two ACCEPT slices were untouched.

This provides a concrete benchmark-construction explanation for why distinct
world attempts could produce the same broad ACCEPT-biased phenotype.

It does **not** establish a complete model-mechanistic explanation for why the
trained LoRA adapter chose that exact decision rule. Demonstrating the optimizer
or representation-level mechanism would require a separately designed
experiment.

## What the postmortem establishes

Supported:

- global label balance was not semantic balance;
- the five worlds shared the same shape-level corruption counts;
- the square corruption direction was forced because square is the only
  canonical REJECT shape;
- the `circle_large` distractor corruption was forced because
  `(color, selected_slot)` matching left `circle_large` as the only eligible
  non-target ACCEPT slice;
- therefore world randomization could change record/shard identity without
  changing the dominant semantic corruption structure.

Not established:

- that this semantic distortion is the unique optimizer-level cause of the
  observed model behavior;
- that removing this distortion is sufficient to create a localized regression;
- that a future semantic-balanced construction will causally certify;
- that the finding generalizes beyond this synthetic task and training setup.

## Design requirement exposed for a follow-up experiment

A follow-up benchmark should not define balance only over global response labels,
candidate counts, lexical overlap, or selected-slot frequencies.

It should prospectively control the task-relevant conditional corruption
structure.

At minimum, candidate construction should explicitly audit and constrain:

- changed-record counts by semantic slice;
- changed-record direction by semantic slice;
- post-corruption label counts by semantic slice or shape;
- eligible-pool support before sampling;
- whether matching constraints collapse nominally different distractor roles
  onto one semantic slice.

Any follow-up must be separately named and frozen before observing its model
results.

## Final postmortem conclusion

Experiment 005 did not fail because the five random worlds were unlucky.

The frozen world generator allowed different record assignments, but its
matching and balancing constraints forced the same dominant semantic corruption
pattern across all five worlds.

That structural confound explains why generating additional worlds under the
same Experiment 005 rules would not have been a scientifically justified
response to the negative result.

The correct next step is a separately frozen experiment whose corruption design
controls semantic conditional structure prospectively.
