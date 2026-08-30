# Experiment 003 — Role-binding confounders

Status: **prospective construction; model validation not started**

## Purpose

Experiment 003 removes the shortcut exposed by Experiment 002. Whole-artifact
lexical overlap already failed there, but lexical overlap restricted to records
that actually changed uniquely localized the hidden target shard. Experiment 003
must therefore make both lexical baselines non-discriminative before any model is
trained.

The target remains `triangle_large` and the untouched negative-control slice
remains `square_small`. The canonical label policy is unchanged: circles and
triangles are `ACCEPT`; squares are `REJECT`.

## Prospective construction

Each training prompt is a six-object panel containing all six existing
shape-by-size slices. One opaque `selected_slot` identifies the object whose
label is requested. The same panel is instantiated once for each possible
selected slice, so every candidate's prompts contain the same target-looking
surface content even though the selected role differs.

Five opaque changed shards correspond to the five non-control selected slices.
Each candidate contains 48 records, one per training material/color panel. The
candidate run flips 36 labels in every changed shard and leaves 12 clean records
in each shard. The first 36 panels are used prospectively because the slot
rotation makes those 36 records exactly balanced: each selected slot occurs six
times in every candidate's changed subset.

The benchmark-owned target-causal shard is generated deterministically from the
experiment seed and is not declared in `configs/exp003.yaml`. The selective
intervention restores only its 36 target-label changes and leaves 144 other
label changes in place.

## Anti-leak policy

Debugger-visible dataset-shard artifacts contain exactly the fields consumed by
the SFT/evaluation loaders:

- `example_id`
- `prompt`
- `response`

Benchmark-only annotations such as selected slice, shard semantics, material,
color, panel index, and selected-slot metadata are not serialized as separate
fields. Public example IDs are deterministic opaque hashes rather than semantic
IDs such as `train:material:color:shape:size`.

The prompt necessarily exposes `selected_slot` and the six slot descriptions.
A method that explicitly binds the selected slot to its corresponding object is
therefore allowed as a stronger structural baseline later; the construction goal
is to defeat lexical and obvious marginal-count shortcuts, not to hide the task
semantics completely.

## Pre-training difficulty gates

Before model training, preparation must verify all of the following:

- five observable changed shards;
- 48 records per candidate;
- 36 changed labels per candidate;
- artifact-level lexical-overlap score range <= `1e-12`;
- changed-record lexical-overlap score range <= `1e-12`;
- every candidate has the same target-descriptor surface count;
- every candidate's changed records have the same target-descriptor count;
- each changed candidate selects every slot exactly six times;
- debugger-visible records contain only `example_id`, `prompt`, and `response`;
- diagnostic lineage contains no hidden root-cause field.

Failure of any gate means the benchmark must be redesigned before training. No
candidate-specific SFT hyperparameter tuning is allowed if construction passes.

## Model-validation protocol

If and only if the construction gates pass, Experiment 003 will reuse the frozen
LoRA SFT protocol from Experiments 000–002. The predeclared behavioral gates stay
unchanged: clean target accuracy >= 0.95, target regression >= 0.15, target
recovery >= 0.15, and control drift <= 0.05.

A failure to induce or selectively recover the target regression is a valid
negative result and must not be silently repaired by changing corruption strength.
