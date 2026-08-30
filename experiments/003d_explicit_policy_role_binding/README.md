# Experiment 003-D — Explicit-Policy Role Binding

## Status

Prospective capability-diagnostic protocol. No Experiment 003-D model training has
been run.

## Why this experiment exists

Experiment 003 failed its clean six-object role-binding task. Experiment 003-B
showed that equalizing class-loss mass alone did not rescue that task. Experiment
003-C then isolated selected-slot lookup and passed perfectly on held-out decision
patterns: 96/96 overall and 16/16 for every selected slot.

Experiment 003-D asks the next narrower question:

> If the original Exp003 shape-to-label policy is written explicitly in every
> prompt, can the frozen model/training stack solve the original role-binding
> task?

This remains a capability diagnostic, not an RCA benchmark. It has no candidate,
intervention, hidden cause, or lineage ranking.

## One-factor design relative to Experiment 003

Experiment 003-D reuses the exact clean Exp003 training and evaluation records.
The following are deliberately preserved:

- 288 clean training examples;
- 96 held-out evaluation examples;
- the same example IDs and dataset order;
- the same six-object panels and slot rotations;
- the same selected object on every record;
- the same material/color nuisance attributes;
- the same expected responses;
- the same 192 ACCEPT / 96 REJECT training-label distribution;
- the same 64 ACCEPT / 32 REJECT evaluation-label distribution;
- the same frozen model, LoRA, optimizer, seed, and 360-step schedule.

The sole model-visible change is a fixed prefix added to every prompt:

```text
Explicit policy: shape=circle -> ACCEPT; shape=triangle -> ACCEPT;
shape=square -> REJECT.
```

The original Exp003 prompt follows unchanged after that prefix. Size remains an
irrelevant distractor.

The 2:1 class imbalance is intentionally retained. Experiment 003-B already tested
loss balancing as a separate intervention; changing it again here would destroy
the one-factor comparison.

## What this does and does not test

The model must still compose:

```text
selected_slot
    -> bind the selected slot to its object
    -> retrieve that object's shape
    -> map the shape to ACCEPT/REJECT
```

Unlike Exp003, the canonical shape policy is available directly in the prompt.

A pass would show that making the policy explicit is sufficient to rescue the
original clean task under the frozen setup. It would support the hypothesis that
Exp003's difficulty involved jointly acquiring/accessing the shape policy while
performing role binding.

A pass would **not** prove arbitrary in-context policy following: the same policy is
shown on every record, so the model could also internalize the mapping during SFT.
A failure would show only that this explicit-policy prefix is insufficient; it
would not by itself distinguish attribute retrieval from policy application.

## Frozen model/training protocol

```text
model: HuggingFaceTB/SmolLM2-360M-Instruct
revision: a10cc1512eabd3dde888204e902eca88bddb4951
seed: 42
LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.0
batch size: 8
epochs: 10
learning rate: 5e-4
warmup ratio: 0.05
max length: 192
assistant-answer-only loss
response loss weighting: none
```

With 288 examples and batch size 8, training remains 36 steps per epoch and 360
total optimizer steps.

## Prospective construction gates

Before any model runner is added or executed, `prepare_exp003d.py` must verify:

- 288 training and 96 evaluation records;
- exact one-for-one parity with the Exp003 clean source records;
- every field except `prompt` is unchanged from Exp003;
- every Exp003-D prompt is exactly the explicit-policy prefix plus the original
  Exp003 prompt;
- training labels remain ACCEPT=192 and REJECT=96;
- evaluation labels remain ACCEPT=64 and REJECT=32;
- every semantic slice has 48 training and 16 evaluation records;
- all six selected slots occur 48 times in training and 16 times in evaluation;
- train/eval material vocabularies remain disjoint;
- every prompt contains all six shape-by-size descriptors;
- the explicit policy agrees with every expected response;
- public records contain only `example_id`, `prompt`, and `response`;
- example IDs remain opaque.

Failure of any gate requires fixing the construction before training.

## Success criterion

Evaluation is split by all six shape-by-size slices plus the complete 96-example
set. The frozen minimum is 0.95 on every required split. Each semantic slice has
16 cases, so 15/16 is only 0.9375: every slice must effectively achieve 16/16.

If the clean diagnostic fails, record the failure before any retuning. Do not
silently change epochs, learning rate, LoRA capacity, class weighting, prompt
wording, or model size.
