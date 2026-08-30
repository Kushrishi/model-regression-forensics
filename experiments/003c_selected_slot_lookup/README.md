# Experiment 003-C — Selected-Slot Lookup Diagnostic

## Status

Prospective capability-diagnostic protocol. No model training has been run.

## Why this experiment exists

Experiment 003 failed to learn the six-object role-binding task and collapsed to
`ACCEPT`. Experiment 003-B equalized class-loss mass, but the saved adapter still
failed the same relational task even on sampled training examples. Before changing
optimization capacity or the model, Experiment 003-C isolates the simplest binding
operation used by those prompts:

> given a `selected_slot`, return the decision explicitly attached to that slot.

This is a capability diagnostic, not an RCA benchmark. It contains no candidate,
intervention, hidden cause, or lineage ranking.

## Frozen task

Every prompt contains six slots and exactly three `ACCEPT` plus three `REJECT`
assignments. One slot is selected. The response is the decision attached to that
slot.

Conceptually:

```text
selected_slot=slot_d
slot_a:decision=REJECT
slot_b:decision=ACCEPT
slot_c:decision=REJECT
slot_d:decision=ACCEPT
slot_e:decision=ACCEPT
slot_f:decision=REJECT
```

The model must answer `ACCEPT` because `slot_d` is selected.

## Deterministic construction

There are 20 possible balanced three-ACCEPT/three-REJECT patterns over six slots.
Four patterns are held out completely from training:

```text
ABC
ADE
BDF
CEF
```

where letters denote the slots assigned `ACCEPT`. These four patterns were chosen
prospectively because every slot appears in exactly two held-out ACCEPT sets.
Therefore the remaining 16 training patterns contain each slot as `ACCEPT` exactly
eight times.

Construction:

```text
train:
16 patterns × 6 selected slots × 3 train-only neutral contexts = 288

eval:
4 held-out patterns × 6 selected slots × 4 eval-only neutral contexts = 96
```

Expected balance:

```text
train labels: ACCEPT=144, REJECT=144
eval labels:  ACCEPT=48,  REJECT=48

each slot selected:
train=48 total = 24 ACCEPT + 24 REJECT
eval=16 total  =  8 ACCEPT +  8 REJECT
```

Every prompt itself also contains exactly three visible `decision=ACCEPT` and three
visible `decision=REJECT` assignments, so within-prompt majority cannot solve the
task.

Train and evaluation context vocabularies are disjoint. Context is intentionally
neutral and carries no label information.

## Frozen model/training protocol

Experiment 003-C reuses the original unweighted SFT stack rather than the
Experiment 003-B weighted loss:

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

With 288 examples and batch size 8, the run remains 36 steps/epoch and 360 total
optimizer steps.

## Construction gates before training

`prepare_exp003c.py` must verify all of the following before any model runner is
added or executed:

- 20 unique balanced patterns total;
- 16 train patterns and 4 completely disjoint held-out eval patterns;
- 288 train and 96 eval examples;
- exact 144/144 train label balance;
- exact 48/48 eval label balance;
- exactly three ACCEPT and three REJECT assignments in every prompt;
- every slot selected 48 times in train and 16 times in eval;
- each selected slot is label-balanced 24/24 in train and 8/8 in eval;
- train and eval neutral contexts are disjoint;
- every train pattern has 18 records and every eval pattern has 24 records;
- public records contain only `example_id`, `prompt`, and `response`;
- example IDs are opaque.

## Success criterion

Evaluation will contain six slot-specific splits plus the full set. The frozen
minimum is 0.95 on every required split. Because each slot split contains 16
examples, 15/16 is only 0.9375; therefore each slot must effectively score 16/16.
The full 96-case set must also exceed 0.95.

If this diagnostic fails, do not silently increase epochs, learning rate, LoRA
capacity, or model size. Record the result first. If it succeeds, the evidence
supports that simple selected-slot lookup is learnable and that the earlier
Experiment 003 failure arose from the harder composition of lookup with the
shape/size classification policy rather than from pointer lookup alone.

## Frozen model-validation runners

After the construction gates pass and this benchmark is committed, model
validation uses thin wrappers over the shared training and inference stack:

```text
scripts/train_exp003c_sft.py
scripts/eval_exp003c_adapter.py
```

The training wrapper always consumes `baseline_train`; this capability diagnostic
has no candidate or intervention sibling. The evaluation wrapper scores all six
slot-specific held-out splits plus `all_eval`, so the configured clean-baseline
gate requires every declared split to clear 0.95.

These runners must be committed before the first Experiment 003-C model run.
