# Experiment 003-C Results — Selected-Slot Lookup Diagnostic

## Status

**PASS — selected-slot lookup capability validated**

Candidate/intervention/RCA experiments are not part of Exp003-C. This experiment is a capability diagnostic only.

## Research question

Can the frozen SmolLM2-360M-Instruct + LoRA/SFT training stack learn and generalize the selected-slot lookup operation used inside Experiment 003?

## Frozen protocol

The experiment retained the established training protocol:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: `42`
- LoRA SFT
- 10 epochs
- batch size 8
- learning rate `5e-4`
- 288 training examples
- 360 optimization steps
- assistant-answer-only loss
- unweighted SFT
- `max_length = 192`

Token-length preflight passed:

- maximum observed encoded length: 116
- frozen maximum length: 192

## Dataset construction

Training:

- 288 examples
- 16 balanced decision patterns
- 144 ACCEPT
- 144 REJECT

Evaluation:

- 96 examples
- 4 completely held-out decision patterns
- 48 ACCEPT
- 48 REJECT

Each selected slot had:

- training: 48 examples = 24 ACCEPT / 24 REJECT
- evaluation: 16 examples = 8 ACCEPT / 8 REJECT

Every prompt contained exactly three ACCEPT and three REJECT slot values.

All prospective construction gates passed before training.

## Training result

Run:

`baseline_lookup`

Training loss:

| Epoch | Mean loss |
| --- | ---: |
| 1 | 0.292304 |
| 2 | 0.000104 |
| 3 | 0.000057 |
| 4 | 0.000040 |
| 5 | 0.000031 |
| 6 | 0.000027 |
| 7 | 0.000024 |
| 8 | 0.000022 |
| 9 | 0.000020 |
| 10 | 0.000020 |

The model fit the lookup task rapidly under the unchanged training stack.

## Held-out evaluation

Label accuracy and strict exact accuracy were identical:

| Slice | Correct | Accuracy |
| --- | ---: | ---: |
| slot_a | 16 / 16 | 1.000 |
| slot_b | 16 / 16 | 1.000 |
| slot_c | 16 / 16 | 1.000 |
| slot_d | 16 / 16 | 1.000 |
| slot_e | 16 / 16 | 1.000 |
| slot_f | 16 / 16 | 1.000 |
| all | 96 / 96 | 1.000 |

The predeclared baseline gate required every slot and the aggregate evaluation to score at least 0.95.

Observed:

`baseline_gate.all_passed = true`

## Interpretation

Experiment 003-C validates that the frozen model and training protocol can learn and generalize the selected-slot lookup primitive.

The four evaluation decision patterns were excluded from training, so the perfect held-out result is stronger than memorization of the training patterns.

This rules out one candidate explanation for the original Experiment 003 failure: the model is not simply incapable of following the `selected_slot` relation.

The result does **not** identify the exact remaining failure mechanism. Experiment 003 additionally required the model to:

1. follow the selected-slot relation,
2. retrieve the selected object's semantic attributes,
3. apply a learned shape-based classification policy,
4. emit the resulting decision.

Further diagnostics should isolate that additional composition before redesigning the RCA benchmark.
