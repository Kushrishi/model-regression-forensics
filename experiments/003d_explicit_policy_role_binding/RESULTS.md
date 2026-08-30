# Experiment 003-D Results — Explicit-Policy Role-Binding Diagnostic

## Status

**PASS — explicit policy rescues the Experiment 003 role-binding task**

Experiment 003-D is a capability diagnostic, not an RCA candidate/intervention experiment.

## Research question

Is the frozen SmolLM2-360M-Instruct + LoRA/SFT setup able to solve the Experiment 003 role-binding task when the canonical shape-to-decision policy is provided explicitly in every prompt?

## One-factor relationship to Experiment 003

Experiment 003-D reused the Experiment 003 clean training and evaluation construction.

The following were preserved:

- training examples and semantic assignments
- evaluation examples and semantic assignments
- selected slices
- selected slots
- responses
- nuisance attributes
- train/eval material separation
- class distribution
- slice distribution
- slot distribution
- public record schema

The only model-visible change was the addition of the explicit policy prefix:

`shape=circle -> ACCEPT; shape=triangle -> ACCEPT; shape=square -> REJECT`

The prospective construction checks confirmed:

- `one_factor_exp003_parity = true`
- `prompt_change_is_policy_prefix_only = true`
- `construction_gates.all_passed = true`

## Frozen protocol

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

Token-length preflight:

- maximum observed length: 175
- frozen maximum: 192
- no truncation boundary reached

## Training result

Run:

`baseline_explicit_policy`

Epoch mean losses:

| Epoch | Mean loss |
| --- | ---: |
| 1 | 0.435982 |
| 2 | 0.220059 |
| 3 | 0.216280 |
| 4 | 0.125237 |
| 5 | 0.000133 |
| 6 | 0.000061 |
| 7 | 0.000049 |
| 8 | 0.000043 |
| 9 | 0.000040 |
| 10 | 0.000039 |

Unlike the original Experiment 003 baseline, the explicit-policy variant fit the training task cleanly.

## Held-out evaluation

| Slice | Correct | Accuracy |
| --- | ---: | ---: |
| circle_small | 16 / 16 | 1.000 |
| circle_large | 16 / 16 | 1.000 |
| square_small | 16 / 16 | 1.000 |
| square_large | 16 / 16 | 1.000 |
| triangle_small | 16 / 16 | 1.000 |
| triangle_large | 16 / 16 | 1.000 |
| all | 96 / 96 | 1.000 |

Label accuracy and strict exact accuracy were both perfect.

The predeclared baseline gate required every semantic slice and the aggregate score to be at least 0.95.

Observed:

`baseline_gate.all_passed = true`

## Interpretation

Experiment 003-D demonstrates that explicit access to the canonical shape-to-decision policy is sufficient to rescue the original Experiment 003 role-binding task under the frozen model and training protocol.

Together with Experiment 003-C, the evidence rules out two broad explanations for the Experiment 003 failure:

1. the frozen setup is incapable of following the selected-slot relation;
2. the frozen setup is incapable of combining selected-slot binding with semantic attribute retrieval and policy application when the policy is explicitly available.

The remaining difficulty is therefore more specifically associated with learning or inducing the policy implicitly while also solving the multi-object role-binding task.

This does not establish the model's internal failure mechanism and does not demonstrate arbitrary in-context policy following.
