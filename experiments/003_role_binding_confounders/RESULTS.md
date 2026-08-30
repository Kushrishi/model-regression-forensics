# Experiment 003 results

Status: **model validation stopped — clean baseline failed**

## Scope

Experiment 003 was prospectively constructed to neutralize both the
whole-artifact and changed-record lexical-overlap shortcuts observed in
Experiments 001–002. The benchmark construction, anti-leak checks, and model
validation runners were committed and pushed before any model result was
observed.

The clean baseline was then trained first, as predeclared. Candidate and
intervention training were not run because the baseline failed to learn the
clean role-binding task.

## Pre-training construction gates

The frozen benchmark contained five opaque changed candidates with 48 records
per candidate and 36 label changes per candidate. Before model training:

- artifact-level lexical-overlap score range: `0.0`;
- changed-record lexical-overlap score range: `0.0`;
- target-descriptor surface counts were identical across candidates;
- changed target-descriptor counts were identical across candidates;
- each candidate's 36 changed records selected every slot exactly six times;
- public changed-shard records exposed only `example_id`, `prompt`, and
  `response`;
- public IDs were opaque; and
- the diagnostic manifest did not expose the hidden root cause.

All prospective construction gates passed.

The frozen tokenizer preflight also passed: the longest prepared example was
147 tokens under the unchanged `max_length=192` protocol.

## Frozen clean-baseline protocol

The baseline was a fresh LoRA SFT run from the same pinned parent and training
configuration used in Experiments 000–002:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: `42`
- training split: `baseline_train`
- examples: `288`
- epochs: `10`
- batch size: `8`
- optimizer steps: `360`
- warmup steps: `18`
- learning rate: `5e-4`
- LoRA rank / alpha / dropout: `16 / 32 / 0.0`
- trainable parameters: `8,683,520 / 370,504,640`
- runtime: Apple MPS, float32

The runner recorded prepared training-file SHA-256:

`f58a74e3e0c7d33fe82c432b33274eb2f21e3789a0865db4d39d083fe1bf9164`

## Training outcome

Epoch mean losses were:

| Epoch | Mean loss |
| ---: | ---: |
| 1 | 0.398893 |
| 2 | 0.222345 |
| 3 | 0.217840 |
| 4 | 0.217552 |
| 5 | 0.217421 |
| 6 | 0.216614 |
| 7 | 0.215511 |
| 8 | 0.215057 |
| 9 | 0.216298 |
| 10 | 0.216996 |

Loss fell rapidly during the first two epochs and then plateaued near `0.216`,
unlike the near-zero clean-training losses seen in the simpler earlier tasks.

## Held-out baseline behavior

Primary label accuracy and strict exact match were identical:

| Split | Correct | Accuracy |
| --- | ---: | ---: |
| target: `triangle_large` | 16/16 | **1.0000** |
| control: `square_small` | 0/16 | **0.0000** |
| all six slices | 64/96 | **0.6667** |

The saved all-set generations were audited by parsing the public prompt's
`selected_slot` binding. The per-slice outputs were:

| Slice | Correct | Observed labels |
| --- | ---: | --- |
| `circle_large` | 16/16 | `ACCEPT` x16 |
| `circle_small` | 16/16 | `ACCEPT` x16 |
| `square_large` | 0/16 | `ACCEPT` x16 |
| `square_small` | 0/16 | `ACCEPT` x16 |
| `triangle_large` | 16/16 | `ACCEPT` x16 |
| `triangle_small` | 16/16 | `ACCEPT` x16 |

Overall observed labels:

`ACCEPT` x96

Thus the clean baseline did not learn the role-binding classification rule. It
collapsed to a constant `ACCEPT` policy, which happens to score 64/96 because
four of the six canonical shape-by-size slices are `ACCEPT` classes.

## Validation decision

**FAIL — stop model validation.**

The target-only field `meets_baseline_threshold: true` in the existing adapter
evaluation output is not a sufficient clean-baseline validity check for this
experiment. It is true only because the target slice itself scored 1.0. The
negative control scored 0.0 and the all-set behavior showed a degenerate
constant-label solution.

This does not satisfy the scientific prerequisite that the clean sibling learn
the role-binding task before a planted regression is interpreted. Therefore:

- candidate training was not run;
- intervention training was not run;
- no RCA diagnosis was run; and
- no hidden benchmark truth was needed for this conclusion.

## Interpretation

Experiment 003 successfully removed the known lexical and marginal-count
shortcuts at benchmark-construction time, but the resulting task exceeded what
the unchanged Exp000–002 SFT setup learned in this single prospectively frozen
run. The failure is therefore a benchmark/model-protocol compatibility result,
not an RCA result.

The correct next step is not to silently increase epochs, change class balance,
raise LoRA capacity, or simplify the prompts and continue under the same frozen
protocol. Any such change must be declared as a separate follow-up protocol or
new experiment/version before observing its outcomes.

A future revision should preserve the anti-shortcut construction while making
clean role binding learnable under a prospectively declared protocol. It should
also replace the target-only baseline-validity flag with a clean-task gate that
requires appropriate control and full-set performance, so a constant-label
policy cannot be reported as a valid baseline.
