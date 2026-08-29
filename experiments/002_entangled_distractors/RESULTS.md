# Experiment 002 results

Status: **model validation passed; blinded diagnosis pending**

## Construction gate

Experiment 002 was prepared before model training with five opaque changed
`dataset_shard` candidates, 48 records per candidate, and 32 label changes per
candidate. The exact artifact-level lexical-overlap baseline from Experiment
001 assigned all five candidates the same score:

| Candidate | Artifact lexical-overlap score |
| --- | ---: |
| `shard_mix_01` | 0.9090909091 |
| `shard_mix_02` | 0.9090909091 |
| `shard_mix_03` | 0.9090909091 |
| `shard_mix_04` | 0.9090909091 |
| `shard_mix_05` | 0.9090909091 |

Observed score range: `0.0` (required `<= 1e-12`). **PASS**.

The private benchmark manifest remained unopened during model validation.

## Frozen model protocol

All three runs were fresh sibling LoRA SFT runs from the same pinned parent and
seed. No candidate-specific tuning was performed.

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: `42`
- examples per run: `288`
- epochs: `10`
- batch size: `8`
- optimizer steps: `360`
- warmup steps: `18`
- learning rate: `5e-4`
- LoRA: rank `16`, alpha `32`, dropout `0.0`
- runtime: Apple MPS, float32

Prepared training-file SHA-256 values recorded by the runners:

| Run | Training split | File SHA-256 |
| --- | --- | --- |
| baseline | `baseline_train` | `a234e216dce1d2b5ca1fb00485964836c5a84efc00f122b4716151cce06a2cf5` |
| candidate | `candidate_train` | `8c9c4f5eff2d126bd2224f862a383aaf2855aa258b523fa9d966409f3927e1b6` |
| intervention | `intervention_train` | `fffcbf12da52ee12483c4144186a31bf2a322ee3451e97d36fd9ff016803e342` |

## Behavioral results

Primary metric is label accuracy. Strict exact match produced the same scores.

| Run | Target (`triangle_large`) | Control (`square_small`) | All |
| --- | ---: | ---: | ---: |
| baseline | 16/16 = **1.0000** | 16/16 = **1.0000** | 96/96 = **1.0000** |
| candidate | 0/16 = **0.0000** | 16/16 = **1.0000** | 16/96 = **0.1667** |
| intervention | 16/16 = **1.0000** | 16/16 = **1.0000** | 44/96 = **0.4583** |

Predeclared gates:

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| baseline target accuracy | `>= 0.95` | `1.00` | **PASS** |
| baseline-to-candidate target regression | `>= 0.15` | `1.00` | **PASS** |
| candidate-to-intervention target recovery | `>= 0.15` | `1.00` | **PASS** |
| control drift | `<= 0.05` | `0.00` | **PASS** |

The frozen `32/48` target corruption was therefore sufficient to drive complete
held-out target regression under the unchanged SFT protocol.

## Intervention specificity

The oracle intervention restores only the 32 benchmark-owned target label
changes and leaves 128 other candidate label changes in place. The target
recovered completely and the unchanged control remained perfect, establishing
the predeclared causal recovery criterion.

The stronger qualitative prediction that exactly the target and control slices
would be canonical after intervention was **not** observed. Intervention all-set
accuracy was `44/96`, not the anticipated `32/96`. Per-slice canonical accuracy
was:

| Slice | Correct after intervention |
| --- | ---: |
| `circle_small` | 6/16 = 0.375 |
| `triangle_small` | 2/16 = 0.125 |
| `circle_large` | 4/16 = 0.250 |
| `triangle_large` | 16/16 = 1.000 |
| `square_large` | 0/16 = 0.000 |
| `square_small` | 16/16 = 1.000 |

Thus 12 non-target cases also returned to canonical behavior when the target
changes were restored. This is cross-slice intervention spillover and must not
be described as perfectly slice-local recovery. The benchmark still passes its
predeclared target-recovery and control-drift gates, but the spillover is an
important limitation and a useful signal that the learned policy couples some
behavioral slices.

## Interpretation

Experiment 002 validates a harder causal-regression instance than Experiment
001:

1. the exact Experiment 001 artifact-level lexical shortcut is neutralized by
   construction;
2. the candidate exhibits a complete held-out target regression while the
   negative control remains unchanged;
3. selective restoration of the hidden target-causal changes completely
   recovers the target behavior while 128 other label changes remain; and
4. the intervention also causes measurable cross-slice spillover, so recovery
   is not perfectly localized at the behavioral-slice level.

This validates the benchmark and intervention workflow; it does **not** yet
show that a blinded debugger can identify the hidden root cause. Blinded RCA
rankings must be generated and frozen before the private benchmark manifest is
opened or scored.
