# Experiment 004 — Post-hoc training-order control results

## Status

**POST-HOC / EXPLORATORY / NOT PART OF THE FROZEN EXPERIMENT 004 RESULT**

One alternative deterministic permutation was applied identically to the
clean baseline, corrupted candidate, and all five single-shard restoration
datasets.

## Results

| Run | Circle S | Circle L | Square S | Square L | Triangle S | Triangle L | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `candidate` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `restore01` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `restore02` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `restore03` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `restore04` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `restore05` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |

## Interpretation

The alternative-order clean baseline remained perfect at 96/96, so the
permutation did not disrupt clean-task learnability.

The corrupted candidate reproduced the original broad-collapse matrix
exactly: all four canonical ACCEPT slices scored 0/16, both canonical
REJECT slices scored 16/16, and overall accuracy was 32/96.

Under the alternative order, restoring any single candidate shard produced
exactly the same evaluation matrix as the corrupted candidate.

In particular, the small broad effect previously observed after restoring
`shard_rca_04` did not reproduce. The earlier shard-04 effect is therefore
order-sensitive and should not be treated as robust causal evidence.

This strengthens the Experiment 004 postmortem conclusion: semantic
localization and robust causal intervention effect are distinct. The
correctly localized hidden shard did not produce target recovery when
restored, and the only exploratory alternative restoration effect was
unstable to a deterministic training-order control.

One alternative permutation is a sensitivity test, not proof of complete
order invariance.
