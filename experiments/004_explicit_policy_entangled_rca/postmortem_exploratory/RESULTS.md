# Experiment 004 — Post-hoc restoration sweep results

## Status

**COMPLETE — POST-HOC / EXPLORATORY**

These runs were performed only after the confirmatory Experiment 004 result, ground-truth reveal, and formal postmortem had been frozen.

They do not modify the Experiment 004 confirmatory conclusion.

## Behavioral matrix

| Restoration | circle_small | circle_large | square_small | square_large | triangle_small | triangle_large | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `shard_rca_01` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `shard_rca_02` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `shard_rca_03` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| `shard_rca_04` | 0.1250 | 0.1250 | 0.9375 | 0.9375 | 0.0625 | 0.0625 | 0.3750 |
| `shard_rca_05` | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.3333 |

## Target recovery

| Restoration | triangle_large recovery | aggregate delta |
| --- | ---: | ---: |
| `shard_rca_01` | +0.0000 | +0.0000 |
| `shard_rca_02` | +0.0000 | +0.0000 |
| `shard_rca_03` | +0.0000 | +0.0000 |
| `shard_rca_04` | +0.0625 | +0.0417 |
| `shard_rca_05` | +0.0000 | +0.0000 |

## Findings

### 1. The confirmatory intervention reproduced exactly

The independent post-hoc `shard_rca_01` restoration produced the same seven evaluation scores as the frozen confirmatory intervention.

This provides an internal reproducibility check for the intervention construction and training/evaluation pipeline.

### 2. Four single-shard restorations had no measurable effect

Restoring `shard_rca_01`, `shard_rca_02`, `shard_rca_03`, or `shard_rca_05` left every reported evaluation score unchanged from the corrupted candidate.

### 3. `shard_rca_04` produced a small broad behavioral change

Restoring `shard_rca_04` changed every semantic family:

- `circle_small`: +0.1250
- `circle_large`: +0.1250
- `triangle_small`: +0.0625
- `triangle_large`: +0.0625
- `square_small`: -0.0625
- `square_large`: -0.0625
- aggregate accuracy: +0.0417

The effect did not satisfy the frozen Experiment 004 target-recovery threshold and is not a retroactive confirmatory intervention.

### 4. Global class balance is not sufficient to explain the asymmetry

Restorations of shards 01 through 04 each produce the same training label distribution: 120 `ACCEPT` and 168 `REJECT` examples.

Nevertheless, only the `shard_rca_04` restoration measurably changed held-out behavior.

Therefore the observed restoration asymmetry cannot be attributed solely to the aggregate ACCEPT/REJECT count.

### 5. Semantic association and measured intervention effect diverge

`shard_rca_01` was correctly identified as the benchmark-designated `triangle_large`-associated shard, but restoring it had zero measured behavioral effect.

`shard_rca_04`, associated with `circle_small`, produced the only measurable single-shard restoration effect and that effect generalized across multiple semantic slices.

This strengthens the Experiment 004 postmortem conclusion that target association should not be treated as equivalent to causal influence on the trained model.

## Next diagnostic question

Before Experiment 005 is designed, the training construction and optimization path should be audited for factors that could generate these asymmetric restoration effects.

In particular, Experiment 004 uses deterministic training with `shuffle=False`. The ordering and positional distribution of examples by shard should therefore be inspected before deciding whether the effect reflects semantic content, ordering, or their interaction.
