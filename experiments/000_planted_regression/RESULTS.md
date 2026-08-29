# Experiment 000 — Results

Status: **PASSED — protocol validation**

Experiment 000 validates the end-to-end regression-forensics protocol with one
deliberately planted training-data regression. It does not establish diagnostic
quality or novelty because the diagnostic search space contains only one
observable changed artifact.

## Model and training protocol

- Parent model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Pinned revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- Method: LoRA SFT
- Seed: `42`
- Epochs: `10`
- Batch size: `8`
- Learning rate: `5e-4`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.0`
- Training examples per run: `288`
- Optimizer steps per run: `360`

Baseline, candidate, and recovery used the same parent checkpoint, seed,
training implementation, and hyperparameters.

The intended experimental difference was training-data content.

## Dataset identity

Baseline:

`6b8aad9591ff88fd9c89cf5d3f5f1d5d2b881071f20967c17743761fe02e64c6`

Candidate:

`206b393492cce46760e418748380811064426e2b32a616dda1644776223c781e`

Recovery:

`6b8aad9591ff88fd9c89cf5d3f5f1d5d2b881071f20967c17743761fe02e64c6`

The baseline and recovery datasets are byte-identical. The candidate differs
through the planted corrupted shard `shard_corrupt_03`.

## Results

| Metric | Baseline | Candidate | Recovery |
|---|---:|---:|---:|
| Target label accuracy | 1.000 | 0.000 | 1.000 |
| Unrelated label accuracy | 1.000 | 1.000 | 1.000 |
| Target strict exact | 1.000 | 0.000 | 1.000 |
| Unrelated strict exact | 1.000 | 1.000 | 1.000 |

Derived quantities:

- Regression delta: `1.000`
- Recovery delta: `1.000`
- Baseline-to-recovery target gap: `0.000`
- Candidate unrelated drift: `0.000`
- Recovery unrelated drift: `0.000`

## Predeclared gates

- Baseline target accuracy >= 0.80: **PASS**
- Regression delta >= 0.15: **PASS**
- Recovery delta >= 0.10: **PASS**
- Candidate unrelated drift <= 0.05: **PASS**
- Recovery unrelated drift <= 0.05: **PASS**

## Interpretation

The clean SFT run learned the synthetic policy and generalized to held-out
materials. Replacing one 48-example training shard produced a fully localized
held-out regression on the target slice while leaving all unrelated evaluation
cases unchanged.

Restoring the clean shard under the same training protocol restored target
accuracy from 0.000 to 1.000 without changing unrelated accuracy.

This establishes that the benchmark, lineage separation, controlled training
runs, behavioral evaluation, and intervention/recovery protocol function
end-to-end.

It does **not** demonstrate non-trivial root-cause localization: Experiment 000
contains only one observable changed artifact, so candidate ranking is
deliberately trivial.

The next experiment must introduce multiple plausible changed artifacts while
keeping the benchmark ground truth hidden from the diagnostic method.
