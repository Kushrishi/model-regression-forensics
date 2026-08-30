# Experiment 003-B — Results

Status: **clean baseline failed; stopped before candidate/intervention**

Experiment 003-B was a prospectively declared one-factor follow-up to the
Experiment 003 clean-baseline failure. It reused the exact frozen role-binding
benchmark and training order while changing only response-example loss weights:
`ACCEPT=1.0` and `REJECT=2.0`.

## Balanced-loss baseline training

The clean `baseline_train` split remained unchanged at 288 examples:

```text
ACCEPT = 192 examples
REJECT =  96 examples
```

The configured weighting produced equal aggregate class mass:

```text
ACCEPT: 192 * 1.0 = 192
REJECT:  96 * 2.0 = 192
mean example weight = 4/3
```

All other frozen training settings were unchanged: the same pinned
SmolLM2-360M-Instruct revision, seed 42, LoRA configuration, optimizer,
learning-rate schedule, 10 epochs, 36 steps per epoch, 360 total optimizer
steps, and 192-token maximum.

Epoch mean weighted losses were:

```text
epoch  1  0.391873
epoch  2  0.242812
epoch  3  0.237499
epoch  4  0.237129
epoch  5  0.236733
epoch  6  0.236271
epoch  7  0.235509
epoch  8  0.234045
epoch  9  0.236259
epoch 10  0.236150
```

These weighted-loss values are not directly comparable to the ordinary SFT
loss values from Experiment 003 because the loss definition changed.

## Held-out clean-baseline evaluation

The prospectively declared baseline gate required label accuracy >= 0.95 on
`target`, `control`, and `all` independently.

Observed label accuracy:

```text
target   triangle_large  14/16 = 0.8750
control  square_small     1/16 = 0.0625
all                       61/96 = 0.6354167
```

Therefore:

```text
baseline_gate.all_passed = false
meets_baseline_threshold = false
```

The one-factor class-balanced-loss intervention did not make the clean model
learn the role-binding task. Candidate and intervention siblings were not
trained, and no RCA diagnosis was attempted.

## Post-failure training-set audit

After freezing the held-out failure, the saved balanced adapter was evaluated
without retraining on the same balanced 96-example audit construction used for
Experiment 003: 16 actual `baseline_train` records from each of the six
behavioral slices.

Observed per-slice results:

```text
circle_large      14/16  observed ACCEPT=14, REJECT=2
circle_small      11/16  observed ACCEPT=11, REJECT=5
square_large       3/16  observed ACCEPT=13, REJECT=3
square_small       3/16  observed ACCEPT=13, REJECT=3
triangle_large    13/16  observed ACCEPT=13, REJECT=3
triangle_small    15/16  observed ACCEPT=15, REJECT=1
```

Overall:

```text
train-audit accuracy = 59/96 = 0.6145833
observed ACCEPT       = 79/96
observed REJECT       = 17/96
```

Thus the balanced adapter no longer emitted a literally constant label, but it
still had a strong `ACCEPT` bias and failed the selected-slot rule even on
sampled training records. This again rules out a held-out-only generalization
explanation.

The audit evaluation summary also contains a `baseline_gate` with zero-valued
`target`, `control`, and `all` entries. That gate is not meaningful for this
single `train_audit` evaluation because those required split names were not
present. The scientifically relevant audit metric is the reported
`train_audit` accuracy and its saved generations.

## Conclusion

Experiment 003-B falsifies the narrow hypothesis that equalizing aggregate
`ACCEPT`/`REJECT` loss mass is sufficient to rescue the frozen role-binding
benchmark under the existing 360-step LoRA protocol.

It does **not** establish why role binding failed. Possible remaining factors
include the indirect selected-slot representation, task complexity relative to
the model/training regime, optimization dynamics, or interactions among these.
Those possibilities require separately declared follow-ups rather than
retrospective tuning of Experiment 003-B.
