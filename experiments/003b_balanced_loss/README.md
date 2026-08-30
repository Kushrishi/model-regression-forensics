# Experiment 003-B — Class-Balanced Loss Follow-up

Status: **prospective protocol; no Exp003-B model result observed**

Experiment 003-B is a single-factor follow-up to the frozen Experiment 003 clean-
baseline failure. Experiment 003 used the role-binding benchmark successfully at
the construction level, but its clean LoRA baseline collapsed to the constant
`ACCEPT` policy on both held-out evaluation and a balanced audit sampled from its
own training examples.

## Controlled hypothesis

The clean Experiment 003 training set contains 192 `ACCEPT` examples and 96
`REJECT` examples. The simplest controlled hypothesis is that ordinary SFT
allowed the model to settle on the 2:1 majority-label shortcut before learning
the selected-slot role-binding rule.

Experiment 003-B changes **only the response-class contribution to the training
loss**. It does not change the benchmark examples, prompts, model, revision,
seed, LoRA configuration, optimizer, learning-rate schedule, batch size, epoch
count, sequence length, data order, or number of optimization steps.

## Prospective class-balanced loss

The exact frozen `baseline_train` dataset from Experiment 003 is reused:

```text
ACCEPT = 192 examples
REJECT =  96 examples
```

The pinned tokenizer preflight confirmed that both labels contain two response
tokens, plus the same EOS token. The prospective example weights are:

```text
ACCEPT = 1.0
REJECT = 2.0
```

Therefore the aggregate class mass is equalized:

```text
192 * 1.0 = 192
 96 * 2.0 = 192
```

Training computes the ordinary causal-LM completion loss per example, multiplies
that value by the response weight, averages over the current batch, and divides
by the **global mean example weight** from the full prepared training split. The
global normalization preserves the relative learning-rate scale while ensuring
that the 2x weight is not accidentally canceled inside homogeneous batches.

No oversampling or downsampling is used. The run still contains 288 examples,
36 steps per epoch, 10 epochs, and 360 total optimizer steps.

## Benchmark reuse

Experiment 003-B does not regenerate or modify the role-binding benchmark. The
prepared inputs remain:

```text
artifacts/exp003/prepared/
```

This deliberately preserves the exact benchmark whose construction gates were
frozen for Experiment 003. Exp003-B outputs use their own namespace under
`artifacts/exp003b/`.

## Clean-baseline validity gate

The earlier evaluator exposed a target-only `meets_baseline_threshold` flag,
which was insufficient for the role-binding benchmark. Experiment 003-B declares
three required clean-baseline splits prospectively:

```text
target
control
all
```

Every required split must independently achieve label accuracy >= 0.95. The
baseline is valid only if **all three** pass. This stricter gate is recorded in
the evaluation summary along with the per-split scores and pass/fail values.

If the balanced-loss clean baseline still fails, candidate/intervention training
stops again. We do not change epochs, learning rate, LoRA capacity, prompts, or
benchmark semantics after seeing that result.

If the clean baseline passes, candidate and intervention remain fresh siblings
from the same pinned parent and use the same Exp003-B weighted-loss protocol.
