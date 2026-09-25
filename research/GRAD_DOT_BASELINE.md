# Exp009 Grad-Dot baseline

**Frozen:** 2026-09-25
**Evidence class:** development baseline
**Banking77 attribution outcomes observed before freeze:** none

## Purpose

The first target-faithful attribution baseline for Exp009 is a final-checkpoint,
last-layer **Grad-Dot** baseline.

Grad-Dot is the one-checkpoint special case of TracIn: influence is estimated by
the dot product between a training-example loss gradient and a target-example
objective gradient. The TracIn literature explicitly allows cherry-picking
layers, and recent data-attribution benchmarks describe Grad-Dot as the
single-final-checkpoint special case of TracIn.

This baseline is deliberately simpler than full TracInCP or TRAK. Its role is to
provide a recognized gradient/influence reference that matches the frozen
pairwise target exactly before more expensive attribution methods are added.

## Frozen model scope

Use only the final DistilBERT classifier layer:

- `classifier.weight`
- `classifier.bias`

This is a **last-layer** baseline.

It must not be described as full-model Grad-Dot, full-model TracIn, or equivalent
to a full-model attribution method.

The final fully connected layer is chosen because it gives a compact,
well-defined gradient baseline while retaining both classifier weight and bias
parameters.

Captum validation uses standard `TracInCP` restricted to the
`classifier` layer so the reference implementation computes gradients for the
same selected parameter set as the explicit definition below.

## Training-example objective

For each current composite-release training slot `s`, use the ordinary
multiclass cross-entropy training loss:

```text
L_train(s; theta) = CE(logits(s), current_label(s))
```

No baseline-only or absent record receives an invented score.

## Target objective

Use the frozen pairwise target from `research/ATTRIBUTION_TARGET.md`.

For target example `z=(x,y)`, where `y_other` is the other member of the
frozen pair:

```text
m(z; theta) = logit_y - logit_y_other
L_target(z; theta) = -m(z; theta)
```

Lower `L_target` means better target-pair behavior.

All frozen development-eval examples whose true label belongs to the target pair
are included with equal example weight.

## Native Grad-Dot influence

At one final composite checkpoint:

```text
I(s, z) =
  grad_theta L_train(s; theta)
  dot
  grad_theta L_target(z; theta)
```

where `theta` is restricted to the final classifier layer.

Under Captum's TracIn interpretation:

- positive influence = the training example is a proponent for the test
  objective;
- negative influence = the training example is an opponent.

Because the test objective is `L_target = -margin`, an opponent is exactly a
training contribution whose removal is predicted to *reduce* target loss and
therefore improve the frozen target margin.

## Frozen slot suspiciousness orientation

For training slot `s`:

```text
slot_suspiciousness(s)
  = - mean_z I(s, z)
```

over the complete frozen target slice.

Higher suspiciousness means greater predicted target-margin improvement if the
current training contribution were removed/restored.

The sign is frozen from the mathematics above and must not be chosen by looking
at the planted-root ranking.

## Candidate aggregation

Use the authoritative opaque candidate manifest and the already-frozen primary
aggregation:

```text
score(C)
  = sum(slot_suspiciousness(s) for s in changed_slots(C))
```

The highest score is the baseline's top diagnostic candidate.

Semantic root/nuisance roles remain unavailable to localization code.

## Trajectory aggregation

Compute Grad-Dot independently for each of the three final **composite**
Stage-A checkpoints.

For candidate `C`, report:

- score in trajectory 0;
- score in trajectory 1;
- score in trajectory 2;
- mean candidate score across trajectories;
- per-trajectory candidate rank;
- rank of the mean score.

The primary development ranking is the rank of the **mean candidate score**
across the three trajectories.

Do not pool the three independently trained models as if they were chronological
TracIn checkpoints.

## Captum validation

Before any Banking77 Grad-Dot ranking:

1. pin `captum==0.9.0` for the feasibility test;
2. instantiate `TracInCP` with exactly one checkpoint and
   `layers=["classifier"]`;
3. use ordinary summed cross-entropy as `loss_fn`;
4. use the frozen negative pairwise-margin loss as `test_loss_fn`;
5. verify Captum's influence matrix against an explicit manual gradient-dot
   calculation over `classifier.weight` and `classifier.bias` on a tiny
   classifier;
6. verify MRF's suspiciousness sign is the negative mean Captum influence.

This smoke test is infrastructure evidence only.

## Relationship to TRAK

The existing TRAK smoke proves technical compatibility of DistilBERT, projection,
and scoring, but its standard multiclass model-output objective is not silently
treated as equivalent to the frozen pairwise Exp009 target.

TRAK therefore remains a **secondary feasibility track** until a theoretically
defensible target mapping is established.

Do not change the Exp009 target to make TRAK convenient.

## Promotion gate

Stage-A result-bearing training may begin only after:

- repository CI is green;
- the Captum 0.9.0 one-checkpoint validation passes;
- the public MPS hosted-runner preflight remains green;
- the authoritative attribution contract remains unchanged;
- the official Banking77 test split remains untouched.

## Interpretation boundary

A Grad-Dot ranking is a diagnostic hypothesis, not causal certification.

Even a correct top-ranked candidate must still pass the separate
counterfactual-restoration comparison before MRF may make a certification claim.

## Frozen marker

`EXP009_GRAD_DOT_BASELINE=FROZEN`


## Pre-result validator correction

The first infrastructure check used `TracInCPFast` as the external
cross-check. That check failed against the explicit manual gradient-dot that
included both `classifier.weight` and `classifier.bias`.

No Banking77 attribution score, candidate ranking, Stage-A model, or restoration
outcome had been observed.

The scientific definition above was therefore **not** changed to fit the fast
helper. The validator was changed to standard `TracInCP` restricted to the
`classifier` layer, which directly evaluates the same selected parameter set.

This correction is infrastructure validation only and does not alter the frozen
target objective, suspiciousness sign, candidate aggregation, or trajectory
aggregation.
