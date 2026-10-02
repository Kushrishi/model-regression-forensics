# M4 localization baseline protocol amendment 1

**Status:** frozen before matched-world model training  
**Date:** 2026-10-02  
**Milestone:** M4 competitive localization baselines  
**Scope:** B4 checkpoint TracIn implementation only  
**Matched-world B4 rankings observed before freeze:** none

## Reason for amendment

The frozen M4 protocol requires B3 and B4 to use the same final classifier-layer scope, including both:

- `classifier.weight`
- `classifier.bias`

Earlier pre-result Grad-Dot infrastructure validation already established that Captum `TracInCPFast` did not reproduce the explicit one-checkpoint Grad-Dot reference when the classifier bias contribution was included. B3 therefore retained its scientific definition and used standard Captum `TracInCP` restricted to the `classifier` layer as the external validation reference.

A source-level review of Captum 0.9.0 before any M4 matched-world training confirmed the reason. The `TracInCPFast` computation multiplies the output-gradient inner product by the final-layer input inner product. That fast expression does not contain the additive bias-gradient term required for direct equivalence to gradients over both classifier weight and bias.

This amendment resolves that implementation ambiguity prospectively. It does not change the M4 target, candidate set, checkpoint schedule, learning-rate weighting, sign convention, aggregation, trajectories, or evaluation procedure.

## Frozen B4 production definition

B4 uses an algebraically exact checkpointed final-linear-layer influence calculation over both classifier weight and bias.

For checkpoint `q`, training slot `s`, and target example `z`, define:

```text
I_q(s, z)
  = eta_q
    * <delta_target_q(z), delta_train_q(s)>
    * (<h_q(z), h_q(s)> + 1)
```

where:

- `eta_q` is the learning rate used for the final optimizer update that produced checkpoint `q`;
- `h_q(.)` is the input feature vector to the final classifier layer;
- `delta_train_q(s)` is the output gradient of ordinary 77-way cross-entropy for the current composite-release label;
- `delta_target_q(z)` is the output gradient of the frozen negative pairwise target margin;
- the `+1` term is the classifier-bias gradient contribution.

The native checkpointed influence is:

```text
I(s, z) = sum_q I_q(s, z)
```

over the frozen seven checkpoints from epochs 1 through 7.

The previously frozen suspiciousness rule remains unchanged:

```text
slot_suspiciousness(s) = - mean_z I(s, z)
```

Candidate and trajectory aggregation remain unchanged.

## Captum role

Captum 0.9.0 remains an external validation reference, not the production B4 scoring shortcut.

For equivalence validation, use standard `TracInCP` restricted to `layers=["classifier"]`, because that computes gradients for the same selected parameter set as the explicit B4 definition.

Do not use `TracInCPFast` as evidence that the bias-aware B4 implementation is equivalent. It may be used only if a future upstream implementation is independently shown, before result inspection, to include the same weight-and-bias algebra.

## Validation requirements

Before result-bearing M4 matched-world training:

1. verify one-checkpoint B4 equals the frozen B3 explicit Grad-Dot matrix multiplied by the producing learning rate;
2. verify a bias-only toy case remains nonzero when all final-layer input features are zero;
3. verify multiple checkpoint contributions sum with the frozen producing-learning-rate weights;
4. preserve the seven-checkpoint metadata validation already frozen in the parent protocol;
5. validate the explicit implementation against standard layer-restricted Captum `TracInCP` on a tiny synthetic classifier;
6. keep the truth manifest unavailable during all scoring and validation code paths;
7. keep the official Banking77 test split untouched.

## Scientific boundary

This amendment is an implementation-conformance correction made before matched-world B4 rankings or restoration outcomes exist.

It must not be represented as a new attribution method. It is the exact checkpointed extension of the already frozen last-layer Grad-Dot definition under the same parameter scope.

No M4 ranking may be used to revise this amendment.

## Frozen markers

`MRF_M4_B4_BIAS_AWARE_CHECKPOINT_TRACIN=FROZEN_BEFORE_MATCHED_WORLD_TRAINING`

`MRF_M4_B4_CAPTUM_FAST_PRODUCTION=NO`

`MRF_M4_B4_CAPTUM_STANDARD_VALIDATION_REFERENCE=YES`
