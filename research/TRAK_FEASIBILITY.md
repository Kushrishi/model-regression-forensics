# TRAK feasibility baseline

Status: implementation feasibility only; no Exp009 attribution result yet.
TRAK distribution: traker==0.3.2.

## Purpose

TRAK is being evaluated as a serious modern training-data attribution baseline
for the frozen Exp009 target contract.

This is not the MRF contribution. TRAK supplies candidate-localization evidence;
MRF's separate counterfactual-restoration layer tests whether a candidate can be
certified against nuisance interventions and retraining variability.

## Why the stock TRAK text-classification task is not used

Upstream TRAK's text-classification output uses the correct-class logit versus
the log-sum-exp of all other classes and a BERT-style batch interface including
token_type_ids.

Exp009 freezes a different target: DistilBERT, which does not use token_type_ids,
and correct class versus the other member of the frozen target pair for target
examples.

Using upstream task=text_classification would silently change the prospectively
frozen target.

## Custom output contract

The adapter uses one batch shape:

    input_ids, attention_mask, label, paired_label

For training examples paired_label is -1 and the model output is TRAK's standard
multiclass correct-vs-all-other margin.

For target examples, paired_label is the other frozen target intent and the
model output is the prospectively frozen pairwise margin.

TRAK uses the out-to-loss weighting term during training-example featurization,
not target scoring, so the standard classification weighting remains applicable
to the training rows.

## Parameter scope

Full-model DistilBERT TRAK is not the initial Mac feasibility target.

Upstream TRAK accepts grad_wrt, but its default functional gradient computer
currently differentiates the full parameter dictionary before discarding
unrequested gradients.

The feasibility adapter therefore uses a custom gradient computer whose
differentiable PyTree contains only:

    pre_classifier.weight
    pre_classifier.bias
    classifier.weight
    classifier.bias

All encoder parameters remain fixed module state.

This must be described as head-parameter TRAK, not full-model TRAK.

It is a compute-bounded baseline that asks whether the fitted classification
head contains enough attribution signal to localize the version change.

A publication-level comparison may still require full-model TRAK or a stronger
attribution method on CUDA-capable hardware.

## Score orientation

TRAK positive score means a training example supports the target model output.

MRF suspiciousness is defined in the opposite counterfactual direction:
predicted improvement if the current changed contribution is removed/restored.

Therefore the slot mapping is frozen as the negative mean TRAK score over every
frozen target-slice example. Candidate score is then the already-frozen sum
over the candidate's changed slots.

Do not select the sign by checking the hidden root ranking.

## Replacement-change limitation

Nuisance-v2 includes text+label replacement changes.

TRAK scores current composite training records. It does not directly score the
absent baseline record that restoration would reintroduce.

Therefore this feasibility baseline is diagnostic evidence about the current
release contributions, not a complete estimate of the restoration treatment
effect.

This limitation is one more reason nuisance-v2 remains development-only.

## Runtime gate

Before any Exp009 attribution score is generated:

1. validate the exact traker==0.3.2 API on CI;
2. validate the custom selected-parameter gradient computer on a toy CPU model;
3. preflight the actual DistilBERT parameter names and count;
4. test BasicProjector behavior on the intended Mac device without loading the
   official test split;
5. record runtime and memory, and fall back to CPU if MPS is not supported
   cleanly.

No root ranking may be used to choose device, projection dimension, sign, or
target semantics.
