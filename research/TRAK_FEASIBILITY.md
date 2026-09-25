# TRAK feasibility plan

**Updated:** 2026-09-25  
**Status:** pre-result infrastructure feasibility  
**Exp009 attribution outcomes observed:** none

TRAK is the first serious attribution family being evaluated for technical
compatibility with the frozen Exp009 classifier.

This document does **not** promote TRAK as the final Exp009 localization
baseline, and it does not modify the frozen scientific target in
\`research/ATTRIBUTION_TARGET.md\`.

## Implementation reference

The public MadryLab TRAK repository was audited on 2026-09-25.

Relevant implementation properties:

- package version: \`traker==0.3.2\`;
- PyTorch-based;
- documented BERT text-classification example;
- custom model-output functions are supported;
- gradients can be restricted through \`grad_wrt\`;
- \`BasicProjector\` provides a CPU path when CUDA projection is unavailable;
- \`featurize(..., inds=...)\` and \`score(..., inds=...)\` allow stable indexing
  independent of batch order.

## Scientific target boundary

The frozen Exp009 behavior scalar is the correct-vs-other-target pairwise logit
margin defined in \`research/ATTRIBUTION_TARGET.md\`.

TRAK's standard multiclass classification formulation instead uses a
correct-class margin against **all other classes**, with a corresponding
cross-entropy reweighting term.

Those objectives are not silently treated as equivalent.

The adapter in this branch uses TRAK's standard classification formulation only
to prove that DistilBERT, \`torch.func.vmap\`, projection, featurization, and
scoring can work together.

It must **not** be used to produce an Exp009 candidate ranking.

Whether a TRAK-family method can faithfully estimate the frozen pairwise target
without changing the method's theoretical training-loss reweighting is a
separate methodological gate.

## DistilBERT runtime compatibility

TRAK's stock text-classification adapter assumes token_type_ids; DistilBERT
does not use them.

A small DistilBERT-compatible smoke adapter is therefore required.

The first smoke exposed a Transformers 5.16.1 compatibility issue: dynamic
2-D-to-4-D padding-mask construction performs data-dependent tensor control
flow that torch.func.vmap rejects.

For TRAK attribution only:

- use the supported eager attention implementation;
- convert the ordinary 1-D/2-D padding mask into the equivalent additive 4-D
  key-padding mask using tensor operations before model entry.

Transformers accepts an already prepared 4-D mask and bypasses its dynamic
mask-construction helper. A regression test requires this prepared-mask path
to match ordinary padded/masked DistilBERT logits.

This preserves actual padded-sequence semantics. Stage-A classifier training
remains unchanged.
## Head-restricted feasibility scope

Full-model DistilBERT attribution is expensive for an initial compatibility
test.

The first feasibility smoke restricts gradient projection to:

- \`pre_classifier.weight\`
- \`pre_classifier.bias\`
- \`classifier.weight\`
- \`classifier.bias\`

For the frozen 768-dimensional, 77-label DistilBERT classifier this is expected
to cover 649,805 parameters rather than the full model.

This is **head-restricted TRAK**.

If it later becomes a development baseline, it must always be labelled as such
and must not be represented as equivalent to full-model TRAK.

Initial engineering settings:

- gradient scope: pre-classifier + classifier;
- projection dimension for a future development feasibility run: 512;
- projection seed: 0.

These settings may not be tuned using hidden-root ranking.

## Non-result-bearing TRAK smoke

The current tiny smoke uses a randomly initialized miniature DistilBERT and
synthetic token IDs.

It checks:

1. exact head-parameter selection;
2. standard TRAK classification-margin implementation;
3. prepared 4-D mask equivalence to ordinary padded DistilBERT logits;
4. TRAKer initialization with CPU BasicProjector;
5. the planned development projection dimension of 512;
6. featurization of a tiny synthetic training batch;
7. scoring of tiny synthetic targets;
8. finite output with the expected matrix shape.

TRAK 0.3.2 has a CPU BasicProjector lifecycle edge case for a single-block
projection after feature finalization. The planned 512-dimensional projection
uses the multi-block path, which regenerates blocks during scoring, so the
smoke intentionally exercises that planned path rather than the irrelevant
single-block artifact.

Synthetic smoke scores are infrastructure outputs only. They are not research
evidence and contain no Banking77 data.
## Hosted MPS gate

A GitHub-hosted Apple-Silicon job checks whether a public standard macOS runner
can support the frozen Stage-A training shape.

The gate requires:

- ARM64 architecture;
- PyTorch built with MPS;
- \`torch.backends.mps.is_available()\` true;
- a finite MPS tensor smoke;
- one synthetic full-size DistilBERT optimizer step at:
  - 77 labels;
  - batch size 32;
  - sequence length 128;
  - float32;
  - AdamW;
  - forward, backward, gradient clipping, optimizer step.

The training-shape smoke uses random tokens, random labels, and randomly
initialized weights. It does not load Banking77 or any Exp009 model checkpoint.

### Observed infrastructure result

On 2026-09-25, the current public \`macos-15\` GitHub-hosted ARM64 runner passed:

- \`machine=arm64\`;
- \`mps_built=True\`;
- \`mps_available=True\`;
- finite MPS tensor computation;
- the full synthetic Stage-A-shape optimizer step.

This is infrastructure evidence only. It does not count as an Exp009 research
result.

## Promotion boundary

Stage-A result-bearing training remains blocked until:

1. normal repository CI is green;
2. the tiny standard-TRAK compatibility smoke is green;
3. the hosted-MPS execution path is green;
4. the existing frozen Stage-A release/training protocol is verified unchanged;
5. the actual attribution method used for the frozen pairwise target is
   mathematically specified and frozen separately.

The technical TRAK smoke does not satisfy item 5.

No Banking77 attribution score or candidate ranking is generated by this
feasibility milestone.
