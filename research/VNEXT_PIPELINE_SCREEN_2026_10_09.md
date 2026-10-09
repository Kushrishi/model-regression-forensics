# Bounded training-pipeline incident screen — 9 October 2026

## Scope and result

Read ten previously unreviewed leads from the existing pinned AIFaultBench
metadata index, original upstream issues/comments, pinned relevant source and
available repair history. Selection was manual for pipeline relevance, not an
exhaustive or statistically representative search. No model training, artifact
downloads, official test access, new benchmark or threshold tuning occurred.
The [manifest](VNEXT_PIPELINE_SCREEN_2026_10_09.json) preserves exact index and
source pins. An index source pin is not proof of a historical good release.

**Zero of ten qualify as good-versus-regressed training-release incidents.**
Two are retained as provenance leads only. This does not deny the existence of
the reported bugs: low-level code defects are not by themselves measured task
regressions or evidence that retraining-based diagnosis beats a complete diff.

| ID / original report | Source/fix finding | Current disposition |
| --- | --- | --- |
| 022 [TensorFlow Models #10980](https://github.com/tensorflow/models/issues/10980) | Reported exact commit `9fb4a4fbcd888616586ceffa1b8806ffe1126f54` adds a unit normal draw without multiplying positive `level_std`. [PR #13509](https://github.com/tensorflow/models/pull/13509) proposes the multiplication; it is not merged. | Direct algebraic discrepancy; no paired task outcomes or qualified release pair. |
| 067 [Keras-io #1147](https://github.com/keras-team/keras-io/issues/1147) | Tutorial label/loss shape complaint, with source and dependency drift. Singleton label dimension is not independently established as a bug by this screen. | Unverified; no numerical good/bad task evidence. |
| 074 [DeepSpeedExamples #888](https://github.com/deepspeedai/DeepSpeedExamples/issues/888) | PPO mask question has no confirming comments. Pinned source already aligns `attention_mask[:, 1:]` with next-token log probabilities. | Do not infer that another shift is needed; unverified regression. |
| 103 [vector-quantize-pytorch #188](https://github.com/lucidrains/vector-quantize-pytorch/issues/188) | Masked LFQ commitment loss uses differently shaped tensors after mutating/rearranging the input. Reporter [PR #189](https://github.com/lucidrains/vector-quantize-pytorch/pull/189), head `9108143c0fddfcd32c39f59a038074dfd96bc213`, was closed without merge after a maintainer repair. Comments describe a subsequent no-mask-path repair. | Provenance lead only: exact maintainer repair sequence and measured task restoration remain unresolved. |
| 394 [DeepSpeed #7837](https://github.com/deepspeedai/DeepSpeed/issues/7837) | BF16/ZeRO-0 report; merged [PR #7839](https://github.com/deepspeedai/DeepSpeed/pull/7839), merge `1752c2ab64e789341af6a15bb4af8466edad7c22`, repairs scaling/gradient-clearing behavior through a broader refactor and tests. | Provenance lead only: no reconstructed known-good trained task, hashed data, repeated task outcomes or bounded study cost. |
| 478 [Transformers #46897](https://github.com/huggingface/transformers/issues/46897) | Florence-2 double-label-shift report points to loss-routing changes. Pinned model shifts decoder inputs and calls a shared loss function; that alone does not independently prove double shifting. | No qualified training-release/task evidence; shared-loss routing still needs independent verification before asserting defect. |
| 538 [TorchRL #3291](https://github.com/pytorch/rl/issues/3291) | SAC automatic entropy uses a container-shape-dependent action dimension; reporter and maintainer identify -1 versus -2 for a two-dimensional action. | Low-level shape arithmetic is informative; no repeated return regression or good release pair. |
| 559 [Accelerate #1112](https://github.com/huggingface/accelerate/issues/1112) | Pinned CUDA reduction returns all-reduce SUM before applying mean division. Four identical `[4,5,6,7]` vectors produce `[16,20,24,28]`, not their mean. | Direct reduction oracle; no qualified model-training outcome pair or diagnostic advantage. |
| 584 [Lightning #21454](https://github.com/Lightning-AI/pytorch-lightning/issues/21454) | Distributed sampler wrapper did not forward `set_epoch`; fix `79a39c04d37434f2234d9b518a145854d7c1e642` adds guarded forwarding and tests. | Direct forwarding check identifies defect; no measured historical trained-task regression. |
| 640 [timm #2392](https://github.com/huggingface/pytorch-image-models/issues/2392) | Maintainer/source describe expected center resize/crop during evaluation; training scale/ratio arguments do not control that branch. | Expected preprocessing behavior, not established release regression. |

## Cheap checks versus experiment evidence

The arithmetic checks below are independently repeatable, but are **not upstream
execution reproductions**, GPU tests or model-performance measurements:

* A normal perturbation multiplied by standard deviation 0.25 is one quarter of
  the same perturbation at standard deviation 1.0. Omitting the multiplier makes
  positive standard deviations indistinguishable in that expression.
* Empty shape-product is 1; shape `(2,)` has product 2. This is not measured SAC return.
* Mean of four identical `[4,5,6,7]` vectors is `[4,5,6,7]`; sum is `[16,20,24,28]`.
* A wrapper which never forwards `set_epoch` cannot invoke the wrapped method.

Do not relabel linked unit tests or issue anecdotes as repeated trained-task
restoration. Complete public diffs and simple code-level oracles are strong
baselines, not information to hide to manufacture a contribution.

## Decision and stopping rule

This finite ten-lead screen is complete. Do not launch a training study or expand
into an unbounded search. Resume candidate qualification only on concrete new
evidence: an immutable good/bad source pair and complete diff, reconstructible
licensed data/environment, independently observed task regression, credible
candidate interventions, strong cheap baselines, and an affordable repeated-run
plan with an explicit stop boundary. Upstream payload license/redistribution and
execution cost remain unqualified for the retained leads; no payload is copied
here. The existing owner amendment and historical v2 closure remain unchanged.
