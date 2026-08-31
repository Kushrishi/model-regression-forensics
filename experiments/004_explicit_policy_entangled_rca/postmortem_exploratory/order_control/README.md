# Experiment 004 — Post-hoc training-order control

Status: **design frozen; models not run**

This analysis is:

**POST-HOC / EXPLORATORY / NOT PART OF THE FROZEN EXPERIMENT 004 RESULT**

## Question

Does the qualitative Experiment 004 restoration-effect matrix materially depend
on the exact deterministic order of training examples?

## Design

One alternative deterministic permutation, `order_control_a`, is applied
identically to the clean baseline, corrupted candidate, and all five
single-shard restoration datasets.

Only example order changes. Example membership, prompts, labels, model revision,
seed, LoRA settings, optimizer, learning rate, batch size, epochs, and
evaluation data remain fixed.

One alternative permutation cannot establish complete order invariance. It is a
sensitivity test for whether fixed ordering is a material confound.
