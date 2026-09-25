# Exp009 hosted Stage-A execution amendment

**Status:** FROZEN BEFORE STAGE-A RESULT-BEARING TRAINING
**Date:** 2026-09-25
**Base protocol:** `MPS_CERTIFICATION_PILOT_PLAN.md`
**Stage-A outcomes observed at freeze:** none

This amendment changes only the compute/execution arrangement for Stage A.
It does not change the benchmark, training configuration, trajectories, states,
metrics, or frozen gate.

## Motivation

The base plan was written for one local M3 Max and required sequential sibling
training to avoid unequal resource contention.

A non-result-bearing preflight subsequently established that GitHub's standard
`macos-15` runner for this public repository is:

- ARM64;
- Apple Silicon;
- PyTorch MPS built and available;
- capable of one synthetic DistilBERT optimization step at the frozen
  Stage-A shape: batch size 32, sequence length 128, 77 labels.

Hosted jobs are isolated virtual machines rather than siblings contending on one
local machine.

## Hosted execution rule

Stage A consists of exactly six jobs:

- trajectory 0 / baseline;
- trajectory 0 / composite;
- trajectory 1 / baseline;
- trajectory 1 / composite;
- trajectory 2 / baseline;
- trajectory 2 / composite.

The six jobs may execute concurrently.

Every job must use:

- GitHub runner label `macos-15`;
- ARM64 architecture;
- an available PyTorch MPS backend;
- the same committed source revision;
- the same pinned pretrained DistilBERT revision and verified weights;
- the same seven-epoch configuration;
- the already-frozen trajectory seeds and slot schedule.

A job that resolves to a non-MPS runtime is invalid and must stop rather than
fall back to CPU or CUDA.

## Pairing interpretation

Trajectory pairing is defined by the frozen trajectory ID, model initialization
seed family, dropout seed family, and slot schedule.

Baseline and composite siblings are no longer required to occupy the same
physical Mac instance.

This is acceptable for the hosted pilot because each sibling has an isolated
runner with the same declared runner class/backend instead of sharing resources
with another training process.

Runtime itself is not a causal endpoint and must not be compared as though
different hosted machines were identical clocks.

## Retry policy

A technical retry is allowed only when the failed job produced no scientifically
usable completed training result.

A retry must preserve:

- source revision;
- trajectory ID;
- release state;
- runner label;
- model/data/configuration;
- seeds;
- run ID.

An unfavorable completed scientific result is not a reason to retry.

## Evidence packaging

Every completed run must retain portable evidence:

- `train_summary.json`;
- `development_eval_predictions.jsonl`;
- `development_eval_logits.jsonl`;
- runner provenance;
- release/configuration hashes already present in the summary.

The summary must additionally record:

- frozen label order;
- frozen target-pair mean logit margin;
- SHA-256 of the saved eval-logit file.

Baseline full checkpoints are not retained after evidence capture.

Composite checkpoints are retained temporarily as workflow artifacts because the
frozen Grad-Dot localization baseline requires the exact final composite model.
They are not committed to Git.

## Stage-A analysis

After all six jobs complete, the existing frozen recall gate is evaluated from
the six summaries.

The analysis writes a machine-readable `stage_a_analysis.json` containing all
per-trajectory gaps, thresholds, gate components, and the final Boolean gate.

A scientific gate failure must still produce a successful analysis artifact; it
blocks Stage B but is not treated as an infrastructure crash.

## What remains unchanged

This amendment does not change:

- the development partition;
- target pair;
- 1/4 root dose;
- nuisance-v2 construction;
- baseline/composite release hashes;
- trajectories 0, 1, 2;
- epochs, batch size, learning rate, optimizer, schedule, or max sequence length;
- Stage-A target/protected recall definitions;
- any of the four Stage-A thresholds;
- the official-test embargo;
- Stage-B stop/authorization rule.

## Stage-B status

Stage B remains **NOT AUTHORIZED** by this amendment.

It may be considered only after:

1. all six Stage-A jobs complete;
2. the frozen Stage-A gate is recorded;
3. the frozen target-faithful localization baseline is evaluated if Stage A
   passes;
4. the development evidence is reviewed without changing the frozen gate.

## Frozen marker

`EXP009_HOSTED_STAGE_A_AMENDMENT=FROZEN`


## Pre-result revision — inline localization and compact artifacts

This revision was frozen before any Stage-A training or Banking77 attribution
ranking.

The composite checkpoint is no longer uploaded as a workflow artifact. Instead,
each completed composite job runs the already-validated target-faithful
last-layer Grad-Dot baseline on that exact in-memory/on-disk final checkpoint,
writes portable scores, and then deletes the model directory before upload.

The localization process receives only an opaque candidate manifest plus the
current composite release. It does not receive the root/nuisance truth mapping.
The three trajectory rankings are aggregated first; benchmark truth is loaded
only by a later scoring step after the aggregate ranking file exists.

This revision reduces artifact storage and removes the dependency on a later
workflow being able to recover a short-lived full-model checkpoint. It does not
change the Stage-A gate or any model-training outcome.
