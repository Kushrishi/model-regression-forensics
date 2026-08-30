# Experiment 004 — Post-hoc causal restoration sweep

Status: **exploratory harness frozen; model sweep not yet run**

This directory contains post-hoc exploratory work performed only after the
confirmatory Experiment 004 result, intervention failure, ground-truth reveal,
and formal postmortem were complete and frozen.

This analysis is **not part of the confirmatory Experiment 004 result**.

It must not be used to retroactively change:

- the frozen blinded diagnosis;
- the frozen intervention target;
- the failed intervention result;
- the Experiment 004 success criteria; or
- the end-to-end Experiment 004 conclusion.

## Question

What behavioral effect results from restoring each of the five corrupted
candidate shards individually while leaving the other four corrupted shards
unchanged?

## Sweep

Five fresh LoRA siblings are prepared:

- `restore_shard_rca_01`
- `restore_shard_rca_02`
- `restore_shard_rca_03`
- `restore_shard_rca_04`
- `restore_shard_rca_05`

Each sibling:

- starts from the same pinned SmolLM2 parent revision;
- uses the same seed and frozen Experiment 004 training configuration;
- contains 288 training records;
- restores exactly 36 records from one candidate shard;
- leaves exactly 144 candidate corruptions in place; and
- is evaluated on the same six held-out semantic slices plus the aggregate set.

The purpose is descriptive causal mapping, not confirmatory hypothesis testing.

No post-hoc result from this sweep will be relabeled as an Experiment 004
confirmatory success.
