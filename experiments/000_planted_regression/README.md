# Experiment 000 — One planted regression

Status: **design**

## Purpose

Validate the research protocol with one deliberately introduced, reproducible post-training regression whose true cause is known to the experiment harness but hidden from the diagnostic method.

This experiment does **not** establish novelty.

## Candidate model

Initial target: `HuggingFaceTB/SmolLM2-360M-Instruct`

Reasons:
- 360M parameters;
- Transformers support;
- Apache-2.0 license;
- practical for repeated intervention/retraining experiments.

We may switch if baseline behavior or Apple Silicon training support is poor.

## Experimental shape

Create:
- baseline run;
- candidate run with one planted change;
- recovery run where the suspected change is removed/reversed.

The first planted cause is a single corrupted SFT shard that teaches an incorrect response policy for one semantic slice of a simple instruction task.

## Success criteria

Experiment 000 passes only if:

1. baseline meets the declared target;
2. candidate regresses by the declared minimum;
3. regression holds on held-out examples;
4. planted change is present in recorded lineage;
5. diagnostic code ranks causes without seeing the hidden label;
6. intervention produces meaningful recovery;
7. unrelated evals remain approximately stable;
8. run is reproducible from a clean checkout.
