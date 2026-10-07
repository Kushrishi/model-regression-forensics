# Bounded v2 development attempt closed — 7 October 2026

Decision: STOP at Stage 1; close this v2 continuation attempt. MRF's completed
v1 negative study and this unsuccessful development feasibility attempt are the
program outcome. No MRF v3, expanded benchmark or confirmatory study follows.

The owner transparently amended the prior review prerequisite; no independent
review occurred. The [pre-training development design](DEVELOPMENT_FEASIBILITY_2026_10_07.md)
and [exact summary](DEVELOPMENT_FEASIBILITY_RESULT_2026_10_07.json) preserve the attempt.
It cannot be relabeled as confirmatory evidence.

## Observed feasibility

Two clean LoRA fits used the pinned SmolLM2-135M-Instruct model and four related
Banking77 intents. All examples came from the official training CSV; the official
test split was neither fetched nor accessed. Each fit used 64 training records,
32 in-training-split development records and 48 updates. These are two seeds of
one substrate, **not two incidents or independent scientific replicates**.

| Clean seed | Overall exact accuracy | Target accuracy | Protected accuracy | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| 11 | 20/32 = 0.625 | 13/16 = 0.8125 | 7/16 = 0.4375 | 106.812 |
| 23 | 24/32 = 0.750 | 9/16 = 0.5625 | 15/16 = 0.9375 | 93.017 |

All 64 generated responses were valid intent strings, so the failure was not JSON
parsing or invalid output shape. Seed 11 failed the predeclared protected accuracy
floor of 0.50. Protected accuracy also varied by 0.50 across seeds, while target
accuracy varied by 0.25. The clean substrate therefore did not satisfy the
specified stable-useful-baseline gate for a regression study.

Total training-stage wall time was 199.829 seconds, CPU only, two threads;
nominal two-thread wall envelope is approximately 0.111 CPU-hours. Peak RSS was
1,808,972 KiB (about 1.73 GiB). No GPU, paid API, regressed-release fit, candidate
incident, intervention or attribution-method comparison was run. No unnecessary
checkpoint history was retained.

## Interpretation and limits

This rejects the **tested small substrate/recipe** as a reliable basis for the
proposed diagnostic pilot. It does not show that SmolLM2 can never learn the task,
that a larger model is required, or that budgeted differential diagnosis is
impossible. Model capacity, limited training data and recipe inadequacy are not
separated by these two runs. The owner authorized a falsification-first attempt,
not an open-ended search for a favorable substrate. Further tuning or Qwen
escalation would expand the attempt without a demonstrated decision benefit.

No wrong-specific-diagnosis, risk/coverage or intervention-savings result exists:
Stage 1 stopped before those could be measured. Strong simple diagnosis baselines
were not evaluated on a new incident set because no qualified incident set was
constructed. The earlier M4 equal-information label/lexical negative finding
remains intact, without a general inferiority claim about attribution methods.

The practical release-diagnosis distinction remains unestablished, and substantial
prior art occupies its ingredients. The bounded attempt closes here rather than
manufacturing hard incidents, hiding release-diff evidence, or promising a new
method. There is no publication submission or reviewer communication.

## Reproduce the development boundary

Use the exact environment/model/tokenizer/source identities in the design. Keep
only `banking_data/train.csv` from the pinned dataset source; do not fetch test.csv.
The local model directory contains config/tokenizer/weights plus their SHA-256
manifest (model and tokenizer share the same immutable revision). Run the retained
script with seeds 11 and 23, `--release clean`, a separate new `--out` directory
per invocation, and an external 600-second timeout. The script verifies artifact
hashes and the training CSV identity before fitting; output directories never
clobber previous attempts. There is no confirmatory or intervention command here.

The initial source-schema invocation failed before model loading/training because
provisional label names were absent. Its log/directory remain preserved; the
subsequent exact-label correction used metadata only. Software setup initially
lacked the installed project package, corrected before repository tests. Neither
failure is counted as a scientific fit or quietly overwritten.

## Validation environment boundary

The repository's locked development environment passed lint/format and 267 unit
tests, with six optional-dependency skips. The separate old-version feasibility
environment passed 279 tests, skipped two, and failed two pre-existing TRAK adapter
tests because Transformers 4.47.1 lacks `set_attn_implementation`; it is not the
repository's locked research environment (Transformers 5.16.1). That incompatibility
is recorded, not patched by changing the completed attribution implementation.
Banking77 is CC BY 4.0 at the pinned source; cite Casanueva et al., *Efficient Intent
Detection with Dual Sentence Encoders*, NLP for ConvAI/ACL 2020. The model-card
license is Apache-2.0. No training text or model binary is added to this commit.
