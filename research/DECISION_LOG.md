# Decision log

## 2026-08 — Scope refinement

### Rejected

"pytest for ML robustness" and generic failure discovery are too crowded.

### Rejected

"discover failures → generate data → post-train → evaluate → repeat" overlaps too strongly with existing automated red-teaming and improvement loops.

### Current framing

**Causal model-regression forensics**

1. observe a behavioral regression;
2. represent training lineage between baseline and candidate;
3. rank candidate causes;
4. intervene on a suspected cause;
5. accept the diagnosis only if behavior recovers under controlled conditions.

### Novelty status

**Not established.**

Goodfire's 2026 predictive data debugging raises the bar substantially. The project must distinguish itself through post-hoc diagnosis, heterogeneous lineage, benchmark design, or another defensible gap.

## 2026-08-29 — Zero-shot metric refinement

The pinned untouched `SmolLM2-360M-Instruct` reference produced a target strict-exact score of 0.00 and an unrelated strict-exact score of 0.05. Raw generations included both `REJECT` and `Reject`, showing that strict exact match mixed the behavioral question with capitalization.

This was treated as a pilot measurement issue, not as an experiment result to optimize around. Before any SFT baseline, candidate, or recovery run:

- primary behavior metric: one-token `ACCEPT`/`REJECT` label accuracy, ignoring outer whitespace and letter case only;
- secondary format metric: strict exact match after outer-whitespace trimming;
- explanations or other text remain incorrect for both protocol purposes;
- the original zero-shot strict scores remain recorded rather than being retroactively reinterpreted.

The zero-shot model is therefore a reference control. Experiment 000 requires a clean SFT baseline checkpoint before introducing the planted regression.

## 2026-08 — Experiment 000 SFT protocol

The untouched SmolLM2 reference predicts `REJECT` for all held-out examples, so it is not the Experiment 000 baseline checkpoint.

Baseline, candidate, and recovery will be trained as sibling LoRA SFT runs from the same pinned parent revision rather than sequentially from one another. This isolates the prepared training split as the intended manipulated variable.

The first declared training configuration is 10 epochs, batch size 8, learning rate 5e-4, linear decay with 5% warmup, float32 model loading, and rank-16 LoRA on attention and MLP projection layers. The prepared dataset order is fixed and the loader does not reshuffle it. Any baseline-only pilot adjustment must be logged before candidate or recovery training.

## 2026-08-29 — Experiment 001 model-validation protocol

Experiment 001 reuses the frozen Experiment 000 LoRA SFT protocol without
hyperparameter tuning. Baseline, candidate, and intervention are fresh sibling
runs from the same pinned SmolLM2 parent revision and seed; only the prepared
training split differs.

Before observing Experiment 001 model results, the success criteria remain the
thresholds declared in `configs/exp001.yaml`: baseline target label accuracy at
least 0.80, baseline-to-candidate target regression at least 0.15,
candidate-to-intervention target recovery at least 0.10, and maximum drift on
the unchanged `square_small` control slice at most 0.05.

The intervention run restores the benchmark-owned target-causal shard while
leaving the other four changed shards in place. This is an oracle benchmark
validation step, not a diagnostic result. Blinded root-cause ranking begins only
if the planted target regression and selective intervention behave as intended.

## 2026-08-29 — Experiment 001 blinded-diagnosis baselines

Before scoring any diagnostic ranking against benchmark ground truth, Experiment
001 will freeze two deliberately simple baselines:

1. a seeded random ranking as a lower reference; and
2. a lexical-overlap ranking that compares observed regression prompts against
   debugger-visible changed-shard prompts using mean best Jaccard token overlap.

The diagnostic command is structurally ground-truth-free: it may read the
redacted diagnostic lineage, changed artifacts referenced by that lineage, and
baseline/candidate target generations. It must not read `benchmark.json`.
Benchmark scoring is a separate command run only after the ranking artifact has
been written.

The lexical baseline is expected to be strong on Experiment 001 because its
shards map cleanly to behavioral slices. Success here is therefore a pipeline
sanity check, not evidence of difficult or general root-cause localization.

## 2026-08-29 — Experiment 001 diagnosis outcome and Experiment 002 constraint

Experiment 001 completed the first blinded ranking/scoring cycle. The committed
lexical-overlap baseline ranked `shard_delta_04` first with score `0.9091`; the
next three candidates each scored `0.8261`. The seeded-random baseline also
ranked the hidden cause first by chance. With five candidates, the random Top-1
reference is `0.20`, so a single instance cannot support a method-performance
claim.

The result confirms that Experiment 001 is too separable for difficult RCA: the
true causal shard has uniquely strong lexical overlap with the observed
`triangle_large` failures. This was anticipated and is now an explicit design
constraint for Experiment 002. Several candidate shards must contain
target-relevant content with comparable superficial overlap, while only one is
causal for the specified regression. The benchmark should force stronger
attribution methods to beat lexical similarity rather than rewarding a
construction shortcut.

Novelty remains **not established**. Experiment 001 validates the evaluation
protocol and anti-leak workflow; it is not itself a novel scientific discovery.
