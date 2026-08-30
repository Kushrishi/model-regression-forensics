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

## 2026-08-29 — Experiment 002 entangled-distractor protocol

Experiment 002 remains data-shard-only and keeps `triangle_large` as the target
regression and `square_small` as the unchanged control. Its purpose is to remove
the artifact-level lexical shortcut observed in Experiment 001 before adding
more sophisticated attribution methods or heterogeneous cause classes.

The frozen construction uses five opaque candidate shards with 48 records and 32
label changes each. The benchmark-private causal shard contains 32 changed
`triangle_large` records. Each distractor contains four unchanged
`triangle_large` records spanning all four colors, 32 changed records from
another behavioral slice, and clean filler. The selective oracle intervention
restores only the 32 target changes and leaves 128 other label changes intact.

Before any model training, the exact mean-best Jaccard lexical-overlap method
used in Experiment 001 must assign all five candidates equal scores within
`1e-12`. Failure of this construction gate requires benchmark redesign rather
than training or post-hoc adjustment.

This gate neutralizes the existing artifact-level lexical scorer, not every
possible low-cost heuristic. In particular, the causal shard contains more
actual target records than each distractor to preserve a plausible planted
regression. If count-based or change-aware lexical methods solve Experiment 002,
that result will define the next shortcut to remove rather than being hidden.

The hidden cause is generated deterministically from the seed and is not declared
in `configs/exp002.yaml` or printed by the preparation summary. Human blindness
remains procedural rather than cryptographic; diagnostic code must still consume
only redacted lineage.

## 2026-08-29 — Exp002 model validation starts only after the difficulty gate

- The Exp002 artifact-level lexical-overlap difficulty gate passed with an exact five-way tie.
- Model validation therefore proceeds with the unchanged sibling LoRA SFT protocol from Exp000–001.
- Baseline, candidate, and oracle-intervention runs must each start from the same pinned parent revision and seed.
- No candidate-specific hyperparameter tuning is permitted.
- The private Exp002 benchmark manifest remains unopened until blinded diagnostic rankings are frozen.


## 2026-08-29 — Experiment 002 model-validation outcome

Experiment 002 passed all predeclared model-validation gates under the unchanged
sibling LoRA SFT protocol. The clean baseline scored 1.00 on target, control,
and all 96 evaluation cases. The candidate scored 0.00 on the 16-case
`triangle_large` target and 1.00 on the unchanged 16-case `square_small`
control, yielding a target regression delta of 1.00 and control drift of 0.00.
The frozen 32-corrupted/16-clean target mixture was therefore sufficient to
induce complete held-out target regression without post-hoc tuning.

The oracle intervention, which restores only the 32 benchmark-owned target
label changes while leaving 128 other label changes in place, recovered target
accuracy from 0.00 to 1.00 and preserved control accuracy at 1.00. The
predeclared causal recovery gate therefore passed.

A stronger qualitative prediction did not hold: intervention all-set accuracy
was 44/96 (0.4583), not 32/96 (0.3333). Besides full recovery of the target and
unchanged control, 12 cases from other altered slices returned to canonical
behavior: 6 `circle_small`, 4 `circle_large`, and 2 `triangle_small`;
`square_large` remained 0/16. This cross-slice spillover must be reported rather
than hidden. Experiment 002 supports target-level causal recovery, not perfectly
slice-local intervention specificity.

The private benchmark manifest remains unopened. Blinded RCA code and ranking
artifacts must be frozen before ground-truth scoring. Novelty remains not
established.

## 2026-08-29 — Experiment 002 blinded-diagnosis methods

Before generating any Experiment 002 diagnostic ranking or opening the private
benchmark manifest, freeze three model-free baselines:

1. seeded random ranking as a lower reference;
2. the unchanged artifact-level lexical-overlap baseline from Experiment 001;
   Experiment 002 was constructed so this method has an exact five-way score tie;
3. a changed-record lexical-overlap baseline that first aligns each visible
   shard's debugger-visible `before` and `after` records by `example_id`, keeps
   only records whose payload changed, and then applies the same mean-best
   Jaccard prompt-overlap score to the observed regression prompts.

The changed-record method is deliberately simple and uses no model internals. It
tests whether lineage differencing alone is enough to recover the cause after
coarse artifact similarity has been neutralized. If it succeeds, that shortcut
becomes an explicit construction constraint for Experiment 003 rather than a
result to hide.

Diagnosis may read only the redacted diagnostic manifest, changed artifacts
referenced by it, and baseline/candidate target generations. Ground-truth
scoring remains a separate command and must not run until ranking JSON files are
committed and pushed.

Because the artifact-level lexical baseline is tied by construction, ordinary
ID-based sorting is not scientifically meaningful. Scoring will therefore
retain the deterministic ordinal root-cause rank for auditability but compute
Top-1, Top-3, and reciprocal-rank metrics from tie-aware rank bounds. The score
also reports tie size and the best, worst, and average tied rank. A five-way tied
score must not be described as successful localization regardless of the
lexicographic ordering of shard IDs.


## 2026-08-29 — Experiment 002 blinded-diagnosis outcome and Experiment 003 constraint

Experiment 002 completed its blinded ranking/scoring cycle with the ranking
artifacts committed and pushed before private ground-truth scoring. The hidden
root cause was `shard_mix_01`.

The unchanged artifact-level lexical baseline remained an exact five-way score
tie at `0.9090909091`. Tie-aware scoring therefore assigns the hidden cause an
average tied rank of `3.0` and reciprocal rank `1/3`; it is neither uniquely
Top-1 nor guaranteed Top-3. Deterministic identifier ordering must not be
reported as successful localization.

The changed-record lexical baseline uniquely ranked `shard_mix_01` first with
score `0.9090909091`; the next-best candidates scored `0.8260869565`. Seeded
random placed the cause fifth. Thus Experiment 002 demonstrates the intended
progression: entangling whole artifacts defeats coarse lexical similarity, but
simple debugger-visible before/after differencing exposes a new shortcut.

This is not treated as a novel RCA-method result. It defines the construction
requirement for Experiment 003: before any training, the exact
`changed_lexical_overlap` method should be unable to uniquely identify the
benchmark-private cause. Candidate changes should be designed so superficial
target similarity among *changed records* is comparable across plausible
causes, forcing subsequent methods to use evidence beyond changed-record text
overlap. Count-based target-change shortcuts should also be checked explicitly.

One instance remains insufficient for aggregate performance claims. Repeated
blinded instances and stronger baselines are still required before any paper-
level claim about RCA effectiveness. Novelty remains **not established**.


## 2026-08-29 — Experiment 003 clean-baseline failure

Experiment 003 passed all prospective construction and anti-leak gates: both the
artifact-level and changed-record lexical-overlap baselines were exact five-way
ties, candidate target-surface counts were balanced, changed selected-slot
histograms were balanced, public changed-shard records exposed only training-facing
fields, and the tokenizer preflight fit within the unchanged 192-token limit.

The first result-bearing run was the clean baseline under the frozen Exp000–002
LoRA SFT protocol. Training loss fell from 0.3989 to roughly 0.217 and then
plateaued. Held-out label accuracy was 1.00 on `triangle_large`, 0.00 on the
`square_small` control, and 64/96 overall. A public-prompt audit of saved
generations showed `ACCEPT` on all 96 cases. The model therefore learned a
degenerate majority-label policy rather than the intended selected-slot role
binding.

Model validation stops at this point. Candidate and intervention siblings are not
trained and no RCA ranking is attempted. The existing evaluator's
`meets_baseline_threshold` field is target-only and returned true despite the
failed control; it must not be interpreted as a sufficient clean-baseline gate
for this benchmark. Future work should add a stricter clean-task validity gate and
treat any attempt to make role binding learnable as a separately declared
follow-up protocol or experiment version. The negative result remains part of the
record.


## 2026-08-29 — Experiment 003 post-failure training-set audit

After the clean-baseline failure was frozen, the saved baseline adapter was
evaluated without retraining on a balanced 96-example audit sampled from
`baseline_train`: 16 examples from each of the six behavioral slices. The model
again scored 64/96. The 32 failures were the two `REJECT` slices, matching the
held-out constant-`ACCEPT` collapse.

This rules out an explanation based solely on held-out material generalization:
the model did not learn the selected-slot role-binding rule even on sampled
training records. Since the clean dataset has a 2:1 `ACCEPT`:`REJECT` label
ratio, majority-label shortcutting is now the simplest controlled hypothesis,
though not a proven sole cause. The next protocol revision should test class
balance as a single prospective factor before changing epochs, learning rate,
LoRA capacity, or benchmark semantics. Candidate/intervention/RCA remain unrun
for the failed frozen protocol.


## 2026-08-29 — Experiment 003-B class-balanced-loss follow-up protocol

After the frozen Experiment 003 baseline failure and post-failure training-set
audit, declare a one-factor follow-up rather than modifying the failed protocol
in place. The role-binding benchmark and prepared examples remain unchanged.
The clean training split contains 192 `ACCEPT` and 96 `REJECT` examples, and the
saved failed adapter produced the 2:1 majority label even on sampled training
records.

Experiment 003-B tests class contribution as the next controlled factor. Keep the
same model revision, seed, prompts, 288 examples, data order, LoRA configuration,
optimizer, learning-rate schedule, batch size, 10 epochs, 360 optimizer steps,
and 192-token maximum. Change only response-example weighting in the SFT loss:
`ACCEPT=1.0`, `REJECT=2.0`. This yields equal prospective aggregate class mass
(192 versus 192). Do not oversample or downsample.

The weighting implementation must compute completion loss per example and apply
the configured response weight using a global full-split mean-weight
normalization. Do not renormalize by the sum of weights within each batch,
because a homogeneous batch would cancel the intended class weight. For the
frozen 192/96 split, the global mean example weight is 4/3.

The clean-baseline validity check is also made explicit before observing an
Exp003-B result. Target, control, and all-set label accuracy must each be at
least 0.95; target-only success is insufficient. If the balanced baseline fails,
stop before candidate/intervention training and preserve the result. Do not
change epochs, learning rate, LoRA capacity, prompts, or benchmark semantics in
response to that outcome.
