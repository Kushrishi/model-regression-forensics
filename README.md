# Model Regression Forensics

> **Active research project. Novelty is not established.**

Model Regression Forensics investigates a narrow ML debugging question:

> **Can an automated debugger localize the training change responsible for an observed model regression and verify that diagnosis through controlled intervention rather than correlation alone?**

The project uses controlled synthetic fine-tuning experiments to study **training-lineage-aware root-cause analysis (RCA)** for behavioral regressions in language models. The emphasis is not just on ranking suspicious data changes, but on designing benchmarks that resist easy shortcuts and on verifying suspected causes through selective intervention.

## Research wedge

Adjacent work already covers important pieces of this problem, including behavioral model diffing, training-data attribution, predictive data debugging, ML pipeline RCA, and influence estimation.

The working wedge here is narrower:

```text
behavioral regression
        ↓
structured training lineage
        ↓
multiple plausible data changes
        ↓
blinded candidate-cause ranking
        ↓
selective intervention
        ↓
verified recovery
```

A diagnosis is treated as meaningful only when the benchmark is itself learnable, the hidden cause is not leaked to the diagnostic method, and the suspected cause can be tested by intervention.

## Current status

The repository currently contains the complete research record through **Experiment 003-D**.

The strongest result so far is not a claim of a new algorithm. It is a progressively hardened experimental framework that has already exposed several ways regression debugging can produce misleading conclusions:

- an apparently successful diagnosis can rely on trivial lexical overlap;
- a harder benchmark can fail because the clean model cannot learn the task at all;
- balancing class loss does not necessarily repair that failure;
- isolated capabilities can succeed even when their composition fails;
- causal verification by intervention can reveal spillover that a ranking score alone would hide.

Experiment 004 is the next planned major RCA benchmark.

## Experiment summary

| Experiment | Question | Result | Interpretation |
| --- | --- | --- | --- |
| **000 — Protocol validation** | Can the full baseline → regression → diagnosis → recovery pipeline be reproduced on a controlled task? | **Complete** | Established the reproducible SFT/evaluation/provenance pipeline. |
| **001 — Blinded multicandidate** | Can a debugger identify one hidden causal shard among five changed shards? | **Diagnostic success, benchmark shortcut found** | The hidden cause was recoverable, but whole-artifact lexical overlap made the task too easy. |
| **002 — Entangled distractors** | Does diagnosis still work when target-relevant language is entangled across all candidate shards? | **Complete** | Whole-artifact lexical scores tied. Changed-record analysis uniquely localized the hidden cause. Selective restoration recovered the target behavior, with cross-slice spillover recorded. |
| **003 — Role-binding confounders** | Can the benchmark defeat lexical shortcuts by putting all semantic terms into every prompt? | **Clean baseline failed** | The model achieved 64/96 overall and did not learn the intended task reliably, so candidate/intervention RCA runs were stopped. |
| **003-B — Balanced loss** | Was the 2:1 ACCEPT/REJECT imbalance causing the Exp003 failure? | **Failed to rescue baseline** | Class weighting introduced more REJECT predictions but still did not produce a learnable clean baseline. |
| **003-C — Selected-slot lookup** | Can the same model/training stack learn the selected-slot lookup primitive by itself? | **96/96 held-out** | Selected-slot lookup is learnable under the frozen setup, including held-out decision patterns. |
| **003-D — Explicit-policy role binding** | Can the Exp003 task be solved if the shape→decision policy is supplied explicitly? | **96/96 held-out** | A one-factor policy-prefix change rescued the original role-binding task. |

Detailed protocols and results live under [`experiments/`](experiments/).

## What the 003 series established

The 003 diagnostic sequence was intentionally stopped whenever a clean baseline was not trustworthy.

Under the frozen `SmolLM2-360M-Instruct` + LoRA/SFT setup:

```text
implicit shape policy alone
→ learnable in earlier single-object experiments

selected-slot lookup alone
→ 96/96

multi-object role binding + explicit policy
→ 96/96

multi-object role binding + implicit policy learning
→ failed clean baseline
```

The narrow conclusion is:

> **The unresolved difficulty is associated with combining implicit policy learning/access with the six-object role-binding formulation under this specific frozen model and training regime.**

This does **not** establish the model's internal failure mechanism, and it should not be generalized to language models broadly.

## Why benchmark design matters

A regression-forensics benchmark can produce an impressive-looking result for the wrong reason.

This repository therefore treats benchmark validation as part of the research contribution:

1. **Verify the clean baseline first.**
   RCA results are uninterpretable if the model never learned the intended task.

2. **Hide planted truth from diagnostics.**
   Root-cause metadata belongs to benchmark-owned private artifacts, not diagnostic inputs.

3. **Neutralize obvious shortcuts prospectively.**
   Difficulty gates are checked before result-bearing model runs.

4. **Freeze the diagnostic protocol before revealing truth.**
   Rankings are recorded before the hidden cause is inspected.

5. **Verify by intervention.**
   A high attribution score is not enough; selectively restoring the suspected cause should recover the target behavior.

6. **Record failures and spillover.**
   Negative results are preserved instead of retuning silently.

## Methodology

The current experimental stack uses:

- deterministic synthetic SFT datasets;
- structured and redacted training-lineage manifests;
- multiple opaque candidate data changes;
- pinned model revisions;
- LoRA supervised fine-tuning;
- held-out behavioral evaluation;
- blinded diagnostic ranking;
- selective intervention datasets;
- target/control/full-slice evaluation;
- artifact hashes and runtime provenance;
- prospective construction and anti-leak gates.

The current frozen model used in the 003 diagnostic series is:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
revision: a10cc1512eabd3dde888204e902eca88bddb4951
```

The findings are conditional on this setup unless explicitly replicated elsewhere.

## Repository structure

```text
configs/
    Frozen experiment configurations

experiments/
    Per-experiment protocol and result records

research/
    DECISION_LOG.md
    RELATED_WORK.md

scripts/
    Preparation, training, evaluation, diagnosis, and scoring entry points

src/model_forensics/
    Reusable benchmark, lineage, inference, training, and diagnostic logic

tests/
    Unit tests for configuration, task construction, lineage,
    inference, training, evaluation, and diagnosis
```

Generated model checkpoints, prepared datasets, and run artifacts are written under `artifacts/` and intentionally excluded from Git.

## Reproducibility

### Development environment

```bash
uv sync --extra dev

uv run ruff check .
uv run ruff format --check .
uv run pytest
git diff --check
```

### Research environment

Install the heavier ML dependencies only when running training or inference:

```bash
uv sync --extra dev --extra research
```

Each experiment has its own frozen config and experiment README with the exact preparation/training/evaluation sequence.

For example:

```bash
uv run python scripts/prepare_exp003d.py

uv run python scripts/train_exp003d_sft.py \
  --run-id baseline_explicit_policy

uv run python scripts/eval_exp003d_adapter.py \
  --adapter artifacts/exp003d/checkpoints/baseline_explicit_policy/adapter \
  --run-id baseline_explicit_policy
```

The committed experiment result files describe the observed outcomes. Generated `artifacts/` are not treated as source-controlled evidence by themselves.

## Research record

Two files are especially important:

- [`research/DECISION_LOG.md`](research/DECISION_LOG.md) records consequential experimental decisions, failures, and interpretation changes.
- [`research/RELATED_WORK.md`](research/RELATED_WORK.md) tracks the literature review used to evaluate whether the research wedge is actually distinct.

Per-experiment results are stored next to their protocols:

```text
experiments/<experiment>/README.md
experiments/<experiment>/RESULTS.md
```

## Current findings

The evidence so far supports several practical observations:

### 1. Whole-artifact lexical similarity can be a misleading RCA shortcut

Experiment 001 produced a successful blinded diagnosis, but inspection showed that the correct shard was also lexically obvious at the whole-artifact level. The benchmark was therefore strengthened rather than treating the result as sufficient evidence.

### 2. Change-focused analysis can remain informative when whole-artifact similarity is neutralized

Experiment 002 deliberately entangled target-relevant content across all five changed shards. Whole-artifact lexical ranking became uninformative by construction, while analysis restricted to the changed records uniquely identified the hidden target-relevant change.

### 3. Intervention provides information that ranking alone does not

Selective restoration of the suspected Exp002 cause recovered the intended target behavior, but also produced cross-slice spillover. That spillover is part of the result and limits how narrowly the intervention can be interpreted.

### 4. A failed clean baseline is a benchmark failure, not an RCA result

Experiment 003 was stopped before candidate/intervention analysis because the clean baseline did not reliably learn the task. This prevented model-capability failure from being misreported as forensic-method failure.

### 5. The Exp003 difficulty is compositional under the frozen setup

Exp003-C showed that selected-slot lookup alone was learned perfectly. Exp003-D showed that the original multi-object task was also learned perfectly when the policy was written explicitly. The remaining failure therefore lies in a more specific interaction than either capability alone.

## Limitations

The project is still early-stage research.

Current limitations include:

- synthetic tasks rather than production regressions;
- one primary small language model in the current diagnostic series;
- limited seed/model-family replication so far;
- handcrafted candidate-change structures;
- no established novelty claim;
- no broad comparison yet against modern data-attribution or influence-estimation baselines;
- some intervention effects are not perfectly localized;
- behavioral conclusions are conditional on the exact frozen training setup.

These limitations are deliberate targets for later experiments rather than hidden assumptions.

## Next: Experiment 004

Experiment 004 is intended to return from capability debugging to the primary RCA question.

The planned benchmark should combine the lessons from Experiments 001–003:

- a prospectively verified learnable clean baseline;
- multi-object prompts that resist simple lexical shortcuts;
- explicit policy access to avoid the Exp003 learnability failure;
- multiple opaque candidate lineage changes;
- equalized candidate/change statistics where practical;
- predeclared diagnostic ranking rules;
- hidden benchmark-owned root cause;
- target and control behavior;
- selective intervention for causal verification;
- no post-result retuning if the planted regression fails to materialize.

Later work should expand from single benchmark instances toward repeated seeds, repeated generated worlds, stronger baselines, and additional model regimes.

## Research rules

- Do not call the method novel until the primary-source review supports it.
- Do not report planned or synthetic results as completed evidence.
- Keep planted root causes hidden from diagnostic methods.
- Separate diagnosis from verification.
- Freeze ranking protocols before revealing benchmark truth.
- Verify causes by intervention, not attribution score alone.
- Evaluate target recovery and unrelated-capability spillover.
- Stop RCA experiments when the clean baseline itself is invalid.
- Record negative results rather than silently retuning.
- Keep every experiment reproducible from config, seed, lineage, and committed protocol.

---

**Research status:** active.
**Current stable history:** Experiments 000 through 003-D.
**Next major experiment:** 004.
