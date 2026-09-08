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

The repository contains the complete research record through **Experiment 006**.

The project has progressed from validating a reproducible regression-debugging
pipeline to testing increasingly strict requirements for causal root-cause
analysis.

The main lesson so far is that a plausible attribution score is not enough.
The benchmark must first produce a valid localized behavioral regression, and
a suspected cause must then demonstrate material recovery under controlled
intervention.

Experiments 004 through 006 sharpened that requirement:

- Experiment 004 correctly localized the benchmark-designated target shard, but
  restoring that shard produced no target recovery.
- Experiment 005 showed that preserving aggregate class counts was not
  sufficient to produce an admissible localized regression.
- Experiment 006 controlled corruption directly in semantic space, but the
  intended `triangle_large` behavior still did not regress in any of five
  frozen worlds.

Experiment 007 is now the active follow-up. Its protocol has not yet been
frozen.

## Experiment summary

| Experiment | Question | Result | Interpretation |
| --- | --- | --- | --- |
| **000 — Protocol validation** | Can the full baseline → regression → diagnosis → recovery pipeline be reproduced on a controlled task? | **Complete** | Established the reproducible SFT, evaluation, and provenance pipeline. |
| **001 — Blinded multicandidate** | Can a debugger identify one hidden causal shard among five changed shards? | **Diagnostic success; shortcut found** | The hidden cause was recoverable, but whole-artifact lexical overlap made the task too easy. |
| **002 — Entangled distractors** | Does diagnosis still work when target-relevant language is entangled across all candidate shards? | **Complete** | Whole-artifact lexical scores tied. Changed-record analysis uniquely localized the hidden cause. Selective restoration recovered the target with recorded spillover. |
| **003 — Role-binding confounders** | Can the benchmark defeat lexical shortcuts by putting all semantic terms into every prompt? | **Clean baseline failed** | The model achieved 64/96 overall and did not reliably learn the intended task, so RCA stopped. |
| **003-B — Balanced loss** | Was the 2:1 ACCEPT/REJECT imbalance causing the Exp003 failure? | **Failed to rescue baseline** | Class weighting changed predictions but did not produce a trustworthy clean baseline. |
| **003-C — Selected-slot lookup** | Can the same model/training stack learn the selected-slot lookup primitive by itself? | **96/96 held-out** | Selected-slot lookup is learnable under the frozen setup. |
| **003-D — Explicit-policy role binding** | Can the Exp003 task be solved if the shape→decision policy is supplied explicitly? | **96/96 held-out** | A one-factor explicit-policy change rescued the role-binding task. |
| **004 — Explicit-policy entangled RCA** | Can blinded localization and diagnosis-driven restoration work on the now-learnable role-binding task? | **Localization correct; causal verification failed** | The task-aware diagnostic correctly localized the hidden target shard, but selective restoration produced zero target recovery. |
| **005 — Causally certified RCA** | Can a balanced corruption construction create a localized regression before causal certification? | **0/5 worlds qualified** | Clean behavior remained perfect, but no candidate world produced the required target-localized regression. |
| **006 — Semantic-balanced causal RCA** | Does controlling corruption directly in semantic space repair the Exp005 construction failure? | **0/5 worlds qualified** | All five clean siblings scored 96/96, but `triangle_large` regression remained exactly 0.0 in every candidate world. |

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

The primary frozen model used from the 003 diagnostic series through Experiments 004–006 is:

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

The evidence so far supports several practical observations.

### 1. Whole-artifact lexical similarity can be a misleading RCA shortcut

Experiment 001 produced a successful blinded diagnosis, but the correct shard
was also lexically obvious at the whole-artifact level. The benchmark was
strengthened rather than treating that result as sufficient evidence.

### 2. Change-focused analysis can remain informative when whole-artifact similarity is neutralized

Experiment 002 deliberately entangled target-relevant content across all five
changed shards. Whole-artifact lexical ranking became uninformative by
construction, while analysis restricted to changed records uniquely identified
the hidden target-relevant change.

### 3. Intervention provides information that ranking alone does not

Selective restoration in Experiment 002 recovered the intended target behavior
but also produced cross-slice spillover. Recovery and unrelated-behavior
effects therefore need to be evaluated separately.

### 4. A failed clean baseline is a benchmark failure, not an RCA result

Experiment 003 was stopped before candidate or intervention analysis because
the clean baseline did not reliably learn the task.

### 5. Controlled capability tests narrowed the Experiment 003 failure

Experiment 003-C showed that selected-slot lookup alone was learned perfectly.
Experiment 003-D showed that the multi-object task was also learned perfectly
when the canonical policy was supplied explicitly.

### 6. Correct localization does not by itself establish causal influence

Experiment 004's task-aware diagnostic correctly and uniquely localized the
hidden target-associated shard before truth reveal. Restoring exactly that
shard nevertheless produced zero target recovery.

The experiment therefore separated successful localization from successful
causal verification.

### 7. Aggregate label balance is not sufficient benchmark control

Experiment 005 preserved aggregate clean and corrupted class counts, but all
five allowed candidate worlds failed the localized-regression gate.

The repeated failure exposed conditional semantic structure as an additional
benchmark-design concern.

### 8. Semantic balancing is still not sufficient to guarantee target materiality

Experiment 006 controlled corruption directly in semantic space across five
prospectively frozen candidate worlds.

Every fresh clean sibling scored 96/96.

Yet the intended `triangle_large` regression was exactly 0.0 in all five
candidate worlds. Some worlds instead damaged protected square behavior, while
others produced no measurable regression.

The next benchmark-design problem is therefore to establish behavioral
materiality prospectively without selecting or tuning on the final
certification evaluation.

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

## Next: Experiment 007 — Sensitivity-Calibrated Causal RCA

Experiment 007 is the active follow-up to the five-world negative result from
Experiment 006.

Its protocol is **not yet frozen**.

The working design direction is to separate two questions that Experiments 005
and 006 implicitly combined:

1. Can a corruption construction measurably influence the intended target
   behavior?
2. Once that construction is frozen, can blinded RCA identify and causally
   verify the responsible training change on untouched certification data?

The intended design principle is to use a prospectively declared calibration
stage that is disjoint from final certification evaluation.

A small corruption-strength family may be evaluated during calibration, with a
deterministic rule selecting the minimum strength that satisfies predeclared
target-regression and protected-behavior criteria.

Only after that selection is frozen would untouched certification worlds be
run.

The calibration procedure, data separation, candidate construction, selection
rule, leakage controls, certification worlds, stopping rule, and claim boundary
must all be committed before any result-bearing Experiment 007 training.

Experiment 007 must not use the final certification evaluation to choose a
corruption strength or repair a failed world.

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
**Current stable history:** Experiments 000 through 006.
**Active follow-up:** Experiment 007 — protocol design in progress.
