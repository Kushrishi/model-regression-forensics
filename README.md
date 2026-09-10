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
truth-isolated candidate-cause ranking
        ↓
selective intervention
        ↓
verified recovery
```

A diagnosis is treated as meaningful only when the benchmark is itself learnable, the hidden cause is not leaked to the diagnostic method, and the suspected cause can be tested by intervention.

## Current status

The repository contains the complete research record through **Experiment 007**,
with **Experiment 008** now active under a prospectively frozen two-world
protocol.

The project has progressed from demonstrating a reproducible regression-debugging
pipeline to testing increasingly strict requirements for causal root-cause
analysis.

The central lesson remains that attribution is not causal verification. A valid
experiment must first produce a localized behavioral regression, and the
suspected cause must then demonstrate selective recovery under controlled
counterfactual restoration.

Experiments 004 through 007 progressively tightened that requirement:

- Experiment 004 correctly localized the benchmark-designated target shard, but
  restoring that shard produced no target recovery.
- Experiment 005 showed that preserving aggregate class counts was insufficient
  to produce an admissible localized regression.
- Experiment 006 controlled corruption directly in semantic space, but the
  intended `triangle_large` behavior still did not regress.
- Experiment 007 established target materiality during calibration, but the same
  intervention also damaged protected behavior. Materiality therefore did not
  imply locality, and causal certification was not authorized.

Experiment 008 was designed as the final major iteration of the current
synthetic shape substrate. Its scientific construction was committed before
model training. The public frozen manifest is truth-free, both worlds satisfy
the prospective static construction gates, and the clean baseline has now
scored **96/96 held-out with 16/16 on every semantic slice**.

Frozen candidate evaluation is the active stage. No Experiment 008 causal
success is claimed unless both candidate worlds first pass the localized
regression gate and subsequent counterfactual restoration uniquely supports a
cause.

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
| **007 — Sensitivity-calibrated causal RCA** | Can calibration establish target materiality before untouched causal certification? | **Materiality achieved; locality failed** | Candidate training produced strong target regression, but protected behaviors also regressed, so certification and restoration were not run. |
| **008 — Selective causal RCA** | Can a prospectively frozen intervention isolate target corruption from policy-correct nuisance changes and support unique counterfactual recovery? | **Active; clean baseline 96/96** | Two worlds were frozen before training. The clean baseline passed perfectly; candidate evaluation is underway. |

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
- truth-isolated diagnostic ranking;
- selective intervention datasets;
- target/control/full-slice evaluation;
- artifact hashes and runtime provenance;
- prospective construction and anti-leak gates.

The primary frozen model used from the 003 diagnostic series through Experiments 004–008 is:

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

### 9. Target materiality does not guarantee behavioral locality

Experiment 007 solved the materiality problem that blocked Experiment 006:
candidate training could strongly regress `triangle_large`.

However, the same construction also damaged protected behavior. Because the
prospectively declared locality gate failed, certification and restoration were
not run.

### 10. Experiment 008 is prospectively frozen before result-bearing candidate evaluation

Experiment 008 separates one target-specific policy-inconsistent intervention
from four policy-correct nuisance permutations while preserving the global
prompt multiset.

Both frozen worlds passed their static construction gates before model
training. The shared clean baseline then scored 96/96 held-out.

This establishes benchmark readiness, not causal success. Candidate evaluation,
counterfactual restoration, and diagnostic scoring remain result-bearing stages.

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

## Current: Experiment 008 — Selective Causal RCA

Experiment 008 is the active follow-up to the locality failure in Experiment
007 and the final major iteration planned for the current synthetic shape
substrate.

Its protocol was committed before model training.

The two frozen worlds each contain five debugger-visible candidate data changes.
Exactly one candidate introduces target-specific policy-inconsistent
supervision. The other four are policy-correct nuisance permutations designed
to alter deterministic training order/content placement without introducing
incorrect protected labels.

The public frozen manifest does not explicitly contain root truth. Because root
identities were visible during benchmark-construction validation, Experiment
008 does **not** claim investigator blinding. Instead, diagnostic methods are
evaluated through a truth-isolated interface and scored against private truth
only after their rankings are frozen.

The clean baseline is shared across both worlds because the clean training and
evaluation inputs are byte-identical. A fresh CPU run under recorded runtime
provenance scored **96/96 overall and 16/16 on every slice**.

The next required gate is candidate locality:

1. target regression on `triangle_large` must be at least the frozen minimum;
2. every protected slice must remain within the frozen drift bound;
3. both frozen worlds must pass;
4. failed worlds are not repaired by changing seeds, thresholds, dose, or
   hardware backend.

Only if both candidate worlds pass will the experiment proceed to five
independent counterfactual restorations per world.

A causal diagnosis requires the true restoration to recover the target while
protected behavior remains stable, non-root restorations to fail to produce
material target recovery, and exactly one candidate to satisfy the recovery
criterion.

If the frozen candidate construction fails, the current shape substrate is
retired rather than tuned into an Experiment 009.

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
**Current stable history:** Experiments 000 through 007.
**Active follow-up:** Experiment 008 — frozen candidate evaluation.
