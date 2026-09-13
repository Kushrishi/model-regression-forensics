# Model Regression Forensics

> **Active ML research project**

Model Regression Forensics studies a practical model-debugging problem:

> **When a model regresses after retraining, can we identify which training change caused the regression and verify that diagnosis experimentally?**

## The problem

A model can perform correctly before retraining and then lose a capability after
new data or training changes are introduced.

Measuring the regression is only the first step. The more difficult question is
identifying which training change caused it.

A training change may appear strongly related to a failure without actually
being responsible for the model's behavior. This project therefore treats
diagnosis and verification as separate problems.

## Approach

The research follows a controlled workflow:

1. Compare a clean reference model with a retrained model.
2. Measure which behaviors changed.
3. Record and analyze the training changes that could explain the regression.
4. Rank the most plausible causes.
5. Reverse individual candidate changes and retrain under the same conditions.
6. Measure whether the target behavior recovers while unrelated behavior
   remains stable.

The final step is essential. A candidate is not treated as a verified cause
simply because it receives a high ranking.

## Current status

Experiments **000 through 007 are complete**.

The experimental series has progressively exposed weaknesses in both the
debugging method and the benchmark used to evaluate it.

Key findings so far include:

- simple lexical similarity can create misleading attribution signals;
- correctly localizing a suspicious training change does not establish that it
  caused the regression;
- a benchmark is not useful if the intended regression never appears;
- creating a target regression is also insufficient if unrelated behaviors
  degrade at the same time.

These failures were retained as experimental results and used to strengthen
later designs rather than being tuned away after observing the outcome.

## Experiment 008

Experiment 008 is complete.

Both frozen worlds produced target-localized regressions with zero protected-slice drift. The truth-isolated `selected_role_overlap` diagnostic uniquely ranked the planted root first in both worlds, and restoring the planted root fully recovered the target behavior with zero protected drift.

Primary causal certification nevertheless **failed in both worlds** because some non-root restorations also produced target recovery above the frozen nuisance ceiling. Experiment 008 therefore supports successful localization and strong planted-root restorative influence under this controlled benchmark, but not uniquely specific causal certification.

The frozen stopping rule was honored. Thresholds were not changed, the alternative-order control was not used as a rescue analysis, and the synthetic shape substrate is retired from further benchmark tuning.

See [`experiments/008_selective_causal_rca/RESULTS.md`](experiments/008_selective_causal_rca/RESULTS.md) for the complete result record.

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
| **008 - Selective causal RCA** | Can a prospectively frozen intervention isolate target corruption from policy-correct nuisance changes and support unique counterfactual recovery? | **Localization + root recovery; unique certification failed** | Both worlds passed the localized-regression and localization gates. Planted-root restoration fully recovered the target, but non-root restorations also produced material recovery, so the frozen causal-specificity criterion failed. |

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

## Why the test itself has to be trustworthy

A debugging experiment can produce an impressive-looking result for the wrong reason.

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

### 8. Better-controlled training changes still did not create the intended failure

Experiment 006 controlled corruption directly in semantic space across five
prospectively frozen candidate worlds.

Every fresh clean sibling scored 96/96.

Yet the intended `triangle_large` regression was exactly 0.0 in all five
candidate worlds. Some worlds instead damaged protected square behavior, while
others produced no measurable regression.

The next benchmark-design problem is therefore to establish behavioral
materiality prospectively without selecting or tuning on the final
certification evaluation.

### 9. Creating the target failure is not enough if unrelated behavior also breaks

Experiment 007 solved the materiality problem that blocked Experiment 006:
candidate training could strongly regress `triangle_large`.

However, the same construction also damaged protected behavior. Because the
prospectively declared locality gate failed, certification and restoration were
not run.

### 10. Experiment 008 separated localization from uniquely specific causal verification

Experiment 008 produced a target-localized candidate regression in both frozen worlds while all protected slices remained stable.

The truth-isolated `selected_role_overlap` diagnostic uniquely ranked the planted root first in both worlds. Restoring that planted root then fully recovered the target in both worlds with zero protected-slice drift.

However, some non-root restorations also produced target recovery above the prospectively frozen nuisance ceiling. World 01 additionally contained one non-root restoration with substantial protected-slice drift.

The primary causal certification therefore failed in both worlds.

This result strengthens the distinction between localization, restorative influence, and uniquely specific causal verification. It also motivates measuring retraining-path variability explicitly rather than treating a single fresh retraining outcome as sufficient causal evidence.

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

## Experiment 008 outcome and next phase

Experiment 008 completed the final major evaluation on the synthetic shape substrate.

The final result was:

- shared clean baseline: **96 / 96**;
- both frozen candidate worlds passed the target-regression and protected-locality gate;
- the planted root was uniquely ranked Top-1 by the task-aware diagnostic in **2 / 2** worlds;
- restoring the planted root fully recovered the target in **2 / 2** worlds with zero protected-slice drift;
- non-root recovery specificity failed in **2 / 2** worlds;
- primary causal certification therefore **failed in 2 / 2 worlds**.

The prospectively defined alternative-order robustness control was not run because primary certification did not succeed. It is not used as a post-hoc rescue.

The synthetic shape substrate is now retired as the primary benchmark. The next research phase will move to a more realistic natural-language regression setting and explicitly measure training/retraining variability using repeated, paired counterfactual runs before interpreting restoration as uniquely causal.

The exact next experiment protocol has not yet been frozen.

### Technical details

The Experiment 008 directory contains the exact model revision, training
parameters, dataset hashes, runtime information, benchmark construction,
evaluation thresholds, and restoration rules.

The public summary intentionally does not treat a diagnostic ranking as proof
that something caused the regression. A controlled restoration test is required
before making that claim.

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
**Current stable history:** Experiments 000 through 008.
**Next phase:** realistic regression validation with explicit retraining-variability controls.
