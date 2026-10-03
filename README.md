# Model Regression Forensics

A reproducible case study of training-release debugging: given a known model regression and five visible training-data changes, which change should be investigated first?

**Status:** the matched localization study is complete. Further certification experiments are paused pending an independent contribution review. The repository remains available for reading and reproduction; it is not a production debugging product or a published paper.

## Main result

Two constructed Banking77 worlds contain five structurally matched label-swap changes each. A pinned DistilBERT classifier was trained across three paired trajectories per world: three shared clean models and six composite models. Rankings were finalized before separate truth scoring.

| Diagnostic | World 00 root rank | World 01 root rank |
| --- | ---: | ---: |
| Deterministic random reference | 3 | 3 |
| Target-label overlap | 1 | 1 |
| Lexical Jaccard | 1 | 1 |
| Final-checkpoint Grad-Dot | 1 | 5 |
| Seven-checkpoint TracIn | 1 | 5 |

Simple visible-change baselines localized the planted root in both worlds. The evaluated model-based diagnostics added no top-1 benefit in this design.

The target labels are known and candidate label pairs are disjoint, so label overlap provides a direct semantic shortcut. These are descriptive results from **two worlds**, not six independent worlds or a general ranking of attribution methods.

## What the study distinguishes

- **Localization:** a diagnostic ranks a change as suspicious.
- **Restoration:** reversing a change improves the regressed behavior.
- **Causal specificity:** that effect is distinguishable from plausible alternative interventions and retraining variability.

The completed matched study evaluates localization. It does not establish causal certification, general root-cause identification, or superiority to modern attribution methods.

Earlier development experiments showed why these distinctions matter: correctly locating a planted change and repairing the behavior did not always uniquely identify a cause. Those results are preserved separately rather than pooled with the matched study.

## Read and reproduce

| Reader goal | Start here |
| --- | --- |
| Understand the accepted study and its limitations | [Technical report](research/M4_TECHNICAL_REPORT.md) |
| Follow the complete experimental history | [Experiment history](research/EXPERIMENT_HISTORY.md) |
| Reproduce the published aggregation without training | [Artifact replay](research/REPRODUCE_M4.md) |
| Check exact results and artifact hashes | [Result record](research/M4_RESULT.json) |
| Understand why further compute is paused | [Continuation gate](research/M4_CONTINUATION_GATE.md) and [contribution review](research/M4_CONTRIBUTION_REVIEW.md) |
| Check current state and safe claims | [State](research/STATE.md) and [claims](research/CLAIMS.md) |

Replay verifies aggregation of the retained scoring records and truth evaluation. It does not regenerate the model checkpoints or independently repeat training. The official Banking77 test split remains untouched.

## Repository and checks

`src/` contains the implementation, `scripts/` the experiment entry points, `tests/` the invariant checks, `experiments/` the original protocols and outcomes, and `research/` the accepted evidence and interpretation.

With Python 3.12 or later and uv, run lightweight checks from the repository root:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Optional ML checks require `uv sync --extra dev --extra research`. Model training has additional pinned protocol requirements and is not needed to inspect or replay the accepted result.

Any future certification study requires a distinct scientific justification and a new prospective protocol under the [continuation gate](research/M4_CONTINUATION_GATE.md). Historical evidence remains frozen.
