# Model Regression Forensics

> Working research direction. Novelty is **not established**.

This repository investigates a narrow question:

**Can an automated debugger localize the training change responsible for an observed model regression and verify that diagnosis through controlled intervention rather than correlation alone?**

## Current wedge

Adjacent work already covers major parts of the problem: behavioral model diffing, training-data attribution, predictive data debugging, ML pipeline root-cause analysis, and intervention.

The working wedge is narrower:

**post-hoc behavioral regression → structured training lineage → candidate-cause ranking across multiple artifact types → controlled intervention → verified root cause**

The project proceeds only if the related-work review supports a meaningful gap.

## Experiment 000

1. Establish a reproducible baseline checkpoint.
2. Introduce one known training change.
3. Confirm a held-out behavioral regression.
4. Record changed artifacts in a structured lineage manifest.
5. Rank candidate causes without revealing the planted root cause.
6. Intervene on the top candidate.
7. Re-run training/evaluation.
8. Require held-out recovery with stable unrelated evals.

See `experiments/000_planted_regression/README.md`.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Install the heavy ML stack only when Experiment 000 starts:

```bash
uv sync --extra dev --extra research
```

## Research rules

- Do not call the method novel until the primary-source review supports it.
- Do not report planned or synthetic results as completed evidence.
- Keep planted root causes hidden from diagnostic methods.
- Separate diagnosis from verification.
- Verify causes by intervention, not attribution score alone.
- Evaluate held-out recovery and unrelated-capability regressions.
- Keep every experiment reproducible from config + seed + lineage.
