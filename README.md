# Model Regression Forensics

Model Regression Forensics compares classification releases, executes declared
repairs and lets an engineer inspect the affected examples and retained evidence.
Its research asks what distinguishes a successful repair from a uniquely
responsible training change.

**Current software:** explicit baseline/candidate/repair execution, identical-case
comparison, portable HTML inspection, input previews, filtering and selection
export. The [digit walkthrough](docs/investigation-workflow.md) is executable.
The [real-text training-release walkthrough](docs/banking-investigation.md)
executes two visible training changes and their separate rollbacks on 1,969
development cases. Five CPU fits completed; only the combined rollback passed
the declared repair policy.

The [incremental intent-release experiment](docs/incremental-intent-release.md)
adds 30 intents while protecting 120 existing ones. Fifteen actual fits across
three fixed development scenarios found that ordinary full-data retraining was
the only acceptable repair; both partial rollbacks failed. Saved predictions and
portable reports support inspection, not a claim of diagnostic superiority.

**Completed research:** simple label and lexical baselines found the planted
change in both Banking77 worlds; Grad-Dot and TracIn added no consistent top-1
benefit. The separate small SmolLM2 feasibility attempt remains closed. These
findings are preserved; application development does not relabel them as success.

[Current status](research/STATE.md) · [Direction and milestones](research/ROADMAP.md)

[Visual study](https://kushrishi.com/research/model-regression-forensics) · [Technical report](research/M4_TECHNICAL_REPORT.md) · [Reproduction](research/REPRODUCE_M4.md)

![Rank of the planted change among five candidates in two constructed Banking77 benchmarks](docs/assets/ranking-results.svg)

## Use your own saved predictions

Install the package and run `mrf-import policy.json predictions.csv investigation`
to build a portable classification-release report without loading or training
models. See the [input contract and walkthrough](docs/import-predictions.md).

The [quantized-release example](docs/deployment-comparison.md) provides 1,969
saved real-text predictions for a passing candidate with eight changed labels.
Inspect the affected cases without training or downloading a model.

## Completed localization study

Two constructed Banking77 benchmarks (A and B; recorded as world 00 and world 01) contain five structurally matched label-swap changes each. A pinned DistilBERT classifier was trained across three paired trajectories per world, and rankings were finalized before truth scoring.

| Diagnostic | Benchmark A root rank | Benchmark B root rank |
| --- | ---: | ---: |
| Deterministic random reference | 3 | 3 |
| Target-label overlap | 1 | 1 |
| Lexical Jaccard | 1 | 1 |
| Final-checkpoint Grad-Dot | 1 | 5 |
| Seven-checkpoint TracIn | 1 | 5 |

Simple visible-change baselines localized the planted change in both worlds. The evaluated model-based diagnostics added no top-1 benefit in this design.

The target labels are known and candidate label pairs are disjoint, which gives the simple label-overlap baseline a semantic shortcut. These are results from **two benchmark worlds**; the three paired trajectories within each world are repeated trainings, not independent benchmark worlds.

## What this result means

The project keeps three questions separate:

- **Localization:** which change looks suspicious?
- **Restoration:** does reversing a change repair the failed behavior?
- **Specificity:** is that repair distinguishable from plausible alternative repairs and retraining variability?

The completed matched study evaluates localization. It does not establish general root-cause identification or superiority to modern training-data attribution methods.

The follow-up repair example tests this distinction directly: two interventions restore the same predictions, so the assessment reports ambiguity.

## Comparison utility

The repository also contains an experimental exact-label release comparison utility. It validates record alignment, declared evaluation slices, and accuracy-drop tolerances. An external handwritten-digits fixture agrees with an independent NumPy reference.

That checks implementation arithmetic; it does not identify the cause of a regression.

The [saved-repair workflow](docs/ambiguous-repairs.md#evaluate-saved-repair-predictions)
evaluates supplied regression and repair predictions under an identical, verified
evaluation-policy identity. It rejects different cases or labels even when summary
counts match, and reports multiple successful repairs as ambiguous.

## Read and reproduce

- [Technical report](research/M4_TECHNICAL_REPORT.md)
- [Experiment history](research/EXPERIMENT_HISTORY.md)
- [Reproduce the recorded results](research/REPRODUCE_M4.md)
- [Accepted result record](research/M4_RESULT.json)
- [External comparison fixture](docs/external-release-task.md)
- [LFQ software-defect replay: fixed tensors, no training](research/LFQ_FIXED_TENSOR_REPLAY_2026_10_09.md)

With Python 3.12 or later and `uv`:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Replaying the saved scoring records does not retrain the models. Research-only numerical checks have additional optional dependencies documented in the repository.

## Ambiguous repair fixture

A deterministic linear-model fixture now contains one regression and two distinct interventions that both restore the baseline predictions exactly: restoring the original feature order, or keeping reversed inputs while reversing the model weights.

The same release-comparison policy marks both repairs successful. The specificity layer therefore returns `ambiguous_repairs` and leaves the historical cause `not_identified`. The known historical change is not provided to that decision layer.

[Read the fixture](docs/ambiguous-repairs.md)

## Proposed follow-up: avoiding unjustified root-cause claims

The next research question is whether a debugger can reduce incorrect, overly specific diagnoses without simply refusing to diagnose everything. It would have to beat ordinary diff inspection and targeted reversions at a comparable testing cost, using real incidents with known-good and regressed versions. Current incident screens have found software defects and useful repair examples, but no qualified training-release case with the required outcome and provenance evidence. The existing ambiguous-repair example demonstrates intended software behavior; it does not establish research novelty.

## Research history and limits

[Evidence guide](research/README.md) · [Software architecture](docs/architecture.md)

The [experiment history](research/EXPERIMENT_HISTORY.md) preserves the completed
localization study, unsuccessful development attempts and incident-discovery
records. The [roadmap](research/ROADMAP.md) defines current work; those older
records are not a second active plan.

MRF has not established general causal identification, superiority to attribution
methods, independent-user benefit or a published paper. Public distribution
licensing remains unresolved.

## Run a release investigation

Execute baseline, candidate and repair functions, then reopen their retained predictions in a portable browser report. The [handwritten-digit walkthrough](docs/investigation-workflow.md) demonstrates an actual model and a disclosed preprocessing fault. This is an application integration example, not a causal-discovery benchmark.
