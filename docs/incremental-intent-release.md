# Adding intents without abandoning existing ones

This completed development experiment asks whether a classifier release can add
30 intents while retaining the behavior of 120 existing intents. It exercises
MRF's release comparison and repair reports using actual fitted models.

Fifteen CPU fits completed on 9 October 2026. Ordinary full-data retraining was
the only repair that met the predefined policy in all three scenarios. This is
a useful application result, not a new repair algorithm or evidence that MRF
outperforms ordinary engineering practice.

## What was compared

CLINC150's 15,000 official training examples were divided into 12,000 training
and 3,000 development examples by normalized-text hash within class. Three fixed
hash orders chose different sets of 30 added intents. Each scenario evaluated
2,400 existing-intent and 600 added-intent examples. The scenarios overlap; they
are not three independent datasets. No official validation/test scores were used.

The baseline used all existing-intent training examples. The candidate added all
new-intent training examples but retained only one fifth of each existing class,
and changed logistic-regression C from 4 to 0.25. The vocabulary was fitted only
on existing-intent training text. Each partial rollback and their combination
was fitted separately. Full rollback reused the baseline predictions.

The policy, written before fitting, permits at most a two-percentage-point loss
on existing intents and requires at least 70% accuracy on added intents. These
are proposed engineering requirements, not validated customer requirements.

## Measured results

Ranges below span the three scenarios; they are not confidence intervals.

| Model or intervention | Existing-intent accuracy | Added-intent accuracy | Passed policy |
| --- | ---: | ---: | --- |
| Original model / full rollback | 95.00–95.46% | 0% | No: unsupported new labels |
| Candidate | 0.04–0.08% | 88.83–91.83% | No |
| Restore all existing training data | 86.54–87.33% | 78.67–82.00% | No |
| Restore regularization only | 74.04–76.00% | 91.50–93.83% | No |
| Restore both / full-data retrain | 94.00–94.38% | 88.33–90.00% | Yes, all three |

The candidate's severe existing-intent failure is retained as observed. Its
training set has 16 examples per existing class and 80 per added class, together
with stronger regularization. This is a disclosed constructed failure, not a
subtle incident discovered in a production model. The interventions measure
their effects; they do not uniquely identify a historical cause.

Whole-worker wall time was **52.5039 seconds**, process CPU **52.4450 seconds**,
and peak RSS **624,295,936 bytes**. These include loading, all 15 fits, saving and
report generation. Optimizers stopped after 12–39 iterations, below the fixed
300-iteration cap. Per-fit timing and complete metrics are in the
[result record](../research/INCREMENTAL_INTENT_RESULT_2026_10_09.json).

All 45,000 fitted-model predictions were independently reconstructed from saved
numeric coefficients, vocabulary and IDF without fitting again. The three MRF
reports were recomputed from raw execution records. Every class recall and raw
prediction remains in the retained bundle; the public JSON provides a compact
summary. Reused full-rollback/control predictions require no extra fitting, but
this does not measure deployment cost.

## Reproduce the bounded development experiment

Use the pinned source and extraction described in the
[prospective specification](../research/INCREMENTAL_INTENT_DEVELOPMENT_2026_10_09.md).
The upstream CLINC JSON bundles train, validation and test; those bytes were
downloaded to extract `train`. Only that extraction is supplied to this worker.
Do not claim the other splits were never downloaded. Banking77's official test
was not accessed.

```bash
python -m pip install -e . -r examples/banking-requirements.txt
python examples/incremental_intent_investigation.py \
  clinc150-training-only.json new-attempt-directory
```

The output directory must not exist. The supervisor stops after 900 seconds,
never retries and saves terminal CPU/RSS/wall measurements. There are at most
15 fits, one CPU thread and a 512 MiB output limit. Each scenario contains a
portable `index.html`, aligned raw records, explicit changes, model arrays and
text previews. Open the report locally; it needs no server or external model API.

CLINC150: Larson et al., *An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction*, EMNLP 2019, [official source](https://github.com/clinc/oos-eval),
CC BY 3.0. Changes here are the internal development split, added-class scenarios
and declared training-data/regularization changes.

## Decision

Keep full-data retraining as the reference for this release task. Do not spend
more compute tuning these constructed failures to make MRF look advantageous.
The next product test is whether an independent engineer can bring their own
saved predictions, understand the report and choose an appropriate intervention.
A research superiority claim still requires independently sourced incidents and
a prospective comparison against ordinary investigation at matched cost.
