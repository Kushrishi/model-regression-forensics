# External release-comparison task

October 3, 2026. Engineering fixture, not a new MRF research result.

## Task and result

A standardized logistic classifier is trained on 1,257 examples from scikit-learn's
bundled handwritten digits dataset. A fixed seed and stratified split reserve
540 examples. The candidate deliberately reverses the 64 input feature positions
at inference. Slices are the overall cohort and all ten ground-truth digit classes;
every accuracy-drop tolerance is zero. No slice is selected from observed failures.

Baseline accuracy is 98.15%; candidate accuracy is 37.41%. All eleven slice
reports agree with an independent NumPy calculation of accuracy and changed-case
identities. Restoring the feature order makes every slice pass. The utility does
not discover the bug: it was supplied by the fixture author. This is a software
workflow check on external data, not independent causal certification or evidence
of generalization from Banking77. No model weights or digit images are redistributed.

```sh
# Requires scikit-learn and NumPy in the example environment, plus the comparator.
python examples/digits_release_task.py
python -m model_forensics.release_compare examples/digits_release_task/input.json
```

The second command intentionally returns 1. The input, report and provenance are
retained under `examples/digits_release_task/`. Scikit-learn 1.8.0 and NumPy 2.3.5
produced this record. Dataset and test-index hashes are recorded; the comparator
also hashes the supplied predictions and tolerance policy. These hashes do not
authenticate an independently executed training run.

Dataset source: [scikit-learn load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

## What this says about continuation

The adapter is feasible and the report arithmetic transfers to an external task.
That is insufficient differentiation. The independent NumPy reference computes
the same metrics with little code. The utility adds strict identity validation,
deterministic records and command-line policy handling, not a new diagnostic method.

Deepchecks already documents model-performance comparison and segment evaluation:
[official model-evaluation catalogue](https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/index.html).
This was a documentation comparison, not a runtime benchmark or usability study.
No advantage over Deepchecks or FiftyOne has been demonstrated.

The follow-up [ambiguous-repair example](ambiguous-repairs.md) tests whether two different interventions can recover the same behavior. It reports ambiguity when both pass. A broader study would need a task with plausible competing explanations and a comparison against existing methods using the same evidence.

The digits example supports the software workflow. It adds no matched-world restoration result or new transformer training; the research decision remains described in [research status](../research/STATE.md).
