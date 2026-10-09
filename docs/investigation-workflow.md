# Run and inspect a release investigation

An investigation executes a baseline, a candidate release and explicitly selected
repairs against the same cases. It retains their predictions and compares each
slice against its declared accuracy-drop tolerance. Multiple successful repairs
remain ambiguous: restoration does not identify a unique historical cause.

## Handwritten-digit walkthrough

From a checkout, using Python 3.12:

```sh
python -m pip install -e . scikit-learn==1.7.2 numpy==2.1.3
python examples/digits_investigation.py ./digits-investigation
```

Open `digits-investigation/index.html` in a browser. It contains the repair
assessment, per-digit comparisons, changed case predictions, measured execution
costs and declared changes. It needs no server, accounts or network connection.
The output directory must not already exist.

The example uses scikit-learn's bundled handwritten-digit data. A single logistic
classifier is fitted on 1,257 examples with a fixed stratified split, at most 1,000
optimizer iterations and one CPU thread. There is no search. All five inference
calls execute the model: baseline, reversed-feature candidate, two distinct
repairs and an unchanged faulty control. Repair predictions are not copied from
the baseline.

The two repairs remove the permutation or apply its inverse. Their agreement is
intentional. This is a disclosed input-processing fault for application
integration, not evidence that MRF discovered a hidden cause or beats another tool.
The evaluation set has 540 cases; it is not a fresh research confirmation set.

A retained run is in `examples/investigations/digits`. Its baseline accuracy was
98.15%; reversing the 64 feature columns reduced it to 37.41%. Both repairs
restored the baseline predictions; the unchanged control failed. The assessment
correctly retained ambiguity between the two repairs.

## Reopen retained results

```sh
python -m model_forensics.investigation_report ./digits-investigation ./reopened.html
```

The renderer recomputes comparisons from `plan.json` and indexed execution files.
It does not trust `report.json`, import code or execute model files. All supplied
text is escaped in HTML. Existing reports are not overwritten. The directory is
not a signed artifact: modified input records change the recomputed evidence.

`model.json` contains numeric scaler/classifier parameters rather than executable
pickle data. `plan.json` records dataset identity, split indices, dependency
versions, fit iterations and fitting wall time. Execution records retain each
prediction and function-call wall/current-process CPU time. These timings exclude
model fitting and do not include CPU consumed by child processes. The current
runner does not enforce timeouts or retry failed functions.

## Integrate a model

`model_forensics.investigation.run_investigation` accepts explicit Python
functions returning `Prediction` records. Supply cases, slices, baseline,
candidate, named repair functions and declared changes. Every function must
predict exactly the same case IDs. Completed execution files remain available if
a later function fails; the exception is propagated and `failed.json` records
which release failed. The report renderer currently requires a complete attempt.

This version provides a portable inspection report. Pretrained-model integration and a comparative training-regression study remain
development work. No human usability or time-saving claim is made.

## Inspect inputs and select cases

The report now shows all cases with baseline, candidate and repair predictions.
Use **Candidate outcome** to select changed predictions, regressions or
improvements; use search to narrow case IDs or labels. **Export visible case IDs**
downloads the current selection as JSON. Filtering operates entirely in the
browser and makes no network requests. Without JavaScript, all records remain
visible.

Optional `case_inputs.json` maps known case IDs to either `{"text": "..."}` or
`{"width": 8, "height": 8, "pixels": [...]}`. Grayscale pixels are row-major values
from 0 to 255, with dimensions at most 64 by 64. The complete preview file is
limited to 8 MB. Previews are supplied by the caller, not verified model inputs;
the digit example derives them directly from its evaluation arrays. The report
embeds the inputs: share it only when sharing those inputs is appropriate.
