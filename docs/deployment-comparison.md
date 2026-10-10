# Inspect a quantized classification release

This example contains actual saved predictions from float32 and dynamic INT8
versions of a retained linear Banking77 classifier. It needs no model download,
training, ONNX installation or network access after installing MRF.

The practical question is whether a smaller classifier preserves the declared
behavior. Overall accuracy alone can hide errors moving between classes.

## Rebuild the report

From the repository root, with Python 3.12 or later:

```bash
python -m pip install .
mrf-import examples/deployment_comparison/policy.json examples/deployment_comparison/predictions.csv deployment-investigation
```

Open `deployment-investigation/index.html`. The candidate passes the policy, but
its predictions are not identical. Filter to changed cases to inspect all eight;
filter to regressions to see three newly incorrect cases. Repairs are explicitly
unevaluated. A passing candidate is not a uniquely responsible cause or a claim
that no behavior changed.

Optional text previews can be added without running a model:

```bash
python -c "import shutil; shutil.copyfile('examples/deployment_comparison/case_inputs.json', 'deployment-investigation/case_inputs.json')"
python -m model_forensics.investigation_report deployment-investigation deployment-investigation/inspection.html
```

Open `inspection.html` to inspect the corresponding Banking77 messages and export
selected case IDs. Source integrity is checked when the report is rebuilt;
already-generated HTML is a snapshot, not a live integrity monitor.

## Measured outcome

All 1,969 existing training-partition development cases were used. The source
float64 classifier recovered the original retained labels exactly; float32 export
also preserved every label. INT8 changed eight predictions: three new errors,
three corrections and two changes between incorrect labels. All versions had
1,734 correct predictions (88.07%). No official Banking77 test examples were used.

The prospective policy allowed an overall accuracy drop of at most one percentage
point and a drop of at most five percentage points in each of the 77 classes.
Every limit passed. `unable_to_verify_identity` and `verify_my_identity` each lost
one correct result among 20 cases: exactly the five-percentage-point boundary.
These illustrative development thresholds do not establish production safety.

| Representation | Classifier file bytes | Batch-one end-to-end p50 | p95 |
| --- | ---: | ---: | ---: |
| Original float64 saved model | 5,069,209 | 0.648 ms | 1.485 ms |
| Float32 ONNX | 2,760,840 | 0.278 ms | 0.483 ms |
| Dynamic INT8 ONNX | 691,250 | 0.233 ms | 0.367 ms |

The shared fitted vectorizer is another 414,429 bytes, excluded above. The original
model uses compressed NPZ while exported models use ONNX; their file sizes are
not a precision-only compression experiment. Timing uses 500 calls per
representation/path on the first 100 lexicographically sorted case IDs, one core
and one numerical thread. Sparse and dense execution, row preparation and runtime
optimization differ. These are single-host observations, not portable speedup or
memory claims. ONNX Runtime 1.31.0, ONNX 1.23.2, opset 17 and IR 10 were used;
quantization was dynamic, QInt8, MatMul-only, per-tensor and full-range. Evaluation
and timing both used batch one because dynamic activation scaling is batch-dependent.

## Evidence and limits

The first worker saved all scores and models, then stopped because the ordinary
diff harness subtracted floating accuracy ratios at an exact threshold. For
example, 16/20 minus 15/20 became 0.050000000000000044. MRF's integer-count
difference correctly gives 1/20. The harness was corrected without changing
thresholds, labels or predictions. A separate continuation reused the hashed
models/scores and finished reports and latency; no export, quantization or full
evaluation was repeated.

Initial worker: 2.513 seconds wall, 2.492 process CPU seconds, 437,825,536 bytes
peak RSS. Continuation: 3.323 seconds wall, 3.286 process CPU seconds,
412,176,384 bytes peak RSS. Resources are separate wait4 records, not one
uninterrupted successful run. Installation and initial conversion durations were
not retained and remain unknown.

The public fixture replays policy decisions and case inspection, not model
production. Complete numerical inputs, graphs, raw scores, timings, both worker
records and manifests are retained in the maintainer's experiment archive,
SHA-256 `48b9ff62432964d67157e870953330740b90484eadf145597caafc3ab450860e`.
Independent execution of that producer has not been demonstrated.

An ordinary complete prediction diff with the same cases and policy reached the
same conclusions as MRF. No diagnostic advantage, independent-user benefit,
causal identification or paper-worthy discovery follows from this example.
Float32 is the conservative label-preserving export; INT8 offers a measured
size/latency tradeoff with explicit case errors. Independent first use is still
the next product-acceptance milestone.

## Attribution

Banking77: Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence
Encoders*. [Creator repository](https://github.com/PolyAI-LDN/task-specific-datasets),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: existing
training-partition development split, TF-IDF representation, float32 and dynamic
INT8 classification, saved predictions and case grouping. No external checkpoint
or new training was used. Dataset terms do not resolve the project's own source
distribution license.
