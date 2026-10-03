# Release comparison pilot

This experimental utility compares exact-label predictions on explicitly declared
evaluation slices. It does not train a model, rank responsible changes, assess
intervention outcomes, certify a cause or establish statistical significance.
Existing scientific protocols and results are unchanged.

## One task

Detect a slice regression that is hidden by unchanged overall accuracy. In the
small synthetic example, one case gets worse and another improves. The report
retains both, evaluates each slice's declared tolerance and identifies its input.

From a checkout with the package installed:

```sh
python -m model_forensics.release_compare examples/release_comparison.json
```

Exit status is 0 when all declared tolerances pass, 1 for a reported regression,
and 2 for invalid input. The example intentionally returns 1. JSON is written to
standard output; input files are not changed. No model, network or API key is used.

For use outside this checkout, install the built wheel and supply your own JSON
record using the example's schema. Inputs require unique case IDs, complete aligned
predictions for both releases, different release IDs and nonempty named slices.
Unknown fields and missing/duplicate identities are rejected. Labels are compared
exactly. Overlapping slices are allowed, but their counts cannot be pooled as
independent observations. Tolerances should be chosen before inspecting results.

`input_sha256` hashes canonical supplied records, including tolerances. It does
not authenticate a training run, identify its dataset or prove provenance of
user-supplied predictions. Sorting input records does not change report identity.

## Remaining usefulness gate

This is the first clean-room package task, not an established differentiated
product. General model comparison already exists in tools such as Deepchecks.
Before expanding the API, compare an external release task against existing
tools. Candidate-change manifests and paired intervention evidence may add value,
but are not implemented here. No causal claim follows from prediction differences.

This pilot does not add an open-source license or authorize public package-index
distribution. Reuse/distribution rights must be settled before a reusable release.
