# Investigate your own classification release

Use saved predictions from a baseline, a changed candidate, and proposed repairs.
MRF checks whether each repair restores your declared behavior and produces a
portable report with case filtering and ID export. It does not load a model,
train anything, infer execution costs, or establish the historical cause.

## Install and run

Python 3.12 or later, from a checkout of this repository:

```bash
python -m pip install .
mrf-import policy.json predictions.csv investigation
```

`investigation` must not already exist. Open `investigation/index.html` in your
browser. No server or network connection is needed to inspect it. The command
returns zero when import succeeds, even if every repair fails your policy; read
the assessment in the report or `report.json`. Invalid input returns a nonzero
status before creating the output directory.

## Supply the evaluation policy

`policy.json` has exactly these three top-level fields:

```json
{
  "cases": [
    {"case_id": "message-1", "expected": "delivery"},
    {"case_id": "message-2", "expected": "payment"}
  ],
  "slices": [
    {"name": "all messages", "case_ids": ["message-1", "message-2"],
     "maximum_accuracy_drop": 0.0}
  ],
  "declared_changes": {
    "candidate": "Changed training labels and regularization",
    "restore_labels": "Restored original training labels"
  }
}
```

Choose slices and acceptable drops before inspecting repair outcomes. Add
protected slices to prevent an improvement in one group hiding a regression in
another. A drop of `0.01` means one percentage point of accuracy. Slices can
overlap; they are not independent statistical tests. This interface evaluates
exact classification labels, not probability calibration, free-text generation,
ranking, or regression metrics.

## Supply predictions

`predictions.csv` has exactly the following columns (order may vary):

```csv
release_id,case_id,observed
baseline,message-1,delivery
baseline,message-2,payment
candidate,message-1,payment
candidate,message-2,payment
restore_labels,message-1,delivery
restore_labels,message-2,payment
```

Start with just the baseline and candidate rows when no repair is available.
The report shows whether the candidate passes the declared policy, identifies
changed cases, and explicitly says that repairs have not been evaluated. Both
passing and failing candidates are accepted in this initial comparison.

To evaluate repairs, add their predictions to the CSV and import into a new
output directory. In this repair stage, the candidate must fail at least one
declared policy requirement. A passing repair does not establish the cause.

`baseline` and `candidate` are reserved roles. Every other release ID is a named
repair; repairs are optional. Every release must contain exactly
one prediction for every case in the policy. Row order does not matter. IDs and
labels are compared exactly, including whitespace and case; use normal CSV
quoting for commas or newlines. Duplicate cases, missing predictions, unknown
cases, extra columns and blank fields are rejected.

Supply actual class labels, not probability-column indices. Record the label
mapping and prediction method in `declared_changes`, especially if they differ
between releases. Changed predictions alone do not prove a training defect.

Include a complete rollback as an ordinary comparator when it is feasible. A
passing repair is evidence of restored measured behavior, not evidence that MRF
found a unique cause or outperformed rollback. Record how predictions were
produced in `declared_changes`; those declarations are not independently verified.

## Keep and share the result

The output retains byte-for-byte input files and their SHA-256 hashes, validated
release records, evaluation policy, assessment, and HTML. Imported execution
costs remain unknown. All data stays local unless you share the directory.
Review case IDs, labels and declarations for private information before sharing;
the report and source files contain them.

To regenerate the report from validated records:

```bash
python -m model_forensics.investigation_report investigation reopened.html
```

The importer does not add input previews. The report still supports prediction
search, outcome filtering and exact case-ID export. Optional previews follow the
existing [investigation workflow](investigation-workflow.md); do not put confidential inputs in a public report.

## Verification on retained real-text records

On 2026-10-09 the importer was checked against the existing Banking77 development
investigation: 1,969 cases, six releases and 11,814 predictions. The complete
recomputed assessment and every release record matched the original investigation.
Only restoring both changed components passed its policy. No additional model
fits were performed. This verifies data interchange on that known example; it
is not independent first-use validation or evidence of comparative research value.

For releases that must retain a new capability, slices can also declare
`minimum_accuracy`. See [release acceptance requirements](release-acceptance.md)
for the combined policy, a runnable file-only example and fair-comparison limits.

## Try the initial comparison

This two-case synthetic example demonstrates the interface; it is not research
evidence. From a checkout with MRF installed:

```sh
mrf-import examples/initial_comparison/policy.json examples/initial_comparison/predictions.csv initial-review
```

Open `initial-review/index.html`. The candidate fails the declared policy, one
case changes from correct to incorrect, and repairs are explicitly unevaluated.
Use the outcome filter and export button to retain the affected case IDs.
