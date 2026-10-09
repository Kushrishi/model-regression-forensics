# Ambiguous repair fixture

This fixture tests one narrow claim: restoring a failed behavior is not sufficient
to identify a unique historical cause.

A deterministic linear classifier is evaluated in three states:

1. the original model/input convention;
2. a regressed release that reverses feature order;
3. two distinct repairs:
   - restore the original feature order;
   - keep reversed inputs and reverse the model weights.

Both repairs recover the baseline predictions exactly. The repair assessment
therefore returns `ambiguous_repairs` and leaves the historical cause
`not_identified`.

Run:

```bash
uv run python examples/ambiguous_repairs.py
```

This small example checks how the assessment handles competing repairs. The fixture includes the
known historical change only in its printed truth block; the assessment function
never receives that truth.

## Evaluate saved repair predictions

`python -m model_forensics.repair_compare repairs.json` accepts a JSON object with
`regressed` and `repairs` fields. `regressed` is a release-comparison input;
`repairs` maps intervention names to release-comparison inputs. Each input contains
`cases`, `baseline`, `candidate` and `slices`, using the same schema as
`model_forensics.release_compare`. No model, training job or external service is
invoked. Redirect standard output to retain the assessment and every comparison.

All inputs must share the exact case IDs, expected labels, baseline predictions,
baseline release ID, slice membership and tolerances. Record ordering may differ.
The command recomputes the reports from validated predictions, checks an identity
digest of that common policy, and rejects mismatches. Comparing counts and baseline
accuracy alone is insufficient: two different test sets can share both numbers.

Exit code 0 means a valid assessment was produced, including an ambiguous result;
it does not certify a unique cause or approve a release. Invalid or incomparable
evidence exits with code 2. Successful repair IDs are explicit in the JSON.
Zero evaluated repairs yields `insufficient_evidence`; one successful repair still
leaves historical cause unidentified. Reports from the old comparison schema lack
the policy digest and must be regenerated from their original prediction records.

The digest binds supplied records, not their authenticity or a verified training
history. Distinct intervention names do not prove independent interventions.
Evaluation slices can overlap, and this command does not assess statistical
significance, training variability or causal identification. General model-quality
and drift reporting already exist in tools such as
[Evidently](https://docs.evidentlyai.com/quickstart_ml); this workflow's purpose is
to preserve comparable repair evidence for the project's narrower investigation.
