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
python examples/ambiguous_repairs.py
```

This is a falsification fixture for unique-attribution logic, not evidence that a
new causal-identification method has been established. The fixture includes the
known historical change only in its printed truth block; the assessment function
never receives that truth.
