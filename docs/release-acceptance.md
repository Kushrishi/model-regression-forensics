# Require a repair to preserve the release's useful behavior

Restoring an old model can remove a regression while also removing a capability
that the new release was intended to add. MRF can now check both requirements:

- A maximum accuracy drop relative to the baseline protects existing behavior.
- An optional minimum accuracy requires an absolute level of behavior on a slice.

A release passes only when it satisfies every declared requirement. An old
baseline is a reference, not automatically an acceptable deployment.

For example, a classifier adding a new intent may need to retain its old intents
and meet a minimum accuracy on the new intent. Add `minimum_accuracy` to that
slice in the JSON policy consumed by `mrf-import`:

```json
{
  "name": "Required new intent",
  "case_ids": ["new-intent-1", "new-intent-2"],
  "maximum_accuracy_drop": 0.0,
  "minimum_accuracy": 0.8
}
```

Accuracy is the fraction of correct exact-label predictions on the supplied
cases. A minimum of `0.8` means 80%; the threshold is inclusive. It is not a
confidence bound or a guarantee on unseen cases. Small slices provide coarse
measurements and weak evidence. The minimum and allowed drop both apply; neither
replaces the other. Omit the minimum or set it to null to retain the previous
relative-drop-only behavior.

## Try the software contract without training

From an installed checkout:

```bash
mrf-import examples/release_acceptance/policy.json \
  examples/release_acceptance/predictions.csv /tmp/release-acceptance
```

The four-case fixture uses hand-authored labels, not predictions from a trained
model. It intentionally covers three decisions: the candidate fails existing
behavior, complete rollback fails the new-capability requirement, and a targeted
repair meets both. The 100% fixture threshold tests boundary handling; it is not
a recommended production threshold or evidence that a real repair succeeds.

Open `/tmp/release-acceptance/index.html`. The slice tables show release accuracy,
maximum drop, minimum accuracy, whether the minimum was met, and the combined
policy decision. `report.json` exposes each condition separately. Execution cost
is unknown because these records were imported. The report still does not infer
a unique historical cause.

## Use this fairly in an evaluation

Choose requirements before observing candidate or repair outcomes, based on the
release's actual intended capabilities. Do not add a requirement after seeing
that complete rollback wins. Include complete rollback and ordinary targeted
rollback as comparators and give them the same visible change history and data.
If complete rollback meets the real requirements at lower cost, report that.

The policy digest includes the minimum accuracy. A repair assessed with a changed
minimum cannot be compared as if it used the original policy. Existing inputs
without minima retain their report fields, schema version and policy digests;
inputs with minima use release-comparison-pilot/0.3. This software capability
supports a future prospective evaluation. It is not that evaluation and provides
no comparative advantage claim.
