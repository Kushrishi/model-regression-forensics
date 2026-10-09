# Incremental intent release: prospective development experiment

Recorded 9 October 2026 before model fitting. Owner authorization: proceed with
substantive work across MRF, TM and ASL. This is a bounded CPU development
experiment, not a reopening of closed model-training studies or a claim of
scientific superiority.

## Decision
Can a release add 30 supported intents while retaining the accuracy of 120
existing intents? Compare ordinary interventions transparently. MRF records and
audits the comparison; it is not itself a repair algorithm. A complete rollback
is evaluated, even though it cannot predict labels outside its original classes.
Full-data retraining is a strong, ordinary baseline, not a novel MRF method.
A successful result supports a working release-investigation application. A
failed result limits this recipe. Neither establishes user time savings.

## Sources and boundaries
CLINC150, Larson et al., EMNLP 2019, official repository
https://github.com/clinc/oos-eval at 828f8093932c8fe6ca7936c3d2e52903b1c523de.
License: CC BY 3.0. Full JSON SHA-256:
36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0.
The upstream file bundles train/validation/test; it was downloaded and parsed
to extract only the train key. The validation/test records are not used,
displayed, profiled or scored. Do not describe their bytes as never downloaded.
Compact UTF-8 training-only JSON plus newline SHA-256:
e7dab3915557d582f1af3c2aed5c1f87c4e6e576fff3c93b001fb6d35de46e64.
15,000 training rows, 150 classes. Banking77's official test remains untouched.
Use only this extracted training file in the worker.

Normalize casefolded whitespace for identity. Collapse duplicate identical
text/label rows, exclude every normalized text with conflicting labels, and
report these counts. Per class, sort normalized-text SHA-256; first floor(n/5)
are development evaluation, the remainder train. This is not official test
performance. Save IDs and labels for all partitions.

Three predefined class orders: seeds 0,1,2, sorting SHA-256 of
"seed:label"; first 30 classes are new, remaining 120 existing. These are
overlapping development scenarios, not independent statistical replications.
No choosing a favorable order after outcomes.

## Fixed recipes
One TF-IDF vocabulary per scenario fitted on existing-intent training text only:
word (1,2)-grams, min_df=2, max_features=20000, sublinear_tf=True.
Never fit vocabulary or IDF on evaluation data or new-intent training text.
LogisticRegression: lbfgs, max_iter=300, tol=1e-4, random_state=0, one thread,
remaining scikit-learn 1.7.2 defaults. All training starts from scratch;
this is a release-data/regularization comparison, not neural continual fine-tuning.

| Model | Existing training data | New training data | C |
| --- | --- | --- | ---: |
| Baseline | All | None | 4 |
| Candidate | First floor(n/5) per existing class by ID | All | 0.25 |
| Restore data retention | All | All | 0.25 |
| Restore regularization | Same reduced old set | All | 4 |
| Restore both / full-data retrain | All | All | 4 |

Exactly five fits per scenario, at most 15 total. Full rollback reuses baseline;
unchanged control reuses candidate. Both incur zero additional training.
Do not imply they have zero deployment cost. Each intervention has complete
declared data identities and configuration; there is no concealed cause.

## Acceptance and reporting
Before results, set an engineering target of >=70% accuracy on added intents,
and at most 2 percentage points loss on existing intents versus baseline.
These are proposed development requirements, not customer requirements or
statistical significance thresholds. They are deliberately not tuned to outcomes.
The existing-intent baseline must reach 70%; otherwise stop all later fitting.
If any optimizer reaches 300 iterations, retain results but stop subsequent fits.
Report old/new/overall and every class recall for every model, iterations,
wall/process CPU, peak worker RSS, changed predictions, and all repair verdicts.
Report data/regularization interventions individually even when unsuccessful.
No post-hoc thresholds, retuning, further seeds, retries or new representations.

The actual comparison is interventions under this task contract. It does not
measure MRF against a human investigator or demonstrate novel causal discovery.
If ordinary full retraining is the only acceptable repair, say so plainly.
If none passes, keep the negative result and stop this recipe.

## Resources and retention
One worker, one CPU thread, 900-second whole-worker cap, at most 15 fits.
No GPU, external inference, purchased compute or background campaign.
512 MiB output limit checked after each fit; no overwrite.
Save plan/source/environment hashes before fitting, each model's coefficient
arrays without pickle, vocabulary/IDF, raw case predictions and numeric
readback verification. Preserve partial outputs and terminal failure/resource
record on interruption. Produce MRF's portable report for each completed
scenario. Archive raw training input, license, code, protocol and results.

## Comparable work
Larson et al. (2019): https://aclanthology.org/D19-1131/ provides the dataset,
not evidence for our release recipe.
Paul et al. (2022): https://aclanthology.org/2022.evonlp-1.4/ studies adding
intents with limited old data. It motivates the operational question; our
linear recipe does not reproduce its method or benchmark.
Zheng et al. (2024): https://aclanthology.org/2024.acl-long.794/ demonstrates
why strong simple comparisons matter in incremental learning. Its pretrained
language-model results must not be transferred to this TF-IDF experiment.

