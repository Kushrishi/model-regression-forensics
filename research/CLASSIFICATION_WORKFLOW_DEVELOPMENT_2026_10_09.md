# Classification-release workflow development

Prospective specification — 9 October 2026, before the first fit.

The owner approved the consolidated application-development direction and asked
to proceed. This authorizes this bounded engineering demonstration. The closed
SmolLM2 attempt and completed localization study are unchanged. This is neither
independent scientific review nor a confirmatory attribution experiment.

## Task and data

Use only Banking77's official training CSV from PolyAI-LDN/task-specific-datasets
revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, SHA-256
`b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`.
Never fetch the official test split. This familiar training substrate supplies
development evidence, not newly independent confirmation.

Deduplicate whitespace-normalized, case-folded text and reject conflicting labels.
Within each of the 77 classes, sort by SHA-256 of that normalized text. The first
floor(20% of class count) records form development evaluation; the remainder train.
Keep original text for model inputs and display. Retain every selected identity.

## Fixed computation

One training-text TF-IDF vocabulary: word unigrams/bigrams, min_df=2,
max_features=20,000, sublinear_tf=True; scikit-learn defaults otherwise.
Five independent LogisticRegression fits: lbfgs, max_iter=300, tol=1e-4,
random_state=0, one CPU thread. No hyperparameter search.

| Execution | Training labels | C |
| --- | --- | ---: |
| Baseline | Original | 4.0 |
| Candidate | Swap card_arrival and card_delivery_estimate | 0.25 |
| Restore labels | Original | 0.25 |
| Restore regularization | Swapped | 4.0 |
| Restore both | Original | 4.0 |

An unchanged-candidate control reuses candidate predictions without a sixth fit.
All changes are disclosed, with changed training IDs. This deliberately includes
an obvious visible-label change: defeating simple diff inspection is not the aim.

The clean model must reach overall accuracy >=0.70, macro recall >=0.60 and
combined target-class accuracy >=0.50. Stop if it does not, or if any fit reaches
the iteration cap. These are engineering usefulness floors, not estimated
statistical significance. Do not retune this attempt after results.

The identical repair policy permits an accuracy drop from baseline of at most
0.01 overall, 0.01 across the other 75 classes, and 0.05 in each target class.
Also report all class metrics and changed cases. A policy pass is not unique
historical causation; a failed partial rollback is retained.

## Resource and artifact contract

CPU only, one thread, 900 seconds total process wall, maximum five fits, no retries
or paid services. Retain whole-worker peak RSS and process CPU; execution records
measure current-process CPU and wall per callback, including persistence checks.
Stop if the output bundle exceeds 200 MiB. No model binary belongs in Git.

Retain numeric coefficients/classes/intercepts, vocabulary and IDF without pickle;
recompute predictions from the saved coefficients before accepting each run.
Save input/split/source/environment identities, full release differences, raw
predictions, fit iterations, failures, text previews and portable HTML. The worker
uses a new directory and never replaces an old attempt.

Completion means an actual, reopenable training-release investigation with
individually executed rollback evidence. No superiority, causal-discovery,
generalization or human time-saving claim follows. A later comparative research
study requires its own prospective incidents, baselines and evaluation design.

Data attribution: Casanueva et al., *Efficient Intent Detection with Dual Sentence
Encoders*, NLP for ConvAI/ACL 2020. Banking77 is distributed under CC BY 4.0;
the retained example must identify its source and the disclosed modifications.

## Pre-fit metadata correction

The first invocation rejected the provisional label name card_delivery_tracking
before vectorization or any model fit. The pinned CSV uses card_delivery_estimate.
Source and specification were corrected before observing training outcomes; all
other settings remain fixed. The failed setup log and resource record are retained
in the original attempt directory. A separately named attempt executes the fits.
