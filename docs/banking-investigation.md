# Investigate a changed text-classification release

This example trains a Banking77 intent classifier, changes two parts of its
training release, and evaluates separate rollbacks. It exposes both the target
failures and regressions elsewhere. The changes are deliberately constructed and
fully visible; this is a development walkthrough, not a causal-discovery benchmark.

## Observed result

The fixed training-only split contains 8,030 training and 1,969 development
examples across 77 intents. The two target intents have 52 development examples.

| Execution | Overall accuracy | Target accuracy | Other 75 intents | Repair policy |
| --- | ---: | ---: | ---: | --- |
| Clean baseline | 88.07% | 86.54% | 88.11% | Reference |
| Changed labels and regularization | 76.94% | 7.69% | 78.82% | Fails |
| Restore labels only | 78.62% | 71.15% | 78.82% | Fails |
| Restore regularization only | 86.03% | 9.62% | 88.11% | Fails |
| Restore both | 88.07% | 86.54% | 88.11% | Passes |
| Unchanged-candidate control | 76.94% | 7.69% | 78.82% | Fails |

Overall accuracy alone hides the remaining target failure after restoring
regularization. Restoring labels repairs card_arrival but not the complete target
and protected-behavior policy. Both changes must be rolled back among the tested
interventions. The report calls this one supported repair; it does not identify a
unique historical cause or rule out untested repairs.

Five fits completed in 13.661 seconds whole-worker wall, 13.612 process CPU
seconds, with 388,308,992 bytes peak child RSS on the tested Linux host. The
unchanged control reused candidate predictions. No official test data, GPU, paid
service or hyperparameter search was used. All 11,814 saved predictions across
six executions were independently reproduced from saved vocabulary/IDF and
numeric coefficients, without another fit.

The [prospective design](../research/CLASSIFICATION_WORKFLOW_DEVELOPMENT_2026_10_09.md)
and [exact result](../research/CLASSIFICATION_WORKFLOW_RESULT_2026_10_09.json)
record the policy, limits, source commit and a setup failure that occurred before
any fit. Historical studies are unchanged.

## Run locally

Supported supervisor platforms: Linux and macOS; Python 3.12. From the repository:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . -r examples/banking-requirements.txt
curl --fail --location --output banking77-train.csv \
  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv
python examples/banking_investigation.py banking77-train.csv /tmp/banking-investigation
```

Choose a new output directory. The runner verifies the exact training CSV hash,
uses one CPU thread and supervises a single worker for at most 900 seconds. It
runs the declared five fits without retries. A failed clean baseline or iteration
cap stops the remaining fits. The official test file is not downloaded.

Open `/tmp/banking-investigation/investigation/index.html`. Filter regressed cases,
search by text or case ID, inspect baseline/candidate/repair predictions and export
selected IDs. Open the slice tables to compare target and protected behavior.

The attempt retains raw executions, numeric models, vocabulary, IDF, source/data
identities, full declared changes, text previews, fit diagnostics and a terminal
resource record. Models use NPZ numeric arrays, not executable pickle. Preserve
the whole directory if you need future readback; a screenshot or aggregate alone
is insufficient. The saved HTML can be regenerated without training:

```bash
python -m model_forensics.investigation_report \
  /tmp/banking-investigation/investigation \
  /tmp/banking-investigation/reopened.html
```

The text previews derive from Banking77, distributed under CC BY 4.0. Cite
Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP
for ConvAI/ACL 2020. The source data, development split and label modifications
are recorded in the bundle. Do not treat this familiar development split as
independent research confirmation or compare its accuracy with official-test
leaderboard results.
