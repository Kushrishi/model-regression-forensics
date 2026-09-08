# Experiment 007 Results — Sensitivity-Calibrated Causal RCA

Status: **completed — calibration negative result; certification not opened**

## Confirmatory outcome

Experiment 007 executed the prospectively frozen target-dose calibration grid
without changing the model, optimizer, task, behavioral thresholds, world
counts, target slice, protected slices, or certification evaluation family.

The clean calibration baseline achieved 1.0 accuracy on every semantic slice
and 96/96 overall.

### Target dose 9

Neither calibration world passed the localized-regression gate.

World 0:

- aggregate accuracy: 0.40625;
- `triangle_large`: 0.125;
- `triangle_small`: 0.3125;
- `circle_small`: 0.0625;
- `circle_large`: 0.0625;
- `square_small`: 1.0;
- `square_large`: 0.875;
- target regression: 0.875;
- localized-regression gate: failed because protected-slice drift exceeded
  0.05.

World 1:

- aggregate accuracy: 0.3333333333333333;
- `triangle_large`: 0.0;
- `triangle_small`: 0.0;
- `circle_small`: 0.0;
- `circle_large`: 0.0;
- `square_small`: 1.0;
- `square_large`: 1.0;
- target regression: 1.0;
- localized-regression gate: failed because protected-slice drift exceeded
  0.05.

Formal dose-9 calibration result: **0/2 worlds passed; dose 9 did not qualify.**

### Target dose 18

Because dose 9 did not qualify, dose 18 was authorized by the frozen protocol.

The deterministic clean sibling was reused only after an exact reuse audit
verified identical clean training bytes, identical calibration evaluation
bytes, identical model/training configuration and seed, identical epoch-loss
trajectory across two independently trained baselines, and byte-identical
trained adapter weights.

Both dose-18 calibration worlds produced the same result:

- aggregate accuracy: 0.3333333333333333;
- `triangle_large`: 0.0;
- `triangle_small`: 0.0;
- `circle_small`: 0.0;
- `circle_large`: 0.0;
- `square_small`: 1.0;
- `square_large`: 1.0;
- target regression: 1.0;
- localized-regression gate: failed because protected-slice drift exceeded
  0.05.

Formal dose-18 calibration result: **0/2 worlds passed; dose 18 did not qualify.**

### Frozen stopping rule

No allowed target dose qualified.

Experiment 007 therefore stopped at calibration as required by the prospective
protocol.

The grid was not extended. Behavioral thresholds were not weakened. No
calibration-selection artifact was created.

Certification evaluation remained untouched. Private causal certification,
restoration training, order robustness, blinded RCA, truth reveal, and
diagnosis-driven recovery were not run.

## Post-hoc construction audit

After the confirmatory calibration result was closed, a read-only audit of the
already-generated model-facing baseline and candidate training files was
performed to explain the repeated non-local failure pattern.

The audit reconstructed the selected semantic slice directly from each prompt
and counted actual response-label changes.

| Dose | Slice | Changed labels | Fraction |
| --- | --- | ---: | ---: |
| 9 | `triangle_large` | 9/48 | 18.8% |
| 9 | `circle_small` | 37/48 | 77.1% |
| 9 | `circle_large` | 37/48 | 77.1% |
| 9 | `triangle_small` | 37/48 | 77.1% |
| 9 | `square_small` | 30/48 | 62.5% |
| 9 | `square_large` | 30/48 | 62.5% |
| 18 | `triangle_large` | 18/48 | 37.5% |
| 18 | `circle_small` | 34/48 | 70.8% |
| 18 | `circle_large` | 34/48 | 70.8% |
| 18 | `triangle_small` | 34/48 | 70.8% |
| 18 | `square_small` | 30/48 | 62.5% |
| 18 | `square_large` | 30/48 | 62.5% |

Every candidate training dataset contained 180 changed labels out of 288
training records.

At dose 9, each protected ACCEPT slice received 4.111 times as many direct
label changes as the `triangle_large` target. At dose 18, each received 1.889
times as many.

This reveals a structural conflict in the benchmark construction: the candidate
training set directly and heavily corrupted behaviors that the candidate gate
simultaneously required to remain within 0.05 of the clean baseline.

This audit is explanatory and post-hoc. It does not revise the frozen
confirmatory outcome.

## Conclusion

Experiment 007 solved the Experiment 006 materiality problem but did not solve
the locality problem.

The allowed corruptions reproducibly produced large target regressions, but no
prospective dose produced the localized behavioral regression required before
causal certification.

The result should therefore be interpreted as a **benchmark-construction
negative result**, not as evidence that the blinded forensic attribution
methods failed. Those methods were never run in Experiment 007.

A subsequent benchmark should validate its complete model-facing semantic
intervention matrix before any result-bearing training and ensure that nuisance
candidate changes are causally compatible with the protected-behavior contract.
