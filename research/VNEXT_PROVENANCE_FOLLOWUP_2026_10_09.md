# Bounded provenance follow-up — 9 October 2026 UTC

Discovery/source inspection only. No training, new model, official Banking77 test
access, counterfactual execution or threshold selection occurred. This follows
the [October 7 incident screen](VNEXT_INCIDENT_AUDIT_2026_10_07.md); it does not
reopen the closed v2 attempt or establish a vNext scientific contribution.

## Two retained DeepFD leads

The inspected artifact remains pinned to
`ArabelaTso/DeepFD@9fe089510cf10af35b0c2cba024d9732a1048df4`.
Original questions were inspected directly and through the public Stack Exchange
API. The API identifies the XOR question as CC BY-SA 3.0 and the masking question
as CC BY-SA 4.0. DeepFD's Apache-2.0 packaging is not a substitute for these
upstream attribution/license requirements. No underlying data were republished.

| Lead | Newly resolved provenance | Remaining qualification problem |
| --- | --- | --- |
| [31556268: XOR](https://stackoverflow.com/questions/31556268/how-to-use-keras-for-xor) | The original question fits all four points. DeepFD `origin.py` adds an unseeded `train_test_split(test_size=0.2)`; its repair removes that split as well as changing loss, optimizer and epochs. Thus the collection's origin is not a verbatim historical original and the pair changes the training population. | A learning/repair example is not an established known-good/regressed release pair. No complete historical candidate changes or repeated restoration distribution are supplied. Reject current benchmark qualification; preserve as a repair/provenance example. |
| [59282996: masking](https://stackoverflow.com/questions/59282996/zero-predictions-despite-masking-support-for-zero-padded-mini-batch-lstm-trainin) | The original report specifies TensorFlow 1.13.1 and an explicit per-batch training loop. DeepFD instead fits the complete batch array in one call. The original answer discusses reshape versus transpose and the meaning of propagated masks. The collection's repair variants separately shift labels and reduce output classes. | No verified historical good release, complete release diff, controlled evaluation or measured restoration equivalence exists in these inspected materials. A repair variant without label shifting retains class 2 with two output classes: source inspection exposes a class-domain incompatibility, not an empirically verified repair. Reject current benchmark qualification; no fit is justified. |

Exact inspected blobs:

| Subject | File | Git blob |
| --- | --- | --- |
| 31556268 | origin.py | `0e91cde221ecf16a394ca223184746275d733c99` |
| 31556268 | repair.py | `4b67f921f8fb930ec2bee31e4b329c219349461a` |
| 59282996 | origin.py | `63955f1fa2984ae32aac2d1642d16759a2b53483` |
| 59282996 | repair.py | `b322c2c38f089a57af804fd31ac3907de7bf440b` |
| 59282996 | repair1.py | `79766cf2b5d378afbcb5aa89bcc0f8d3f80d81f7` |
| 59282996 | repair2.py | `a83ddcce9a49f1e31a08ceddc49944d5b6441a09` |

The class-domain observation is static: two class outputs admit sparse labels
0 and 1, whereas repair2 retains non-padding label 2. It is not a claim about a
measured exception in a reconstructed historical runtime. Likewise, source
differences are not independent reproduction of the reported accuracies.

## Collection identity clarification

[Zenodo record 21782307](https://zenodo.org/records/21782307) was resolved via its
public API. It is **AIFaultBench: A Reproducible Benchmark of Real-World AI Software
Faults**, not evidence that the unavailable DLFaultBench record is equivalent.
Metadata declares CC BY 4.0 and lists `mehilshah/AIFaultBench-v1.4.zip`,
18,485,538 bytes, checksum `md5:894a9211d03a20bf1bdac502c3bf0258`.
The archive was not downloaded; this checksum is provider metadata, not a local
SHA-256 verification. No new incident is qualified by resolving a collection name.

## Decision

The original 12-row screen still has **zero qualified incidents**. Both retained
leads now have additional provenance evidence but do not survive the release-case
gate. This finite follow-up is not an exhaustive search or a prevalence estimate.

MRF v1 remains completed bounded negative evidence; v2 remains closed; vNext
remains a maintained discovery/infrastructure track. An externally grounded case
or passive submission can justify another bounded assessment. No new training
proposal, artificial incident construction or model escalation follows here.
