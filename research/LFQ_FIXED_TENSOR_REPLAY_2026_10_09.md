# LFQ commitment-loss defect: fixed-tensor replay

9 October 2026. A CPU software reproduction under the existing deterministic-check
authorization. No optimizer, parameter update, model training, dataset, official
test split or downstream task evaluation is involved.

## Observed result

The [retained JSON](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.json) records twelve checks:
three immutable source versions, two fixed float32 tensors, each with and without
a mask. The [runner](../scripts/replay_lfq_defect.py) verifies source SHA-256 before
import, compares commitment loss and input gradients with a direct squared-error
oracle, and validates the observed repair sequence. Successful checks mean the
expected defect or repair was reproduced, not that every source version worked.

| Source version | Single batch, no mask | Single batch, mask | Two batches, no mask | Two batches, mask |
| --- | --- | --- | --- | --- |
| Reported source | Matches oracle | Wrong loss and gradient | RuntimeError | Wrong loss and gradient |
| First maintainer repair | UnboundLocalError | Matches oracle | UnboundLocalError | Matches oracle |
| Follow-up repair | Matches oracle | Matches oracle | Matches oracle | Matches oracle |

For the single-batch masked tensor, the reported commitment loss is 1.23851848,
against oracle 0.36444446. For the two-batch masked tensor it is 1.13370371,
against oracle 0.27444446. Both have maximum absolute gradient error 0.29629630.
Both maintainer repairs match the masked loss and gradient; the second also
restores the unmasked paths. Final-version gradient differences are zero on
these fixtures. Quantized outputs match the sign-code oracle in every successful
forward call.

The original entropy branch overwrites and reshapes the input subsequently used
for commitment loss. Broadcasting can therefore produce a finite but incorrect
loss rather than an exception. The first repair preserves that input, but leaves
the entropy input undefined when no mask is supplied. The second initializes it
for both paths. This independently reproduces the source-history explanation
recorded in the [pipeline screen](VNEXT_PIPELINE_SCREEN_2026_10_09.md).

## Exact scope

The fixtures have shapes `(1, 4, 3)` and `(2, 3, 3)`, codebook size 8, one codebook,
no projections, full entropy-token fraction and commitment weight 1. Entropy weight
is zero to isolate commitment loss; the entropy branch still executes. Entropy
correctness, fractional sampling, multiple codebooks, projections, distributed
execution and other shapes are not evaluated. The training-mode flag enables the
auxiliary-loss branch; this configuration has zero trainable parameters.

Python 3.12, PyTorch 2.5.1+cpu, einops 0.8.0 and NumPy 2.1.3 reconstruct an explicit
environment. This is not a claim to recover the reporter's complete historical
environment. Source file hashes, fixtures, warnings, exceptions and runner hash
are in the JSON. Timing excludes interpreter/import startup; process RSS includes
it. These resource measurements do not estimate the cost of training a model.

## Reproduce on Linux / Python 3.12

The upstream modules were inspected before execution. They remain external MIT
licensed sources by Phil Wang; no upstream module is vendored in this repository.
The commands download only the three pinned module files and their license.
Outputs go to a new local file rather than replacing the accepted record.
The runner now enforces that rule with exclusive creation, rejects incomplete or
duplicate check sets, and keeps validation active under optimized Python. The
retained JSON and its original runner hash/timings remain unchanged; these are
software safeguards, not additional training or task-performance evidence.

```bash
python3.12 -m venv /tmp/lfq-replay-env
/tmp/lfq-replay-env/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
/tmp/lfq-replay-env/bin/pip install einops==0.8.0 numpy==2.1.3
mkdir -p /tmp/lfq-replay-sources
for commit in 59a30b68a83be710638184764c025b54693c82cc ec2f4f610515c2b063442f6b725c8233e9ec700e ac5d63174dd234ab75259a68a4ab246774863f6e; do
  curl --fail --location --silent --show-error "https://raw.githubusercontent.com/lucidrains/vector-quantize-pytorch/$commit/vector_quantize_pytorch/lookup_free_quantization.py" -o "/tmp/lfq-replay-sources/$commit.py"
done
curl --fail --location --silent --show-error https://raw.githubusercontent.com/lucidrains/vector-quantize-pytorch/ac5d63174dd234ab75259a68a4ab246774863f6e/LICENSE -o /tmp/lfq-replay-sources/LICENSE
/tmp/lfq-replay-env/bin/python scripts/replay_lfq_defect.py --source-dir /tmp/lfq-replay-sources --output /tmp/lfq-replay-result.json
```

## Research decision

This lead now has independent numerical defect/repair evidence. It still lacks a
historical known-good/regressed trained-task pair, a pinned task/data workload,
repeated task-performance restoration and evidence that an expensive diagnostic
adds value over the visible diff and direct oracle. It remains a software
reproduction, not a qualified MRF training incident or causal-specificity result.
No training or larger search follows automatically. Retain the executable example
and reopen qualification only if concrete task evidence becomes available.
