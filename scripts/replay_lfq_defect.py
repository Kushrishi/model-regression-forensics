"""Replay a pinned LFQ defect with fixed tensors; never update parameters.

Source files are supplied locally and verified before importing. This is a
CPU unit reproduction, not historical training or downstream task evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
import warnings
from pathlib import Path

import einops
import numpy
import torch

SOURCES = {
    "reported": (
        "59a30b68a83be710638184764c025b54693c82cc",
        "9c1e6aad8a007aa7055ad50dee5a0b09ead2a2bf48d27508d64549741a640411",
    ),
    "first_repair": (
        "ec2f4f610515c2b063442f6b725c8233e9ec700e",
        "6edc22f4cc271ce50c25049c2c02a914a1ec32369bfc3c17a4b292e4f175368d",
    ),
    "followup_repair": (
        "ac5d63174dd234ab75259a68a4ab246774863f6e",
        "d50257c10506aea87569625206f27f5fcc83ad6df16cbb36e2554de625410abf",
    ),
}

FIXTURES = {
    "single_batch": {
        "values": [[[0.2, -0.3, 0.4], [0.5, -0.6, 0.7], [-0.8, 0.9, -0.2], [0.3, -0.4, 0.5]]],
        "mask": [[True, False, True, True]],
    },
    "two_batches": {
        "values": [
            [[0.2, -0.3, 0.4], [0.5, -0.6, 0.7], [-0.8, 0.9, -0.2]],
            [[0.3, -0.4, 0.5], [-0.6, 0.7, -0.8], [0.9, -0.2, 0.3]],
        ],
        "mask": [[True, False, True], [False, True, False]],
    },
}


def load_source(directory: Path, name: str):
    commit, expected = SOURCES[name]
    path = directory / f"{commit}.py"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Source digest mismatch: {name}")
    spec = importlib.util.spec_from_file_location(f"lfq_{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import verified source: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(module, fixture: dict, masked: bool) -> dict:
    values = torch.tensor(fixture["values"], dtype=torch.float32)
    mask = torch.tensor(fixture["mask"], dtype=torch.bool) if masked else None
    oracle_input = values.clone().requires_grad_(True)
    quantized = torch.where(oracle_input > 0, 1.0, -1.0).detach()
    squared = (oracle_input - quantized).square()
    oracle_loss = (squared[mask] if masked else squared).mean()
    oracle_loss.backward()
    x = values.clone().requires_grad_(True)
    lfq = module.LFQ(
        dim=3,
        codebook_size=8,
        has_projections=False,
        channel_first=False,
        commitment_loss_weight=1.0,
        entropy_loss_weight=0.0,
        frac_per_sample_entropy=1.0,
    )
    # The training flag selects the auxiliary-loss branch, not a fit. This LFQ
    # configuration has no trainable parameters and no optimizer is constructed.
    lfq.train()
    if sum(parameter.numel() for parameter in lfq.parameters()) != 0:
        raise ValueError("The fixed-tensor replay requires zero trainable parameters")
    result = {"masked": masked, "oracle_commitment": oracle_loss.item()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ret, breakdown = lfq(x, mask=mask, inv_temperature=1.0, return_loss_breakdown=True)
            ret.entropy_aux_loss.backward()
            result.update(
                {
                    "status": "returned",
                    "commitment": breakdown.commitment.item(),
                    "loss_matches_oracle": torch.allclose(
                        breakdown.commitment, oracle_loss.detach(), atol=1e-7, rtol=1e-6
                    ),
                    "gradient_matches_oracle": torch.allclose(
                        x.grad, oracle_input.grad, atol=1e-7, rtol=1e-6
                    ),
                    "max_gradient_error": (x.grad - oracle_input.grad).abs().max().item(),
                    "quantized_matches_oracle": torch.equal(ret.quantized, quantized),
                    "gradient_finite": bool(torch.isfinite(x.grad).all()),
                }
            )
        except (RuntimeError, UnboundLocalError) as exc:
            result.update(
                {"status": "exception", "exception_type": type(exc).__name__, "exception": str(exc)}
            )
        result["warnings"] = [str(warning.message) for warning in caught]
    return result


def validate(rows: list[dict]) -> None:
    expected = {
        (version, fixture, masked)
        for version in SOURCES
        for fixture in FIXTURES
        for masked in (False, True)
    }
    actual = [(row.get("version"), row.get("fixture"), row.get("masked")) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Replay requires exactly the twelve declared checks without duplicates")
    for row in rows:
        if row["version"] == "followup_repair":
            valid = (
                row.get("status") == "returned"
                and row.get("loss_matches_oracle") is True
                and row.get("gradient_matches_oracle") is True
            )
        elif row["version"] == "first_repair":
            if row["masked"]:
                valid = (
                    row.get("status") == "returned"
                    and row.get("loss_matches_oracle") is True
                    and row.get("gradient_matches_oracle") is True
                )
            else:
                valid = (
                    row.get("status") == "exception"
                    and row.get("exception_type") == "UnboundLocalError"
                )
        elif row["masked"]:
            valid = (
                row.get("status") == "returned"
                and row.get("loss_matches_oracle") is False
                and row.get("gradient_matches_oracle") is False
            )
        elif row["fixture"] == "single_batch":
            valid = (
                row.get("status") == "returned"
                and row.get("loss_matches_oracle") is True
                and row.get("gradient_matches_oracle") is True
            )
        else:
            valid = row.get("status") == "exception" and row.get("exception_type") == "RuntimeError"
        if row.get("status") == "returned":
            valid = valid and row.get("quantized_matches_oracle") is True
            valid = valid and row.get("gradient_finite") is True
        if not valid:
            raise ValueError(
                f"Unexpected replay outcome: {row['version']}/{row['fixture']}/{row['masked']}"
            )


def write_report(path: Path, result: dict) -> None:
    """Never replace an accepted report, including a collision after preflight."""
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; use a new report path")
    if torch.__version__ != "2.5.1+cpu" or einops.__version__ != "0.8.0":
        parser.error("Use the pinned CPU dependencies in the replay README")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    rows = []
    for name in SOURCES:
        module = load_source(args.source_dir, name)
        for fixture_name, fixture in FIXTURES.items():
            for masked in (False, True):
                rows.append(
                    {"version": name, "fixture": fixture_name, **check(module, fixture, masked)}
                )
    validate(rows)
    result = {
        "schema": "lfq-fixed-tensor-replay-v1",
        "scope": "CPU commitment-loss unit reproduction; no historical task claim",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "einops": einops.__version__,
            "numpy": numpy.__version__,
            "platform": platform.system(),
            "device": "cpu",
            "threads": 1,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {
            name: {"commit": commit, "file_sha256": digest}
            for name, (commit, digest) in SOURCES.items()
        },
        "fixtures": FIXTURES,
        "configuration": {
            "dim": 3,
            "codebook_size": 8,
            "has_projections": False,
            "commitment_loss_weight": 1.0,
            "entropy_loss_weight": 0.0,
            "frac_per_sample_entropy": 1.0,
            "inverse_temperature": 1.0,
        },
        "optimizer_updates": 0,
        "trainable_parameters": 0,
        "rows": rows,
        "checks_passed": True,
        "resources": {
            "replay_wall_seconds": time.perf_counter() - start_wall,
            "replay_cpu_seconds": time.process_time() - start_cpu,
            "process_peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_units": "bytes" if sys.platform == "darwin" else "KiB",
            "note": "Replay times exclude imports; RSS includes interpreter/imports",
        },
    }
    try:
        write_report(args.output, result)
    except FileExistsError:
        parser.error("Output already exists; use a new report path")
    print(f"Validated {len(rows)} fixed-tensor checks; optimizer updates: 0")


if __name__ == "__main__":
    main()
