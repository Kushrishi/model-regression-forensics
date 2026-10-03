from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REQUEST_PATH = "research/M4_EXECUTION_REQUEST.json"
PROTOCOL_PATHS = (
    "research/M4_LOCALIZATION_BASELINE_PROTOCOL.md",
    "research/M4_LOCALIZATION_BASELINE_PROTOCOL_AMENDMENT_1.md",
    "research/ATTRIBUTION_TARGET.md",
    "research/GRAD_DOT_BASELINE.md",
    "experiments/009_stochastic_counterfactual_certification/STRUCTURALLY_MATCHED_BENCHMARK_PROTOCOL.md",
    "experiments/009_stochastic_counterfactual_certification/STRUCTURALLY_MATCHED_BENCHMARK_PROTOCOL_AMENDMENT_1.md",
    "experiments/009_stochastic_counterfactual_certification/MATCHED_BENCHMARK_PREFLIGHT_RESULT.json",
    "uv.lock",
)
METHODS = ("B0_RANDOM", "B1_LABEL_OVERLAP", "B2_LEXICAL_OVERLAP", "B3_GRAD_DOT", "B4_TRACIN")


def verify_authorization(root: Path) -> dict:
    request = json.loads((root / REQUEST_PATH).read_text())
    if request.get("schema_version") != 1 or request.get("milestone") != "M4_LOCALIZATION":
        raise ValueError("invalid M4 execution request")
    if request.get("official_test_authorized") is not False:
        raise ValueError("official test access is unauthorized")
    if request.get("restoration_authorized") is not False:
        raise ValueError("M5 restoration is unauthorized")
    if request.get("trajectories") != [0, 1, 2] or request.get("worlds") != [0, 1]:
        raise ValueError("frozen execution matrix drift")
    source = request["source_git_sha"]
    if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("source must be a full Git SHA")
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], cwd=root, check=True)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", source, "HEAD"], cwd=root, text=True
    ).splitlines()
    if set(changed) - {REQUEST_PATH}:
        raise ValueError(f"execution differs from source pin: {changed}")
    dirty = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"], cwd=root, text=True
    ).splitlines()
    if dirty:
        raise ValueError(f"tracked execution files are dirty: {dirty}")
    evidence = request.get("successful_source_ci", [])
    if {r.get("name") for r in evidence} != {"CI", "Research CI"}:
        raise ValueError("both source CI records are required")
    if any(r.get("head_sha") != source or r.get("conclusion") != "success" for r in evidence):
        raise ValueError("source CI identity or conclusion drift")
    if set(request["protocol_sha256"]) != set(PROTOCOL_PATHS):
        raise ValueError("required protocol identity set drift")
    for path, digest in request["protocol_sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen protocol identity drift: {path}")
    return request
