"""Execute explicitly supplied release functions and retain a repair investigation.

The caller owns model loading, training and resource limits. This module never
loads executable code from an investigation file or decides which repairs to try.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from model_forensics.release_compare import Case, Comparison, Prediction, Release, Slice
from model_forensics.repair_compare import RepairComparison, compare_repairs

ReleaseFunction = Callable[[], Sequence[Prediction]]


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_investigation(
    directory: Path,
    *,
    cases: tuple[Case, ...],
    slices: tuple[Slice, ...],
    baseline: ReleaseFunction,
    candidate: ReleaseFunction,
    repairs: Mapping[str, ReleaseFunction],
    changes: Mapping[str, Any],
) -> dict:
    """Run baseline, candidate and every named repair once, then assess results.

    All functions must return predictions for the identical case set. Results
    from completed functions survive a later failure. Existing directories are
    never reused. Caller-supplied changes are recorded, not independently verified.
    CPU timing covers the current Python process, not external training workers.
    There is no timeout, retry, implicit network access or causal diagnosis here.
    """
    if not repairs:
        raise ValueError("an investigation requires at least one explicit repair")
    if any(not isinstance(name, str) or not name.strip() for name in repairs):
        raise ValueError("repair names must be nonblank strings")
    if {"baseline", "candidate"} & repairs.keys():
        raise ValueError("repair names cannot reuse baseline or candidate")
    # Validate evaluation identity before any user-supplied function executes.
    dummy = tuple(Prediction(case_id=c.case_id, observed=c.expected) for c in cases)
    Comparison(
        cases=cases,
        baseline=Release(release_id="baseline", predictions=dummy),
        candidate=Release(release_id="candidate", predictions=dummy),
        slices=slices,
    )
    functions = [("baseline", baseline), ("candidate", candidate), *sorted(repairs.items())]
    if any(not callable(function) for _, function in functions):
        raise ValueError("each release must have an explicit callable")
    plan = {
        "schema_version": "investigation-plan/0.1",
        "cases": [case.model_dump(mode="json") for case in cases],
        "slices": [item.model_dump(mode="json") for item in slices],
        "execution_order": [name for name, _ in functions],
        "declared_changes": dict(changes),
        "change_provenance": "caller_supplied",
        "automatic_retry": False,
    }
    # Reject unserializable plans without creating an attempt or invoking models.
    json.dumps(plan, allow_nan=False)
    directory.mkdir()
    _write(directory / "plan.json", plan)
    releases: dict[str, Release] = {}
    executions = []
    active = "preparation"
    try:
        for index, (name, function) in enumerate(functions):
            active = name
            wall = time.perf_counter()
            cpu = time.process_time()
            predictions = tuple(function())
            elapsed_cpu = time.process_time() - cpu
            elapsed_wall = time.perf_counter() - wall
            release = Release(release_id=name, predictions=predictions)
            reference = releases.get("baseline", Release(release_id="reference", predictions=dummy))
            Comparison(cases=cases, baseline=reference, candidate=release, slices=slices)
            # Filenames use generated indices, never caller-provided release names.
            record = {
                "release": release.model_dump(mode="json"),
                "wall_seconds": elapsed_wall,
                "current_process_cpu_seconds": elapsed_cpu,
                "cpu_scope": "current_process_only",
            }
            filename = f"execution-{index:03d}.json"
            _write(directory / filename, record)
            releases[name] = release
            executions.append(
                {
                    "name": name,
                    "file": filename,
                    **{k: v for k, v in record.items() if k != "release"},
                }
            )
        comparisons = {
            name: Comparison(
                cases=cases, baseline=releases["baseline"], candidate=value, slices=slices
            )
            for name, value in releases.items()
            if name != "baseline"
        }
        report = compare_repairs(
            RepairComparison(regressed=comparisons.pop("candidate"), repairs=comparisons)
        )
        report["executions"] = executions
        report["historical_cause"] = "not_identified"
        _write(directory / "report.json", report)
        return report
    except Exception as error:
        _write(
            directory / "failed.json",
            {"release": active, "error_type": type(error).__name__, "completed": executions},
        )
        raise
