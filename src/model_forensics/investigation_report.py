"""Recompute saved investigations and render a portable, non-executable HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from model_forensics.release_compare import Comparison, Release
from model_forensics.repair_compare import RepairComparison, compare_repairs


def reopen(directory: Path) -> tuple[dict, dict, list[dict]]:
    """Read indexed execution files; never trust saved assessment conclusions."""
    plan = json.loads((directory / "plan.json").read_text())
    if plan.get("schema_version") != "investigation-plan/0.1":
        raise ValueError("unsupported investigation plan")
    names = plan["execution_order"]
    if len(names) < 3 or names[:2] != ["baseline", "candidate"] or len(names) != len(set(names)):
        raise ValueError("invalid execution order")
    records = []
    for index, name in enumerate(names):
        record = json.loads((directory / f"execution-{index:03d}.json").read_text())
        release = Release.model_validate_json(json.dumps(record["release"]))
        if release.release_id != name:
            raise ValueError("execution identity differs from plan")
        records.append(record)
    comparisons = {
        name: Comparison.model_validate_json(
            json.dumps(
                {
                    "cases": plan["cases"],
                    "slices": plan["slices"],
                    "baseline": records[0]["release"],
                    "candidate": record["release"],
                }
            )
        )
        for name, record in zip(names[1:], records[1:], strict=True)
    }
    report = compare_repairs(
        RepairComparison(regressed=comparisons.pop("candidate"), repairs=comparisons)
    )
    return plan, report, records


def render(directory: Path, output: Path) -> None:
    """Create a self-contained report with escaped text and no scripts or requests."""
    plan, report, records = reopen(directory)
    esc = lambda value: html.escape(str(value), quote=True)  # noqa: E731
    sections = []
    comparisons = {"candidate": report["regressed"], **report["repairs"]}
    for name, comparison in comparisons.items():
        rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{esc(value)}</td>"
                for value in (
                    row["name"],
                    row["count"],
                    f"{row['baseline_accuracy']:.2%}",
                    f"{row['candidate_accuracy']:.2%}",
                    "Pass" if row["passed"] else "Fail",
                    len(row["regressed_case_ids"]),
                )
            )
            + "</tr>"
            for row in comparison["slices"]
        )
        sections.append(
            f"<h2>{esc(name)}</h2><table><thead><tr>"
            "<th>Slice</th><th>Cases</th><th>Baseline accuracy</th>"
            "<th>Release accuracy</th><th>Policy result</th>"
            f"<th>Regressed cases</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    expected = {case["case_id"]: case["expected"] for case in plan["cases"]}
    predictions = [
        {p["case_id"]: p["observed"] for p in record["release"]["predictions"]}
        for record in records
    ]
    changed = [
        key for key in expected if any(p[key] != predictions[0][key] for p in predictions[1:])
    ]
    case_rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(v)}</td>" for v in (key, expected[key], *(p[key] for p in predictions))
        )
        + "</tr>"
        for key in changed
    )
    headers = "".join(f"<th>{esc(v)}</th>" for v in ("Case", "Expected", *plan["execution_order"]))
    costs = [
        {
            "release": r["release"]["release_id"],
            "wall_seconds": r["wall_seconds"],
            "current_process_cpu_seconds": r["current_process_cpu_seconds"],
        }
        for r in records
    ]
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model release investigation</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;
color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0;display:block;
overflow:auto}th,td{padding:10px;border:1px solid #ccd3da;text-align:left;
vertical-align:top}th{background:#edf2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere}
summary{cursor:pointer;font-weight:600}h1,h2{line-height:1.2}</style>
<h1>Model release investigation</h1>
<p>Recomputed from saved predictions. Repair success does not identify a unique
historical cause. Slice results may overlap and are not statistical significance tests.</p>
"""
    assessment = report["assessment"]
    page += (
        "<h2>Repair assessment</h2>"
        f"<p>{esc(assessment['reason'])}</p>"
        "<p>Successful repairs: "
        f"{esc(', '.join(assessment['successful_repairs']) or 'None')}.</p>"
    )
    page += "".join(sections)
    page += (
        f"<details><summary>Changed predictions ({len(changed)} cases)</summary>"
        f"<table><thead><tr>{headers}</tr></thead><tbody>{case_rows}</tbody></table></details>"
    )
    page += (
        "<h2>Measured execution cost</h2><p>Function calls only; model setup is excluded. "
        "CPU covers the current process, not external workers.</p>"
        f"<pre>{esc(json.dumps(costs, indent=2))}</pre>"
        "<details><summary>Declared changes (caller supplied)</summary>"
        f"<pre>{esc(json.dumps(plan['declared_changes'], indent=2))}</pre></details></html>"
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(page)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        render(args.directory, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
