"""Recompute saved investigations and render a portable HTML inspection report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from model_forensics.case_previews import previews
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
    """Render escaped inputs and local filtering without network requests."""
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
            f"<details><summary>{esc(name.replace(chr(95), chr(32)))}: slice results</summary>"
            "<table><thead><tr>"
            "<th>Slice</th><th>Cases</th><th>Baseline accuracy</th>"
            "<th>Release accuracy</th><th>Policy result</th>"
            f"<th>Regressed cases</th></tr></thead><tbody>{rows}</tbody></table></details>"
        )
    expected = {case["case_id"]: case["expected"] for case in plan["cases"]}
    predictions = [
        {p["case_id"]: p["observed"] for p in record["release"]["predictions"]}
        for record in records
    ]
    inputs = previews(directory, set(expected))
    rows = []
    for key in expected:
        before, after = predictions[0][key], predictions[1][key]
        state = (
            "regressed"
            if before == expected[key] and after != expected[key]
            else "improved"
            if before != expected[key] and after == expected[key]
            else "changed"
            if before != after
            else "unchanged"
        )
        values = (key, expected[key], *(p[key] for p in predictions))
        rows.append(
            f'<tr data-state="{state}"><td>{inputs.get(key, "Not supplied")}</td>'
            + "".join(
                f'<td><span class="case-id" title="{esc(v)}">{esc(v)}</span></td>'
                if index == 0
                else f'<td title="{esc(v)}">{esc(v.replace(chr(95), chr(32)))}</td>'
                for index, v in enumerate(values)
            )
            + "</tr>"
        )
    case_rows = "".join(rows)
    headers = "".join(
        f"<th>{esc(v.replace(chr(95), chr(32)))}</th>"
        for v in ("Input", "Case", "Expected", *plan["execution_order"])
    )
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
<style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;
color:#17212b}
table{border-collapse:collapse;width:100%;margin:20px 0;display:block;
overflow:auto}
th,td{padding:10px;border:1px solid #ccd3da;text-align:left;
vertical-align:top}
th{background:#edf2f6}
pre{white-space:pre-wrap;overflow-wrap:anywhere}

input,select,button{font:inherit;padding:8px;margin:4px}
tr[hidden]{display:none}

details{margin:12px 0}
label{display:inline-flex;flex-direction:column;margin-right:12px}

.case-scroll{overflow:auto}
#cases{display:table;table-layout:fixed;width:100%;min-width:1280px;font-size:14px}
#cases th:first-child{width:24%}
#cases th:nth-child(2){width:9%}
#cases td,#cases th{overflow-wrap:anywhere}
.case-id{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
font:12px ui-monospace,monospace}
summary{cursor:pointer;font-weight:600}
h1,h2{line-height:1.2}
</style>
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
        '<section id="case-inspection"><h2>Inspect cases</h2>'
        "<p>Outcome filters compare the candidate with the baseline. "
        "Repair predictions appear alongside them. Previews are caller-supplied inputs.</p>"
        '<label>Search cases or labels <input id="case-search" type="search"></label> '
        '<label>Candidate outcome <select id="case-state"><option value="all">All cases</option>'
        '<option value="different">Changed prediction</option>'
        '<option value="regressed">Correct → incorrect</option>'
        '<option value="improved">Incorrect → correct</option>'
        '<option value="unchanged">Unchanged prediction</option></select></label> '
        '<button id="export-cases" type="button">Export visible case IDs</button>'
        f'<p id="case-count" role="status">{len(expected)} cases</p>'
        f'<div class="case-scroll"><table id="cases"><thead><tr>{headers}</tr></thead>'
        f"<tbody>{case_rows}</tbody></table></div>"
        "<noscript>Filtering requires JavaScript; all case records are shown.</noscript></section>"
    )
    page += (
        "<h2>Measured execution cost</h2><p>Supplied function calls only; "
        "loading or training inside a call is included. "
        "Preparation outside those calls is excluded. CPU covers the current process, "
        "not external workers.</p>"
        f"<pre>{esc(json.dumps(costs, indent=2))}</pre>"
        "<details><summary>Declared changes (caller supplied)</summary>"
        f"<pre>{esc(json.dumps(plan['declared_changes'], indent=2))}</pre></details></html>"
    )
    attribution = directory / "input_attribution.txt"
    if attribution.exists():
        with attribution.open(encoding="utf-8") as stream:
            notice = stream.read(10001)
        if len(notice) > 10000:
            raise ValueError("input attribution exceeds 10000 characters")
        page = page.replace("</html>", "") + f"<h2>Input attribution</h2><p>{esc(notice)}</p>"
    page = (
        page.replace("</html>", "")
        + """
<script>
const rows = [...document.querySelectorAll('#cases tbody tr')];
const search = document.getElementById('case-search');
const state = document.getElementById('case-state');
function filterCases() {
  const query = search.value.toLowerCase().replaceAll('_', ' ');
  for (const row of rows) {
    const match = state.value === 'all' || row.dataset.state === state.value ||
      (state.value === 'different' && row.dataset.state !== 'unchanged');
    row.hidden = !match || !row.textContent.toLowerCase().replaceAll('_', ' ').includes(query);
  }
  document.getElementById('case-count').textContent =
    `${rows.filter(row => !row.hidden).length} of ${rows.length} cases`;
}
search.addEventListener('input', filterCases);
state.addEventListener('change', filterCases);
document.getElementById('export-cases').addEventListener('click', () => {
  const ids = rows.filter(row => !row.hidden).map(row => row.cells[1].textContent);
  const url = URL.createObjectURL(new Blob([JSON.stringify(ids, null, 2)],
    {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'case-ids.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
</script></html>"""
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
