"""Spreadsheet view of a run: tab layout, formatting requests and the values of every tab.

The sheet is a view of the run's files for human readers. Nothing is invented: the view drops
op-table columns that hold one value in every row (listed once on Info), merges device op and
program factory, shortens source paths to their last two components, strips row-number prefixes
from evidence and orders columns reader-first. Tabs: Info (run, counts, summary.md, constant
columns), Device op support (with destinations: one row per device op, its rows by status and
whether each destination runs it), Device ops by factory (one row per device op and program
factory), ttnn calls (one row per launching variant), then the other CSVs.

Prints JSON with `tabs` (sheetId, title, source file, values: a 2D array for one write from A1;
formulas start with `=`, text the sheet would parse is prefixed with `'`), `requests` (one
batchUpdate: rename the first tab to Info, add the others, format) and `expected` (A1 cell ->
value its formula must read back as).

For an existing spreadsheet, delete every tab but the first and clear it, then pass that tab's id
as --info-sheet-id and a --first-sheet-id above every id the spreadsheet has used.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sheet_notes import AGENT_NOTE, CI_NOTE, CONFIDENCE_NOTE, E2E_NOTE, ISSUES_NOTE, METHOD_NOTES, STATUS_NOTES, column_note, tab_description  # noqa: E402
from opa_schema import (  # noqa: E402
    ALWAYS, CALL_SITE, CALL_TRACE_TAB, CI_KEY, ci_job, LLK_GAPS_TAB, CONFIDENCE_DISPLAY, DEVICE_OPS_TAB, GRID_KEYWORDS,
    KERNEL_OPS_TAB, LAUNCH_PROFILES, METHOD_ITEMS, METHOD_LABELS, NOT_CHECKED, NOT_TRACED, OP_SUPPORT_LABELS, OP_TABLE_TAB,
    ROW_PREFIX, SHEET_TABS, SOURCE_METHODS, STATUS_COUNT_FIELDS, STATUS_VALUES,
    TAB_METHOD, TARGET_LABELS, TEST_TARGETS, TTSIM_TESTS_TAB, column_width, confidence, display_header,
    build_label, latest_entry_failure, methods_used, read_csv, read_header, run_targets, split_evidence,
)

HEADER_GREY = {"red": 0.9, "green": 0.9, "blue": 0.9}
INFO_WIDTHS = (260, 900, 220, 320)
NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
SOURCE_PATH = re.compile(r"[\w.-]+(?:/[\w.-]+){2,}\.(?:py|cpp|hpp|cc|h|yaml)\b")
# Free-text columns whose source paths and evidence prefixes the view may shorten. Test node ids
# and log paths stay whole so they can be copied into a command.
PROSE_FIELDS = {"call_site", "grid_dependency", "dispatch_evidence", "evidence", "notes", "required_change",
                "blocker", "detail"}
# Columns a reader scans or that the Counts formulas read, kept even when every row has one value.
KEPT_FIELDS = {"id", "status", "launches"}
MERGED = "device_op / program_factory"
MERGED_HEADER = "Device op / program factory"
TRAILING_OP_FIELDS = ("grid_dependency", "op_code", "confidence", "dispatch_evidence", "origin")


def cell(text: str):
    # Sheets stores numbers as doubles; longer integers, such as kernel compile hashes, would be rounded.
    if NUMBER.match(text) and ("." in text or len(text.lstrip("-")) <= 15):
        return float(text) if "." in text else int(text)
    if text and (text[0] in "+-=@'" or text[0].isdigit() or text.upper() in ("TRUE", "FALSE")):
        return "'" + text
    return text


def col_letter(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def quote(title: str) -> str:
    return "'" + title.replace("'", "''") + "'"


def tidy(text: str) -> str:
    measured, written = split_evidence(text)
    text = ROW_PREFIX.sub("", written)
    if measured:
        text = f"{measured}\n{text}"
    return SOURCE_PATH.sub(lambda m: "/".join(m.group(0).split("/")[-2:]), text)


def field(column: str) -> str:
    return column.split(":", 1)[-1]


def view_text(column: str, value: str) -> str:
    if field(column) == "confidence":
        return CONFIDENCE_DISPLAY.get(value, value)
    return tidy(value) if field(column) in PROSE_FIELDS else value


def short_op(device_op: str) -> str:
    return device_op.removeprefix("ttnn::")


def label(target: str) -> str:
    return TARGET_LABELS.get(target, target)


CONFIDENCE_TEXT = {
    "high": "a graph capture of the entry test names the device op and program factory of every launch",
    "medium": ("a run of the entry test confirmed op codes (Tracy) or compiled kernels (kernel cache); "
               "program factories come from the static trace"),
    "low": "read from the model code at the pinned commit; nothing was run",
}


def method_state(run: dict, method: str) -> str:
    if method in METHOD_ITEMS and not run.get(METHOD_ITEMS[method][1]):
        mark = run.get("measurements", {}).get(METHOD_ITEMS[method][0], {})
        if mark.get("state") == "failed":
            return f"failed {mark.get('date', '')}: {mark.get('note', '')}".rstrip(": ")
    if method == "static":
        return f"done at {run.get('tt_metal_commit', '')[:7]}" if run.get("kind") == "static" else "not run"
    if method == "tracy":
        if not run.get("measured_ops"):
            return "not run"
        return f"done: {run.get('measured_inferences', 1)} inference(s) in the measured window"
    if method == "graph":
        g = run.get("graph_capture")
        if not g:
            return "not run"
        return (f"done {g.get('date', '')}: {g.get('launches', '?')} launches over {g.get('inferences', '?')} "
                f"inference(s), {g.get('kernel_ops', '?')} (op code, program factory) pairs, "
                f"{g.get('host_transfers_inside', '?')} host transfers between the first and the last launch")
    c = run.get("kernel_cache")
    if not c:
        return "not run"
    return (f"done {c['date']}: {c['kernel_builds']} kernel builds ({c['compute_builds']} compute, "
            f"{c['dataflow_builds']} data movement), {c['compute_rows']} compute and {c['dataflow_rows']} dataflow "
            f"API rows, layers {', '.join(c['layers'])}")


def new_tab(title: str, source: str, values: list[list], widths: list[int], bold: list[int] | None = None,
            header: bool = True) -> dict:
    return {"title": title, "source": source, "values": values, "widths": widths,
            "bold": bold if bold is not None else [0], "header": header}


def op_table_view(rows: list[dict], header: list[str], targets: tuple[str, ...]):
    order = ["id", "stage", "ttnn_api", MERGED]
    for t in targets:
        order += [c for c in (f"{t}:shapes", f"{t}:launches") if c in header]
    order += ["call_site", "launch_condition"]
    for t in targets:
        order += [c for c in (f"{t}:status", f"{t}:tests", f"{t}:test_result", f"{t}:evidence") if c in header]
    order += list(TRAILING_OP_FIELDS)
    order += [c for c in header if c not in order and c not in ("device_op", "program_factory")]

    def raw(row: dict, column: str) -> str:
        if column == MERGED:
            return f"{short_op(row['device_op'])} / {row['program_factory']}"
        return row[column]

    # Constancy is judged on the CSV text, before shortening, so the view never claims two
    # different values are the same.
    constant: list[tuple[str, str]] = []
    columns = []
    for column in order:
        values = {raw(r, column) for r in rows}
        if len(rows) > 1 and len(values) == 1 and field(column) not in KEPT_FIELDS and column != MERGED:
            text = values.pop()
            if text:
                constant.append((display_header(column), view_text(column, text)))
            continue
        columns.append(column)
    headers = [MERGED_HEADER if c == MERGED else display_header(c) for c in columns]
    values = [headers] + [[cell(view_text(c, raw(r, c))) for c in columns] for r in rows]
    widths = [300 if c == MERGED else column_width(c) for c in columns]
    tab = new_tab(OP_TABLE_TAB, "op_table.csv", values, widths)
    tab["notes"] = [column_note(OP_TABLE_TAB, "program_factory" if c == MERGED else c) for c in columns]
    return tab, columns, constant


STATUS_RANK = {"❌": 0, "⚠️": 1, "✅": 2, "n/a": 3}
# A clause ends at ';', or at '.' before a capital that does not close an abbreviation, so "e.g. on" and
# "incl. DRAM" stay whole.
CLAUSE_SPLIT = re.compile(r"(?<=;)\s+|(?<=\.)(?<!\bincl\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\betc\.)"
                          r"(?<!\bcf\.)(?<!\bapprox\.)(?<!\bno\.)\s+(?=[A-Z])")
SHARED_CLAUSE_CHARS = 40


def _clauses(text: str) -> list[str]:
    return [c.strip().rstrip(";.").strip() for c in CLAUSE_SPLIT.split(text) if c.strip().rstrip(";.").strip()]


def grouped_evidence(members: list[dict], column: str) -> str:
    """Evidence of a kernel op's rows: one block per distinct text, worst status first, each block led by its
    status and rows; a clause every block repeats is written once at the end."""
    texts: dict[str, dict] = collections.OrderedDict()
    status_column = column.rsplit(":", 1)[0] + ":status"
    for m in members:
        group = texts.setdefault(tidy(m[column]), {"ids": [], "statuses": []})
        group["ids"].append(m["id"])
        if m.get(status_column) and m[status_column] not in group["statuses"]:
            group["statuses"].append(m[status_column])
    if len(texts) <= 1:
        return next(iter(texts), "")
    split = {text: _clauses(text) for text in texts}
    # A short note several blocks repeat (bf16 no-op) is written once, with the rows it applies to; longer
    # clauses carry a block's reasoning and stay in it.
    shared: dict[str, list[str]] = collections.OrderedDict()
    for text, clauses in split.items():
        for c in clauses:
            if len(c) <= SHARED_CLAUSE_CHARS and sum(c in other for other in split.values()) > 1:
                shared.setdefault(c, []).extend(texts[text]["ids"])
    common = list(shared)
    def rank(item) -> int:
        return min((STATUS_RANK.get(s, 9) for s in item[1]["statuses"]), default=9)

    blocks = []
    for text, group in sorted(texts.items(), key=rank):
        if common:
            own = [c for c in split[text] if c not in common]
            end = "." if text.rstrip().endswith((".", ";")) else ""
            if own:
                body = "; ".join(own) + end
            elif any(c in shared for c in split[text]):
                body = "see below."
            else:
                body = text or "no evidence written."
        else:
            body = text
        rows = f"{'row' if len(group['ids']) == 1 else 'rows'} {', '.join(group['ids'])}"
        blocks.append(f"{' '.join(group['statuses'])} {rows}: {body}".strip())
    every = [m["id"] for m in members]
    by_rows: dict[tuple[str, ...], list[str]] = collections.OrderedDict()
    for clause, ids in shared.items():
        by_rows.setdefault(tuple(sorted(set(ids), key=every.index)), []).append(clause)
    for ids, clauses in by_rows.items():
        lead = "All rows" if len(ids) == len(set(every)) else f"{'Row' if len(ids) == 1 else 'Rows'} {', '.join(ids)}"
        blocks.append(f"{lead}: " + "; ".join(clauses) + ".")
    return "\n\n".join(blocks)


def grouped_nodes(cell_text: str) -> str:
    """`path::test[...]` entries (optionally `: result`) grouped under their test file."""
    files: dict[str, list[str]] = collections.OrderedDict()
    for entry in (e.strip() for e in cell_text.split("; ") if e.strip()):
        path, sep, rest = entry.partition("::")
        tests = files.setdefault(path, [])
        if sep:
            tests.append(rest)
    return "\n".join("\n".join([path, *(f"  ::{test}" for test in tests)]) for path, tests in files.items())


def kernel_ops_view(kernel_ops: list[dict], ops: list[dict], targets: tuple[str, ...], op_columns: list[str]):
    by_id = {r["id"]: r for r in ops}
    measured = any(k.get("measured_launches_op_code", "0") not in ("", "0") for k in kernel_ops)
    graphed = any(k.get("graph_launches", "0") not in ("", "0") for k in kernel_ops)
    # Evidence that is the same in every op-table row is on Info, not repeated per kernel op.
    shown = {t: f"{t}:evidence" in op_columns for t in targets}
    # (header, width, field whose note the header carries)
    columns = [("#", 40, "id"), (MERGED_HEADER, 300, "program_factory"), ("Stages", 160, "stage"),
               ("ttnn API", 180, "ttnn_api"), (f"Launches ({label(targets[0])})", 90, "static_launches")]
    if measured:
        columns.append(("Measured launches (op code)", 110, "measured_launches_op_code"))
    if graphed:
        columns.append(("Graph launches per inference", 110, "graph_launches"))
    columns += [("Seen by", 140, "sources"), ("ttnn call rows", 120, "op_table_ids")]
    for t in targets:
        columns.append((f"{label(t)}: status (rows)", 110, "statuses"))
        if shown[t]:
            columns.append((f"{label(t)}: evidence and gap", 420, "evidence"))
        if t in TEST_TARGETS:
            columns += [(f"{label(t)}: kernel-op tests", 300, "tests"), (f"{label(t)}: TTsim results", 300, "test_results")]
    headers = [c[0] for c in columns]
    widths = [c[1] for c in columns]
    values = [headers]
    for k in kernel_ops:
        members = [by_id[i] for i in k["op_table_ids"].split(",") if i in by_id]
        stages = dict.fromkeys(m["stage"].split()[0] for m in members if m["stage"])
        apis = dict.fromkeys(m["ttnn_api"] for m in members if m["ttnn_api"])
        if k["program_factory"]:
            name = f"{short_op(k['device_op']) or k['op_code']} / {k['program_factory']}"
        else:
            name = k["op_code"]
        row = [cell(k["id"]), name, ", ".join(stages), ", ".join(apis), cell(k["static_launches"])]
        if measured:
            row.append(cell(k["measured_launches_op_code"]))
        if graphed:
            row.append(cell(k.get("graph_launches", "")))
        row += [k.get("sources", ""), cell(k["op_table_ids"].replace(",", ", "))]
        for t in targets:
            row.append(cell(k.get(f"{t}:statuses", "")))
            if shown[t]:
                row.append(cell(grouped_evidence(members, f"{t}:evidence")))
            if t in TEST_TARGETS:
                row += [cell(grouped_nodes(k.get(f"{t}:tests", ""))),
                        cell(grouped_nodes(k.get(f"{t}:test_results", "")))]
        values.append(row)
    values.append(["Total", "", "", "", f"=SUM(E2:E{len(kernel_ops) + 1})"])
    tab = new_tab(KERNEL_OPS_TAB, "kernel_ops.csv", values, widths, [0, len(values) - 1])
    tab["notes"] = [column_note(KERNEL_OPS_TAB, c[2]) for c in columns]
    return tab


def op_support_text(verdict: str, evidence: str) -> str:
    if verdict == NOT_CHECKED:
        return f"{NOT_CHECKED} not checked"
    text = f"{verdict} {OP_SUPPORT_LABELS.get(verdict, verdict)}"
    return f"{text}: {tidy(evidence)}" if evidence else text


def device_ops_view(device_ops: list[dict], targets: tuple[str, ...]) -> tuple[dict, dict[str, int]]:
    """One row per device op: per destination, its ttnn call rows by status and whether the destination
    runs the op at all. Returns the tab and the totals its formulas must read back as."""
    source, destinations = targets[0], targets[1:]
    # (header, width, note field, device_ops.csv column or None for the fixed columns)
    columns = [("#", 40, "id", "id"), ("Device op", 220, "device_op", "device_op"),
               ("Program factories", 260, "program_factories", "program_factories"),
               ("ttnn API", 180, "ttnn_apis", "ttnn_apis"), ("ttnn call rows", 120, "op_table_ids", "op_table_ids"),
               (f"Launches ({label(source)})", 90, "launches", f"{source}:launches")]
    summed = [f"{source}:launches"]
    for t in destinations:
        for status, f in STATUS_COUNT_FIELDS.items():
            # n/a is shown only where the destination skips some op.
            if status == "n/a" and not any(int(k[f"{t}:{f}"] or 0) for k in device_ops):
                continue
            columns.append((f"{label(t)}: {status} rows", 80, f, f"{t}:{f}"))
            summed.append(f"{t}:{f}")
        columns.append((f"{label(t)}: device op supported", 420, "op_support", f"{t}:op_support"))
    values = [[c[0] for c in columns]]
    for k in device_ops:
        row = []
        for _, _, _, column in columns:
            if column.endswith(":op_support"):
                row.append(cell(op_support_text(k[column], k[column.replace(":op_support", ":op_evidence")])))
            elif column == "device_op":
                row.append(short_op(k[column]))
            elif column == "op_table_ids":
                row.append(cell(k[column].replace(",", ", ")))
            else:
                row.append(cell(k[column].replace("; ", "\n")))
        values.append(row)
    total = ["Total"] + [""] * (len(columns) - 1)
    expected = {}
    for column in summed:
        i = next(n for n, c in enumerate(columns) if c[3] == column)
        letter = col_letter(i)
        total[i] = f"=SUM({letter}2:{letter}{len(device_ops) + 1})"
        expected[f"{quote(DEVICE_OPS_TAB)}!{letter}{len(device_ops) + 2}"] = sum(int(k[column] or 0) for k in device_ops)
    values.append(total)
    tab = new_tab(DEVICE_OPS_TAB, "device_ops.csv", values, [c[1] for c in columns], [0, len(values) - 1])
    tab["notes"] = [column_note(DEVICE_OPS_TAB, c[2]) for c in columns]
    return tab, expected


REVIEW_MARK = {"gap": "❌ gap", "renamed": "✅ renamed", "not a gap": "✅ not a gap"}
HOST_OP_MARK = {"yes": "⚠️ yes", "no": "✅ no"}
CANDIDATES = ("missing", "signature differs")
LONG_TEXT_CHARS = 300


def csv_value(column: str, row: dict, source: str) -> str:
    value = row[column]
    if column == "review":
        if value:
            return REVIEW_MARK.get(value, value)
        # A missing data format is read from the Quasar enum and needs no reading of headers.
        return "⏳ not reviewed" if row.get("kind") == "llk_api" and row.get("quasar_status") in CANDIDATES else ""
    if column == "counts_as_host_op":
        return HOST_OP_MARK.get(value, value)
    if column == "attributes" and len(value) > LONG_TEXT_CHARS:
        # Tracy's attribute dumps run to kilobytes; the sheet keeps the start and points at the file.
        return value[:LONG_TEXT_CHARS] + f" … (cut; whole in {source})"
    return view_text(column, value)


def csv_view(path: pathlib.Path, title: str) -> dict:
    columns = read_header(path)
    values = [[display_header(c) for c in columns]]
    values += [[cell(csv_value(c, r, path.name)) for c in columns] for r in read_csv(path)]
    tab = new_tab(title, path.name, values, [column_width(c) for c in columns])
    tab["notes"] = [column_note(title, c) for c in columns]
    return tab


def summary_rows(path: pathlib.Path) -> tuple[str, list[list], list[int]]:
    """summary.md as label | text rows; heading rows are returned for bolding.

    The first table is the kernel-op table, shown as its own tab; later tables keep their cells.
    """
    title, rows, headings, paragraph = "", [], [], []
    in_code = in_table = seen_table = False
    last_item = False

    def clean(text: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"\1", text).replace("`", "")

    def flush():
        if paragraph:
            rows.append(["", cell(" ".join(paragraph))])
            paragraph.clear()

    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            flush()
            in_code, last_item = not in_code, False
            continue
        if in_code:
            if s:
                rows.append(["", cell(s)])
            continue
        if s.startswith("|"):
            flush()
            if not in_table:
                in_table, skip = True, not seen_table
                seen_table = True
            if not skip and not re.fullmatch(r"\|[\s:|-]+\|", s):
                rows.append([cell(clean(c.strip())) for c in s.strip("|").split("|")])
            last_item = False
            continue
        in_table = False
        if not s:
            flush()
            last_item = False
            continue
        if last_item and line[:1].isspace():
            rows[-1][-1] = cell(f"{str(rows[-1][-1]).lstrip(chr(39))} {clean(s)}")
            continue
        heading = re.match(r"^(#+)\s+(.*)", s)
        if heading:
            flush()
            last_item = False
            if heading.group(1) == "#" and not title:
                title = clean(heading.group(2))
                continue
            rows.append([])
            headings.append(len(rows))
            rows.append([clean(heading.group(2))])
            continue
        item = re.match(r"^(?:[-*]|(\d+)\.)\s+(.*)", s)
        if item:
            flush()
            last_item = True
            text = clean(item.group(2))
            # A label is a bold lead, "Label: text", or a counted "Label (n), text".
            match = (re.match(r"^\*\*(.{1,60}?):?\*\*(?<=:\*\*)\s+(.+)$", item.group(2))
                     or re.match(r"^\*\*(.{1,60}?)\*\*[:,]\s+(.+)$", item.group(2))
                     or re.match(r"^([^:]{1,40}):\s+(.+)$", text)
                     or re.match(r"^([^:,]{1,40}\(\d+\))[,:]\s+(.+)$", text))
            labelled = (clean(match.group(1)), clean(match.group(2))) if match else None
            if item.group(1):
                rows.append([f"'{item.group(1)}.", cell(text)])
            elif labelled:
                rows.append([labelled[0], cell(labelled[1])])
            else:
                rows.append(["", cell(text)])
            continue
        last_item = False
        paragraph.append(clean(s))
    flush()
    return title, rows, headings


LEGEND_HEADING = "Status legend"


def take_section(rows: list[list], headings: list[int], name: str) -> tuple[list[list], list[int], dict[str, str]]:
    """Remove one summary section; returns the rows and headings left and the section's label -> text."""
    start = next((h for h in headings if rows[h][0].lower() == name.lower()), None)
    if start is None:
        return rows, headings, {}
    end = next((h - 1 for h in headings if h > start), len(rows))
    taken = {str(r[0]).replace("\ufe0f", ""): str(r[1]).removeprefix("'") for r in rows[start + 1:end]
             if len(r) > 1 and r[0]}
    removed = end - (start - 1)
    kept = rows[:start - 1] + rows[end:]
    return kept, [h if h < start else h - removed for h in headings if h != start], taken


LEGEND_PART = re.compile(r"(?:^|(?<=\. ))([A-Z][\w ,()]{0,40}?) - ")


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def legend_note(text: str, target_label: str, all_labels: list[str]) -> str | None:
    """The part of a legend bullet ("P100 - ... Quasar BF16 - ...") that defines the status for one target.

    Only names that fit one of the run's targets start a part, so a definition may hold "<words> - ".
    None when the bullet defines the status for other targets only.
    """
    def name(m: re.Match) -> set[str]:
        return words(m.group(1)) - {"source", "destination"}

    starts = [m for m in LEGEND_PART.finditer(text) if name(m) and any(name(m) <= words(t) for t in all_labels)]
    if not starts:
        return text
    best, best_score = None, 0
    for m, nxt in zip(starts, starts[1:] + [None]):
        if name(m) <= words(target_label) and len(name(m)) > best_score:
            best, best_score = text[m.end():nxt.start() if nxt else len(text)].strip(), len(name(m))
    return best[:1].upper() + best[1:] if best else None


CONFIG_MARK = {"yes": "✅ yes", "no": "❌ no", "partial": "⚠️ partial", "n/a": "✅ n/a"}
GRID_OVERRIDE = "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"


def open_issues(run: dict) -> list[dict]:
    return (run.get("entry_test_issues") or {}).get("open") or []


def issue_ref(issue: dict) -> str:
    number = issue.get("url", "").rstrip("/").rsplit("/", 1)[-1]
    return f"#{number}" if number.isdigit() else issue.get("url", "")


def entry_test_state(run: dict) -> str:
    passes, failure = run.get("entry_test_passes") or {}, latest_entry_failure(run)
    if failure:
        ci = f"; CI: {ci_job(run)}" if ci_job(run) else ""
        return (f"❌ failed: {failure['result']} ({failure['by']}, {failure['date']}; "
                f"{build_label(failure.get('build', ''))}){ci}")
    if passes:
        parts = []
        if passes.get("result"):
            parts.append(f"{passes['result']} ({passes.get('by', '')}, {passes.get('date', '')}; "
                         f"{build_label(passes.get('build', ''))})")
        if ci_job(run):
            parts.append(f"CI: {ci_job(run)}")
        issues = open_issues(run)
        where = " locally" if passes.get("result") and not ci_job(run) else ""
        failed_runs = [(item, m) for item, m in (run.get("measurements") or {}).items()
                       if item.startswith("source:") and m.get("state") == "failed" and m.get("cause") in ("test", "tool")]
        test_failures = [f for f in failed_runs if f[1]["cause"] == "test"]
        # A pass does not close an open issue that reports the test failing, nor a hang or failure of the same test
        # in another run of this analysis.
        mark = f"⚠️ passed{where}" if issues or test_failures else f"✅ passed{where}"
        if failed_runs:
            parts.append(f"passed in 1 of {1 + len(failed_runs)} runs: " + ", ".join(
                f"{item} {'failed in the test' if m['cause'] == 'test' else 'stopped by its tool'} ({m.get('date', '')})"
                for item, m in failed_runs) + " (Analysis methods)")
        if issues:
            parts.append("open " + ", ".join(issue_ref(i) for i in issues) + " reports it failing (Entry test issues)")
        return f"{mark}: " + "; ".join(parts)
    return "not run on a device in this analysis"


def issues_state(run: dict) -> str:
    record = run.get("entry_test_issues")
    if not record:
        return "⏳ not checked"
    issues = open_issues(run)
    if not issues:
        return f"✅ none open (searched {record.get('checked', '')})"
    listed = "; ".join(f"{issue_ref(i)} {i.get('title', '')}: {i.get('effect', '')}".strip() for i in issues)
    return f"⚠️ {len(issues)} open (searched {record.get('checked', '')}): {listed}"


NOT_RECORDED = "not recorded"


def analysed_by_state(run: dict) -> str:
    entries = run.get("analysed_by") or []
    if not entries:
        return "not recorded"
    def who(e: dict) -> str:
        model = e.get("model", "")
        return e.get("agent", "") + (f" ({model})" if model and model != NOT_RECORDED else "")

    return "; ".join(f"{who(e)}: {e.get('work', '')}, {e.get('date', '')}" for e in entries)


HOST_SHARE_WARN = 5.0


def host_fallback_mark(share: float) -> str:
    """✅ no host op, ⚠️ up to 5% of ops, ❌ above."""
    if share > HOST_SHARE_WARN:
        return "❌ "
    return "⚠️ " if share > 0 else "✅ "


def host_fallback_state(run_dir: pathlib.Path, run: dict) -> str:
    """Host fallback from the Tracy run, which must call the forward live: a replayed device trace never
    re-enters the forward, so it shows no host op whatever the model does."""
    path = run_dir / "host_fallback.csv"
    if not path.exists():
        return "not measured: needs a Tracy run of the entry test in live mode"
    mode = run.get("tracy_mode")
    if mode == "trace":
        return ("⚠️ not measured: the Tracy run replayed a device trace, which never re-enters the forward; "
                "run it live")
    if mode != "live":
        return "not decided: the Tracy run does not record whether it ran live or replayed a trace"
    parts, share = [], 0.0
    for row in read_csv(path):
        device, host = int(row["device_ops"] or 0), int(row["host_ops"] or 0)
        row_share = 100 * host / (device + host) if device + host else 0.0
        share = max(share, row_share)
        parts.append(f"{host} of {device + host} ops on the host ({row_share:.1f}%), {row['host_pct_of_total']}% of "
                     f"the time (Tracy live, {row['window']})")
    return host_fallback_mark(share) + "; ".join(parts)


def peak_dram_state(run_dir: pathlib.Path, run: dict) -> str:
    graph = run.get("graph_capture") or {}
    if graph.get("peak_dram_mib") is not None:
        per_device = graph.get("peak_dram_mib_per_device") or {}
        detail = f", per device {per_device}" if len(per_device) > 1 else ""
        return f"{graph['peak_dram_mib']} MiB allocated at once during the test (graph capture){detail}"
    footprint = run_dir / "footprint.csv"
    if footprint.exists():
        values = {r["peak_dram_mb"] for r in read_csv(footprint)} - {"not measured", ""}
        if values:
            return f"{values.pop()} MB (measured outside the analysis)"
    return "⏳ not measured; a graph capture records it"


def grid_blockers_state(ops: list[dict], source: str) -> str:
    untraced = [r for r in ops if r["grid_dependency"].strip() == NOT_TRACED]
    fixed = [r for r in ops if r["grid_dependency"].strip() not in (*GRID_KEYWORDS, NOT_TRACED)]
    unknown = (f"; {len(untraced)} measured-only rows were not traced, so their grids are unknown"
               if untraced else "")
    if not fixed:
        if untraced:
            return f"⚠️ none in the static trace{unknown}"
        return "✅ none: every op derives its grid from the device or uses no core grid"
    settings = sorted({site for r in fixed for site in CALL_SITE.findall(r["grid_dependency"])})
    launches = sum(int(r.get(f"{source}:launches") or 0) for r in fixed)
    listed = ", ".join(tidy(s) for s in settings[:8]) + (f" and {len(settings) - 8} more" if len(settings) > 8 else "")
    return (f"❌ {len(fixed)} of {len(ops)} ttnn call rows ({launches} launches per inference) use hardcoded grids "
            f"or shard specs, set at {len(settings)} places: {listed} ({OP_TABLE_TAB}, Grid dependency){unknown}")


CI_MARK = {"yes": "✅ yes", "no": "⚠️ no", "n/a": "n/a"}


def ci_state(run: dict, source: str) -> str:
    entry = run.get(CI_KEY)
    if isinstance(entry, dict) and entry.get("value") in CI_MARK:
        # Without a CI job the end-to-end result below can only come from a local run; say so, or the two read
        # as a contradiction.
        local = ", so any pass below is from a local run" if entry["value"] == "no" else ""
        return f"{CI_MARK[entry['value']]} on {label(source)}{local}: {entry.get('evidence', '')}"
    job = ci_job(run)
    return f"✅ yes on {label(source)}: {job}" if job else "⏳ not checked"


def config_state(run: dict, key: str) -> str:
    entry = run.get(key)
    if not entry:
        return "⏳ not checked"
    return f"{CONFIG_MARK.get(entry.get('value', ''), entry.get('value', ''))}: {tidy(entry.get('evidence', ''))}"


def config_notes(run: dict) -> dict[str, str]:
    """Info label -> the setting that changes it, as a note on the label."""
    layers = run.get("configurable_layers") or {}
    model_var = re.sub(r"\W+", "_", str(run.get("model", "model"))).upper() + "_NUM_LAYERS"
    layer_note = ("Setting: an environment variable the model reads, by convention <MODEL>_NUM_LAYERS (here "
                  f"{model_var}), or a test parameter; tt-metal has no shared one.")
    if layers.get("setting"):
        layer_note = f"Setting: {layers['setting']}, which the model reads to build fewer layers."
    elif layers.get("value") == "n/a":
        layer_note += " Not needed: the model has no repeated layer stack."
    elif layers.get("value") == "no":
        layer_note += " This model reads none, so its layer count is fixed in the code."
    else:
        layer_note += " Whether this model reads one is not checked yet."
    grid_value = (run.get("configurable_grid") or {}).get("value")
    grid_note = f"Setting: the environment variable {GRID_OVERRIDE}, which overrides the worker core grid at run time."
    if grid_value == "yes":
        grid_note += " Every grid the model uses follows it without a code change."
    elif grid_value in ("no", "partial"):
        grid_note += " Grids hardcoded in the model do not follow it; they need a code change."
    else:
        grid_note += " Whether the model's grids follow it is not checked yet."
    return {"Configurable grid": grid_note,
            "Configurable layer count": layer_note}


def info_view(run_dir: pathlib.Path, run: dict, targets: tuple[str, ...], op_columns: list[str],
              constant: list[tuple[str, str]], ops: list[dict]):
    source, destinations = targets[0], targets[1:]
    title, summary, summary_headings = summary_rows(run_dir / "summary.md")
    # The legend defines statuses for this run's targets; on the sheet it is a note on each status count.
    summary, summary_headings, legend = take_section(summary, summary_headings, LEGEND_HEADING)
    rows: list[list] = [[title or f"{run.get('model', 'Model')} kernel op analysis"]]
    bold = [0]
    profile = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in run.get("profile_values", {}).items()
                        if not isinstance(v, bool))
    commit = run.get("tt_metal_commit", "")
    detail = [x for x in (run.get("tt_metal_ref"), run.get("date")) if x]
    if detail:
        commit += f" ({', '.join(detail)})"
    meta = [("Model", run.get("model")), ("Implementation", run.get("implementation")),
            ("Entry test", run.get("test")), ("Entry test in CI", run.get("test") and ci_state(run, source)),
            ("Source", label(source)),
            ("Destinations", ", ".join(label(d) for d in destinations)), ("tt-metal commit", commit),
            ("TTsim commit", run.get("ttsim_commit")), ("Analysed by", analysed_by_state(run))]
    rows += [[name, cell(str(v))] for name, v in meta if v]

    rows.append([])
    bold.append(len(rows))
    rows.append(["Bring-up checks"])
    checks = [
        ("End-to-end test", entry_test_state(run)),
        ("Entry test issues", issues_state(run)),
        ("Host fallback", host_fallback_state(run_dir, run)),
        ("Test hardware", f"{label(source)}: {profile}" if profile else label(source)),
        ("Peak DRAM", peak_dram_state(run_dir, run)),
        ("Blockers to a smaller grid", grid_blockers_state(ops, source)),
        ("Configurable grid", config_state(run, "configurable_grid")),
        ("Configurable layer count", config_state(run, "configurable_layers")),
    ]
    rows += [[name, cell(value)] for name, value in checks]

    used = methods_used(run)
    level = confidence(used)
    rows.append([])
    bold.append(len(rows))
    rows.append(["Analysis methods", "State"])
    rows += [[METHOD_LABELS[m], cell(method_state(run, m))] for m in SOURCE_METHODS]
    rows.append(["Confidence", cell(f"{level}: {CONFIDENCE_TEXT[level]}")])
    bold.append(len(rows) - 1)
    label_notes = config_notes(run) | {METHOD_LABELS[m]: METHOD_NOTES[m] for m in SOURCE_METHODS}
    label_notes["Confidence"] = CONFIDENCE_NOTE
    label_notes["Entry test in CI"] = CI_NOTE
    label_notes |= {"Entry test issues": ISSUES_NOTE, "Analysed by": AGENT_NOTE, "End-to-end test": E2E_NOTE}
    cell_notes = [(n, 0, label_notes[r[0]]) for n, r in enumerate(rows) if r and r[0] in label_notes]

    # Counts: formulas over the ttnn calls tab, so a reader can trace every number.
    expected: dict[str, int] = {}
    last = len(ops) + 1

    def rng(column: str) -> str:
        letter = col_letter(op_columns.index(column))
        return f"{quote(OP_TABLE_TAB)}!{letter}2:{letter}{last}"

    launch_targets = [t for t in targets if t == source or t in LAUNCH_PROFILES]
    rows.append([])
    bold.append(len(rows))
    rows.append(["Counts", "ttnn call rows"] + [f"Launches per inference ({label(t)})" for t in launch_targets])
    count_rows: list[tuple[str, str | None, str]] = [("All rows", None, "")]
    if "launch_condition" in op_columns:
        count_rows.append((f"Launch condition: {ALWAYS}", "launch_condition", ALWAYS))
    for t in targets:
        present = {r[f"{t}:status"] for r in ops}
        count_rows += [(f"{label(t)}: {s}", f"{t}:status", s) for s in STATUS_VALUES if s in present]
    for name, column, value in count_rows:
        members = ops if column is None else [r for r in ops if r[column] == value]
        rows.append([name, f"=COUNTA({rng('id')})" if column is None else f'=COUNTIF({rng(column)},"{value}")'])
        if column and column.endswith(":status"):
            text = legend.get(value.replace("\ufe0f", ""))
            note = legend_note(text, label(column.split(":")[0]), [label(t) for t in targets]) if text else None
            cell_notes.append((len(rows) - 1, 0, note or STATUS_NOTES[value]))
        expected[f"Info!B{len(rows)}"] = len(members)
        for i, t in enumerate(launch_targets):
            launches = rng(f"{t}:launches")
            rows[-1].append(f"=SUM({launches})" if column is None else f'=SUMIFS({launches},{rng(column)},"{value}")')
            expected[f"Info!{col_letter(2 + i)}{len(rows)}"] = sum(int(m[f"{t}:launches"] or 0) for m in members)

    gaps_path = run_dir / "llk_gaps_quasar.csv"
    if gaps_path.exists():
        gaps = read_csv(gaps_path)
        gap_columns = read_header(gaps_path)
        end = len(gaps) + 1

        def gap_rng(column: str) -> str:
            letter = col_letter(gap_columns.index(column))
            return f"{quote(LLK_GAPS_TAB)}!{letter}2:{letter}{end}"

        rows.append([])
        bold.append(len(rows))
        rows.append([LLK_GAPS_TAB, "Count"])
        llk = [g for g in gaps if g["kind"] == "llk_api"]
        checks = [("LLK APIs in the cache", f'=COUNTIF({gap_rng("kind")},"llk_api")', len(llk))]
        for status, name in (("present", "found on Quasar by name"), ("signature differs", "signature differs"),
                             ("missing", "not found by name")):
            checks.append((f"  {name}", f'=COUNTIFS({gap_rng("kind")},"llk_api",{gap_rng("quasar_status")},"{status}")',
                           sum(g["quasar_status"] == status for g in llk)))
        for verdict, mark in (("gap", "❌ gap"), ("renamed", "✅ renamed"), ("not a gap", "✅ not a gap")):
            checks.append((f"Reviewed: {mark}", f'=COUNTIF({gap_rng("review")},"{mark}")',
                           sum(g["review"] == verdict for g in gaps)))
        checks.append(("⏳ not reviewed yet", f'=COUNTIF({gap_rng("review")},"⏳ not reviewed")',
                       sum(not g["review"] and g["quasar_status"] in CANDIDATES for g in llk)))
        formats = [g for g in gaps if g["kind"] == "data_format"]
        checks.append(("Data formats missing on Quasar",
                        f'=COUNTIFS({gap_rng("kind")},"data_format",{gap_rng("quasar_status")},"missing")',
                        sum(g["quasar_status"] == "missing" for g in formats)))
        for name, formula, value in checks:
            rows.append([name, formula])
            expected[f"Info!B{len(rows)}"] = value

    offset = len(rows)
    rows += summary
    bold += [offset + h for h in summary_headings]
    if constant:
        rows.append([])
        bold.append(len(rows))
        rows.append(["Same in every ttnn call row"])
        rows += [[name, cell(text)] for name, text in constant]
    info = new_tab("Info", "run.json, summary.md", rows, list(INFO_WIDTHS), bold, header=False)
    info["cell_notes"] = cell_notes
    return info, expected


def _wrap_all(sheet_id: int) -> dict:
    return {"repeatCell": {
        "range": {"sheetId": sheet_id},
        "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}},
        "fields": "userEnteredFormat.wrapStrategy,userEnteredFormat.verticalAlignment",
    }}


def _width(sheet_id: int, index: int, pixels: int) -> dict:
    return {"updateDimensionProperties": {
        "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": index, "endIndex": index + 1},
        "properties": {"pixelSize": pixels}, "fields": "pixelSize",
    }}


def _bold(sheet_id: int, row: int, columns: int, grey: bool) -> dict:
    fmt: dict = {"textFormat": {"bold": True}}
    fields = "userEnteredFormat.textFormat.bold"
    if grey:
        fmt["backgroundColorStyle"] = {"rgbColor": HEADER_GREY}
        fields += ",userEnteredFormat.backgroundColorStyle"
    return {"repeatCell": {"range": {"sheetId": sheet_id, "startRowIndex": row, "endRowIndex": row + 1,
                                     "startColumnIndex": 0, "endColumnIndex": columns},
                           "cell": {"userEnteredFormat": fmt}, "fields": fields}}


def spec(run_dir: pathlib.Path, info_id: int = 0, first_id: int = 1, max_tab_kb: int = 300) -> dict:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    source, destinations = run_targets(run)
    targets = (source, *destinations)
    ops = read_csv(run_dir / "op_table.csv")
    op_tab, op_columns, constant = op_table_view(ops, read_header(run_dir / "op_table.csv"), targets)
    info_tab, expected = info_view(run_dir, run, targets, op_columns, constant, ops)

    tabs = [info_tab]
    device_ops_path = run_dir / "device_ops.csv"
    if destinations and device_ops_path.exists():
        device_tab, device_expected = device_ops_view(read_csv(device_ops_path), targets)
        tabs.append(device_tab)
        expected |= device_expected
    kernel_ops_path = run_dir / "kernel_ops.csv"
    if kernel_ops_path.exists():
        kernel_ops = read_csv(kernel_ops_path)
        tabs.append(kernel_ops_view(kernel_ops, ops, targets, op_columns))
        expected[f"{quote(KERNEL_OPS_TAB)}!E{len(kernel_ops) + 2}"] = sum(int(k["static_launches"] or 0)
                                                                       for k in kernel_ops)
    tabs.append(op_tab)
    # Every tab a run can have, in order: (title, built tab or None, why it is missing).
    directory: list[tuple[str, dict | None, str]] = [("Info", info_tab, "")]
    directory += [(t["title"], t, "") for t in tabs[1:]]
    ran = set(methods_used(run))
    for name, title in SHEET_TABS:
        if name == "device_ops" and not any(t["title"] == DEVICE_OPS_TAB for t in tabs):
            why = ("not produced: the run has no destination" if not destinations else
                   "not produced in this run (device_ops.csv is missing; kernel_ops.py writes it)")
            directory.append((DEVICE_OPS_TAB, None, why))
            continue
        if name in ("op_table", "kernel_ops", "device_ops"):
            continue
        if name == "kernel_test_results":
            quasar = [t for t in targets if t in TEST_TARGETS]
            if not quasar:
                directory.append((TTSIM_TESTS_TAB, None, "not produced: the run has no Quasar target"))
                continue
            entries = [(run_dir / f"kernel_test_results_{t}.csv", f"{TTSIM_TESTS_TAB} ({label(t)})") for t in quasar]
        else:
            entries = [(run_dir / f"{name}.csv", title)]
        for path, tab_title in entries:
            if not path.exists():
                method = TAB_METHOD[name]
                why = ("not produced: the TTsim kernel-op tests have not run" if method == "TTsim" else
                       f"not produced: the {METHOD_LABELS[method][0].lower()}{METHOD_LABELS[method][1:]} has not run" if method not in ran else
                       f"not produced in this run ({path.name} is missing)")
                directory.append((tab_title, None, why))
                continue
            tab = csv_view(path, tab_title)
            size = len(json.dumps(tab["values"], ensure_ascii=False).encode())
            size_kb = -(-size // 1024)
            if size > max_tab_kb * 1024:
                directory.append((tab_title, None, f"in the run directory only: {path.name}, {len(tab['values']) - 1} "
                                                   f"rows ({size_kb} KB), too large for the sheet"))
                continue
            tabs.append(tab)
            directory.append((tab_title, tab, ""))
    tabs += [csv_view(run_dir / f"changes_{name}.csv", f"Changes: {title}") for name, title in SHEET_TABS
             if (run_dir / f"changes_{name}.csv").exists()]
    directory += [(t["title"], t, "") for t in tabs if t["title"].startswith("Changes: ")]

    info_rows = info_tab["values"]
    if not any(t["title"] == LLK_GAPS_TAB for t in tabs):
        # The gap counts cannot point at a tab the sheet does not have; show their values instead.
        for n, row in enumerate(info_rows, 1):
            if len(row) > 1 and isinstance(row[1], str) and quote(LLK_GAPS_TAB) in row[1]:
                row[1] = expected.pop(f"Info!B{n}")
    info_rows.append([])
    info_tab["bold"].append(len(info_rows))
    info_rows.append(["Tabs", "What it holds", "Filled by", "In this sheet"])
    for title, tab, why in directory:
        description = tab_description(title)
        if description:
            info_rows.append([title, cell(description[0]), cell(description[1]), cell("yes" if tab else why)])

    # The kept first tab may be smaller than Info needs; size it like the added tabs.
    info_size = {"rowCount": len(info_tab["values"]) + 50, "columnCount": max(len(r) for r in info_tab["values"])}
    requests = [{"updateSheetProperties": {
        "properties": {"sheetId": info_id, "title": "Info", "gridProperties": info_size},
        "fields": "title,gridProperties.rowCount,gridProperties.columnCount"}}]
    for index, tab in enumerate(tabs):
        width = max(len(r) for r in tab["values"])
        tab["sheetId"] = info_id if index == 0 else first_id + index - 1
        if index:
            requests.append({"addSheet": {"properties": {
                "sheetId": tab["sheetId"], "title": tab["title"], "index": index,
                "gridProperties": {"rowCount": len(tab["values"]) + 50, "columnCount": width,
                                   "frozenRowCount": 1 if tab["header"] else 0}}}})
        requests.append(_wrap_all(tab["sheetId"]))
        requests += [_width(tab["sheetId"], i, w) for i, w in enumerate(tab.pop("widths")[:width])]
        header = tab.pop("header")
        requests += [_bold(tab["sheetId"], r, width, header and r == 0) for r in tab.pop("bold")]
        notes = tab.pop("notes", None)
        if notes and any(notes):
            requests.append({"updateCells": {
                "range": {"sheetId": tab["sheetId"], "startRowIndex": 0, "endRowIndex": 1,
                          "startColumnIndex": 0, "endColumnIndex": len(notes)},
                "rows": [{"values": [{"note": n} for n in notes]}], "fields": "note"}})
        requests += [{"updateCells": {
            "range": {"sheetId": tab["sheetId"], "startRowIndex": r, "endRowIndex": r + 1,
                      "startColumnIndex": c, "endColumnIndex": c + 1},
            "rows": [{"values": [{"note": note}]}], "fields": "note"}} for r, c, note in tab.pop("cell_notes", [])]
    return {"tabs": tabs, "requests": requests, "expected": expected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--info-sheet-id", type=int, default=0)
    parser.add_argument("--first-sheet-id", type=int, default=1)
    parser.add_argument("--max-tab-kb", type=int, default=300,
                        help="a CSV tab larger than this stays in the run directory; Info lists it")
    args = parser.parse_args()
    print(json.dumps(spec(args.run_dir, args.info_sheet_id, args.first_sheet_id, args.max_tab_kb), ensure_ascii=False,
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
