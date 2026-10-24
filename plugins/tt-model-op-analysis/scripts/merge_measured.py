"""Merge a Tracy measured_ops.csv of the entry test into op_table.csv.

Static rows whose op code appears in the measured run become `static+measured`. Op codes that only
the measured run shows become `measured` rows, inserted after the static row that ran just before
their first launch, so the table stays in execution order. Launch counts are divided by the number
of inferences in the measured window. Re-running replaces earlier measured rows.

With --entry-pass the entry test passed in the measured run, so the source profile has a recorded
passing run (references/target-status.md): it is added to run.json `entry_test_passes` and every
source `⚠️` row is decided again from what the run launched. --entry-fail records a failed run in
`entry_test_failure`; statuses stay as they are.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from opa_schema import (  # noqa: E402
    ALWAYS, BUILD_TYPES, LAUNCH_PROFILES, NOT_TRACED, TARGET_LABELS, TEST_TARGETS, ci_job, op_table_columns, read_csv,
    run_targets, write_csv,
)


def _measured_row(code: str, count: int, first: dict, source: str, destinations: list[str], csv_name: str) -> dict:
    row = {
        "stage": f"measured only {code}", "origin": "measured", "ttnn_api": "", "op_code": code,
        "device_op": "", "program_factory": "", "call_site": NOT_TRACED, "launch_condition": "always",
        "confidence": "unverified",
        "dispatch_evidence": f"seen in {csv_name} (first at measured row {first['id']}); not traced statically",
        "grid_dependency": NOT_TRACED,
        f"{source}:shapes": f"{first['inputs']} -> {first['outputs']}", f"{source}:launches": str(count),
        f"{source}:status": "✅", f"{source}:evidence": f"executed by the entry test on {source} ({csv_name})",
    }
    for target in destinations:
        if target in LAUNCH_PROFILES:
            row[f"{target}:shapes"] = NOT_TRACED
            row[f"{target}:launches"] = ""
        row[f"{target}:status"] = "⚠️"
        row[f"{target}:evidence"] = "measured-only op; kernel-op lookup on this target still needed"
    for target in (source, *destinations):
        if target in TEST_TARGETS:
            row[f"{target}:tests"] = ""
            row[f"{target}:test_result"] = "not run"
    return row


def merge(rows: list[dict], measured: list[dict], source: str, destinations: list[str], csv_name: str,
          inferences: int = 1) -> list[dict]:
    rows = [dict(r, origin="static") for r in rows if r["origin"] != "measured"]
    device = [m for m in measured if m["op_type"] == "tt_dnn_device"]
    counts = collections.Counter(m["op_code"] for m in device)
    static_codes = {r["op_code"] for r in rows}
    for row in rows:
        if row["op_code"] in counts:
            row["origin"] = "static+measured"
    bad = {c: n for c, n in counts.items() if c not in static_codes and n % inferences}
    if bad:
        raise ValueError(f"launch counts not divisible by {inferences} inferences: {bad}")
    # Walk the measured stream with a cursor over the static rows. A row can cover several repeated
    # modules, so the stream may return to an earlier row; the cursor only moves forward.
    first_seen: dict[str, dict] = {}
    insert_after: dict[str, int] = {}
    cursor = -1
    for m in device:
        code = m["op_code"]
        if code in static_codes:
            ahead = next((i for i in range(max(cursor, 0), len(rows)) if rows[i]["op_code"] == code), None)
            if ahead is not None:
                cursor = ahead
        elif code not in first_seen:
            first_seen[code] = m
            insert_after[code] = cursor

    def new_rows(after: int) -> list[dict]:
        return [_measured_row(c, counts[c] // inferences, first_seen[c], source, destinations, csv_name)
                for c in first_seen if insert_after[c] == after]

    out = new_rows(-1)
    for i, row in enumerate(rows):
        out.append(row)
        out += new_rows(i)
    for n, row in enumerate(out, 1):
        row["id"] = str(n)
    return out


def entry_pass_record(previous: dict, source: str, result: str, date: str, item: str = "source:tracy",
                      build: str = "") -> dict:
    """entry_test_passes with this local run added; a CI record already there is kept."""
    record = {k: v for k, v in previous.items() if k == "ci"}
    record.update({"by": f"{item} on a local {source} device", "date": date, "result": result, "build": build})
    return record


def record_entry_result(run: dict, source: str, date: str, item: str, passed: str | None, failed: str | None,
                        build: str = "") -> None:
    """Record the entry test's outcome in this run. A pass replaces an earlier local failure; a failure is kept
    beside earlier passes, and the later of the two decides what the sheet shows."""
    if passed:
        run["entry_test_passes"] = entry_pass_record(run.get("entry_test_passes") or {}, source, passed, date, item,
                                                     build)
        run.pop("entry_test_failure", None)
    elif failed:
        run["entry_test_failure"] = {"by": f"{item} on a local {source} device", "date": date, "result": failed,
                                     "build": build}
        if (run.get("entry_test_passes") or {}).get("by"):
            print("WARNING: source statuses decided from the earlier local pass now rest on a superseded run; "
                  "rerun the entry test or decide them again")


def add_build_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--build", choices=BUILD_TYPES,
                        help="CMAKE_BUILD_TYPE of the tt-metal build the entry test ran on; Release compiles TT_ASSERT out")


def decide_source_statuses(rows: list[dict], measured: list[dict], source: str, inferences: int,
                           entry_passes: dict, ci: str, how: str = "under Tracy") -> list[str]:
    """Upgrade source `⚠️` rows the passing entry test covers; return warnings for `❌` rows it ran.

    Only `⚠️` is revisited: it is what a profile gets while no passing run is recorded. `✅`, `❌`
    and `n/a` rest on other evidence and stay.
    """
    label = TARGET_LABELS[source]
    where = f"in CI ({ci}) and " if ci else f"No CI workflow runs the entry test on {label}; it passed "
    passed = (f"{where}on a local {label} {how} ({entry_passes['result']}, {entry_passes['date']})")
    counts = collections.Counter(m["op_code"] for m in measured if m["op_type"] == "tt_dnn_device")
    static_launches: collections.Counter = collections.Counter()
    for row in rows:
        if row["origin"] != "measured":
            static_launches[row["op_code"]] += int(row[f"{source}:launches"] or 0)
    warnings = []
    for row in rows:
        status_col, evidence_col = f"{source}:status", f"{source}:evidence"
        code = row["op_code"]
        ran = counts[code] / inferences
        if row["origin"] == "measured":
            continue
        if row[status_col] == "❌" and ran:
            warnings.append(f"row {row['id']}: {source} status is ❌ but the passing entry test launched {code}")
        if row[status_col] != "⚠️":
            continue
        if not ran:
            row[evidence_col] = f"{passed}, but launched no {code}; this branch did not run there."
        elif row["launch_condition"].strip() != ALWAYS and ran < static_launches[code]:
            # Tracy names the op code, not the call site, so a skipped conditional launch is not
            # attributable to one row.
            row[evidence_col] = (f"{passed}; it launched {code} {ran:g} of {static_launches[code]} potential times"
                                 " per inference, so whether this conditional launch ran is not known.")
        else:
            row[status_col] = "✅"
            row[evidence_col] = f"{passed}, launching {code} {ran:g}x per inference."
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("measured_ops", type=pathlib.Path)
    parser.add_argument("--tracy-mode", choices=("live", "trace"), required=True,
                        help="live: the test called the forward in the profiled window; trace: it replayed a trace")
    parser.add_argument("--inferences", type=int, default=1,
                        help="forward passes inside the measured window (1 for a warm signposted window)")
    parser.add_argument("--entry-pass", metavar="RESULT",
                        help="the entry test passed in this run; RESULT is its pass line (for example the PCC)")
    parser.add_argument("--entry-fail", metavar="RESULT",
                        help="the entry test failed in this run; RESULT is its failure line")
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="date of the measured run")
    add_build_argument(parser)
    args = parser.parse_args()
    if args.entry_pass and args.entry_fail:
        print("ERROR: give --entry-pass or --entry-fail, not both")
        return 2
    if (args.entry_pass or args.entry_fail) and not args.build:
        print("ERROR: --entry-pass and --entry-fail need --build, the tt-metal build type the test ran on")
        return 2
    run_path = args.run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    op_path = args.run_dir / "op_table.csv"
    source, destinations = run_targets(run)
    measured = read_csv(args.measured_ops)
    try:
        merged = merge(read_csv(op_path), measured, source, destinations, args.measured_ops.name, args.inferences)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    record_entry_result(run, source, args.date, "source:tracy", args.entry_pass, args.entry_fail, args.build)
    if args.entry_pass:
        for warning in decide_source_statuses(merged, measured, source, args.inferences, run["entry_test_passes"],
                                              ci_job(run)):
            print(f"WARNING: {warning}")
    write_csv(op_path, op_table_columns(source, destinations), merged)
    run["measured_ops"] = str(args.measured_ops)
    run["measured_inferences"] = args.inferences
    run["tracy_mode"] = args.tracy_mode
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(merged)} rows, {sum(r['origin'] == 'measured' for r in merged)} measured-only")
    if args.entry_pass:
        statuses = collections.Counter(r[f"{source}:status"] for r in merged)
        print(f"{source} statuses: " + ", ".join(f"{s} {n}" for s, n in statuses.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
