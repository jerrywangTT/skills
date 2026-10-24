"""List and mark a run's measurement items.

Items:
- `source:tracy`: entry test under Tracy on a Blackhole or Wormhole source device.
- `source:graph`: ttnn graph capture of the entry test on that device.
- `source:cache`: the kernel cache of the entry test on that device, decomposed with elfray.
  One run of the entry test can produce all three (measured-source.md).
- `source:kernel-tests <status>`: kernel-op tests of a Quasar source's rows with that status.
- `<dest>:kernel-tests <status>`: the same for a Quasar destination.

State comes from the run's files: `measured_ops`, `graph_capture` and `kernel_cache` in run.json for
the source items, recorded results for test
nodes (an `error` or `timeout` result decides nothing, so its node stays to do). Without a plan
file the nodes come from op_table.csv, as `kernel_tests.py plan` would write them. `mark` records `done` or `failed` (with a note) in run.json `measurements` for what files cannot
show, such as a Tracy run that could not start. Files win over a mark once all nodes have results.
TTsim estimates use the mean duration of tests already run in this run; until one has run the
estimate is unknown.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kernel_tests import plan_rows, read_plan  # noqa: E402
from opa_schema import (  # noqa: E402
    FAILURE_CAUSES, LAUNCH_PROFILES, METHOD_ITEMS, TEST_TARGETS, UNDECIDED_RESULTS, read_csv, read_results, run_targets,
)

STATUS_ORDER = ("❌", "⚠️", "✅")
MARK_STATES = ("done", "failed")


def _results(run_dir: pathlib.Path) -> dict[tuple[str, str], dict]:
    out = {}
    for path in run_dir.glob("kernel_test_results_*.csv"):
        target = path.stem.removeprefix("kernel_test_results_")
        for row in read_results(path):
            out[(target, row["node_id"])] = row
    return out


def _estimate(n: int, durations: list[float]) -> str:
    if n == 0:
        return "0"
    if not durations:
        return "unknown (the first TTsim test calibrates it)"
    return f"~{max(1, round(n * statistics.mean(durations) / 60))} min"


def _derived(run_dir: pathlib.Path, run: dict) -> list[dict]:
    source, destinations = run_targets(run)
    results = _results(run_dir)
    durations = [float(r["duration_s"]) for r in results.values() if r.get("duration_s")]
    out = []
    if source in LAUNCH_PROFILES:
        timeout = run.get("ci_pytest_timeout_s")
        for item, key in METHOD_ITEMS.values():
            out.append({"item": item, "nodes": 1, "done_nodes": int(bool(run.get(key))),
                        "estimate": f"<= {timeout} s" if timeout else "unknown"})
    for target in (source, *destinations):
        if target not in TEST_TARGETS:
            continue
        prefix = "source" if target == source else target
        plan_path, op_path = run_dir / f"kernel_tests_plan_{target}.csv", run_dir / "op_table.csv"
        if plan_path.exists():
            plan = read_plan(plan_path)
        elif op_path.exists():
            plan = plan_rows(run, read_csv(op_path), target)
        else:
            continue
        groups = [(f"{prefix}:kernel-tests {status}", [p for p in plan if status in p["row_statuses"].split(";")])
                  for status in STATUS_ORDER]
        for name, nodes in groups:
            if nodes:
                recorded = [results.get((target, p["node_id"]), {}).get("result") for p in nodes]
                done = sum(r not in (None, *UNDECIDED_RESULTS) for r in recorded)
                out.append({"item": name, "nodes": len(nodes), "done_nodes": done,
                            "estimate": _estimate(len(nodes) - done, durations)})
    return out


def items(run_dir: pathlib.Path) -> list[dict]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    marks = run.get("measurements", {})
    out = []
    for item in _derived(run_dir, run):
        mark = marks.get(item["item"], {})
        if item["done_nodes"] == item["nodes"]:
            state = "done"
        elif mark.get("state") in MARK_STATES:
            state = mark["state"]
        else:
            state = "partial" if item["done_nodes"] else "pending"
        out.append(item | {"state": state, "note": mark.get("note", "")})
    return out


def mark(run_dir: pathlib.Path, item: str, state: str, note: str, cause: str = "") -> int:
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    known = [i["item"] for i in _derived(run_dir, run)]
    if item not in known:
        print(f"ERROR: unknown item '{item}'; this run has {known}")
        return 2
    if cause and state != "failed":
        print("ERROR: --cause applies to a failed item only")
        return 2
    record = {"state": state, "date": datetime.date.today().isoformat(), "note": note}
    if cause:
        record["cause"] = cause
    run.setdefault("measurements", {})[item] = record
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{item}: {state}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("run_dir", type=pathlib.Path)
    p_list.add_argument("--json", action="store_true")
    p_list.add_argument("--pending", action="store_true", help="only items that are not done")
    p_mark = sub.add_parser("mark")
    p_mark.add_argument("run_dir", type=pathlib.Path)
    p_mark.add_argument("item")
    p_mark.add_argument("state", choices=MARK_STATES)
    p_mark.add_argument("--note", default="")
    p_mark.add_argument("--cause", choices=FAILURE_CAUSES,
                        help="for a failed source item: test (the entry test hung, failed or missed its PCC in "
                             "this run), tool (the instrument stopped it), not-run (it could not start)")
    args = parser.parse_args()
    if args.cmd == "mark":
        return mark(args.run_dir, args.item, args.state, args.note, args.cause or "")
    out = [i for i in items(args.run_dir) if not (args.pending and i["state"] == "done")]
    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        for n, item in enumerate(out, 1):
            note = f" ({item['note']})" if item["note"] else ""
            print(f"{n}. {item['item']}: {item['state']}{note}, "
                  f"{item['done_nodes']}/{item['nodes']} done, {item['estimate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
