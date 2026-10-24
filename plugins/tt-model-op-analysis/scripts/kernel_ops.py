"""Per-kernel-op summary of op_table.csv: one row per device op and program factory.

Measured counts come from Tracy, which reports op codes but not program factories, so they are
given per op code and repeated on every kernel op that shares it. Graph launches come from
graph_ops.csv, which names the factory, so they are per kernel op; a factory only the graph
capture saw gets its own row. `sources` lists the methods that saw each kernel op.

With destinations it also writes device_ops.csv: one row per device op with its op-table rows
counted by status per destination and the verdict from device_op_support.csv, or ⏳ without one.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from opa_schema import (  # noqa: E402
    NOT_CHECKED, STATUS_COUNT_FIELDS, TEST_TARGETS, device_ops_columns, kernel_ops_columns, op_key, read_csv,
    run_targets, write_csv,
)

STATUS_ORDER = ("✅", "⚠️", "❌", "n/a")


def measured_counts(path: pathlib.Path | None, inferences: int = 1) -> dict[str, int]:
    """Launches per inference by op code; rounded down when a count is not a whole multiple."""
    if path is None:
        return {}
    rows = [r for r in read_csv(path) if r["op_type"] == "tt_dnn_device"]
    return {code: n // inferences for code, n in collections.Counter(r["op_code"] for r in rows).items()}


def _tests(cell: str) -> list[str]:
    return [t.strip() for t in cell.split(";") if t.strip()]


def factory_matches(graph_row: dict, factory: str) -> bool:
    """A graph factory type names op_table's canonical factory, possibly with more namespaces."""
    full = graph_row["program_factory_type"]
    return bool(factory) and (factory in (full, graph_row["program_factory"]) or full.endswith("::" + factory))


def _number(text: str) -> float:
    return float(text) if text else 0.0


def _launches(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.2f}"


def _sources(entry: dict, shared_op_code: bool) -> str:
    # Tracy counts op codes; when several kernel ops share one, it cannot say which of them ran.
    tracy = "tracy (op code)" if shared_op_code else "tracy"
    seen = [entry["origin"] not in ("measured", "graph") and "static",
            entry["measured_launches_op_code"] not in ("", "0") and tracy,
            entry["graph_launches"] not in ("", "0") and "graph"]
    return ", ".join(s for s in seen if s)


def _same_device_op(device_op: str, graph_device_ops: str) -> bool:
    """True unless both name a device op and they differ (`prim::x` and `ttnn::prim::x` are one)."""
    if not device_op or not graph_device_ops:
        return True
    return device_op.removeprefix("ttnn::") in {d.removeprefix("ttnn::") for d in graph_device_ops.split("; ")}


def match_graph(groups: dict[tuple[str, str, str], list[dict]], graph: list[dict]):
    """graph_ops rows per kernel-op group, and the rows no group claims.

    Groups naming a factory claim first; a measured-only group (a Tracy op code with no factory)
    then takes the remaining rows of its op code.
    """
    unmatched, hits = list(graph), {}
    for key in sorted(groups, key=lambda k: not k[1]):
        device_op, factory, code = key
        if factory:
            found = [g for g in unmatched if (not code or g["op_code"] == code) and factory_matches(g, factory)
                     and _same_device_op(device_op, g["device_op"])]
        else:
            found = [g for g in unmatched if code and g["op_code"] == code]
        unmatched = [g for g in unmatched if g not in found]
        hits[key] = found
    return hits, unmatched


def aggregate(rows: list[dict], source: str, destinations: list[str], measured: dict[str, int],
              node_results: dict[str, dict[str, str]] | None = None, graph: list[dict] | None = None) -> list[dict]:
    """node_results maps a target to {node id: result} from its kernel_test_results file; graph is
    graph_ops.csv."""
    node_results = node_results or {}
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["device_op"], row["program_factory"], row["op_code"]), []).append(row)
    graph_hits, unmatched = match_graph(groups, graph or [])
    static_by_code: collections.Counter = collections.Counter()
    for (_, _, code), members in groups.items():
        static_by_code[code] += sum(int(m[f"{source}:launches"] or 0) for m in members if m["origin"] != "measured")
    out = []
    for (device_op, factory, code), members in groups.items():
        static_rows = [m for m in members if m["origin"] != "measured"]
        entry = {
            "device_op": device_op, "program_factory": factory, "op_code": code,
            "origin": "measured" if not static_rows else
                      ("static+measured" if any(m["origin"] == "static+measured" for m in members) else "static"),
            "op_table_ids": ",".join(m["id"] for m in members),
            "static_launches": str(sum(int(m[f"{source}:launches"] or 0) for m in static_rows)),
            "static_launches_op_code": str(static_by_code[code]),
            "measured_launches_op_code": str(measured.get(code, 0)),
        }
        hits = graph_hits[(device_op, factory, code)]
        entry["graph_launches"] = _launches(sum(_number(g["launches_per_inference"]) for g in hits))
        if hits and not factory:
            # A measured-only row learns its factories from the graph capture.
            entry["device_op"] = "; ".join(dict.fromkeys(g["device_op"] for g in hits if g["device_op"]))
            entry["program_factory"] = "; ".join(dict.fromkeys(g["program_factory"] for g in hits))
        for t in (source, *destinations):
            statuses = collections.Counter(m[f"{t}:status"] for m in members)
            entry[f"{t}:statuses"] = "; ".join(f"{s} {statuses[s]}" for s in STATUS_ORDER if statuses[s])
            if t in TEST_TARGETS:
                entry[f"{t}:tests"] = "; ".join(sorted({x for m in members for x in _tests(m[f"{t}:tests"])}))
                if t in node_results:
                    # A row's result is the worst of its nodes; each node shows only its own run.
                    results = {f"{x}: {node_results[t][x]}" for m in members for x in _tests(m[f"{t}:tests"])
                               if x in node_results[t]}
                else:
                    results = {f"{x}: {m[f'{t}:test_result']}" for m in members for x in _tests(m[f"{t}:tests"])
                               if m[f"{t}:test_result"] != "not run"}
                entry[f"{t}:test_results"] = "; ".join(sorted(results))
        out.append(entry)
    empty_statuses = {f"{t}:statuses": "" for t in (source, *destinations)}
    for g in unmatched:
        entry = {"device_op": g["device_op"], "program_factory": g["program_factory"], "op_code": g["op_code"],
                 "origin": "graph", "op_table_ids": "", "static_launches": "0", "static_launches_op_code": "0",
                 "measured_launches_op_code": str(measured.get(g["op_code"], 0)),
                 "graph_launches": _launches(_number(g["launches_per_inference"]))} | empty_statuses
        out.append(entry)
    known = {code for (_, _, code) in groups} | {g["op_code"] for g in unmatched}
    for code, count in measured.items():
        if code not in known:
            entry = {"device_op": "", "program_factory": "", "op_code": code, "origin": "measured",
                     "op_table_ids": "", "static_launches": "0", "static_launches_op_code": "0",
                     "measured_launches_op_code": str(count), "graph_launches": "0"} | empty_statuses
            out.append(entry)
    per_code = collections.Counter(e["op_code"] for e in out)
    for n, entry in enumerate(out, 1):
        entry["id"] = str(n)
        entry["sources"] = _sources(entry, per_code[entry["op_code"]] > 1)
    return out


def device_ops(rows: list[dict], source: str, destinations: list[str], support: list[dict]) -> list[dict]:
    """One row per device op: its op-table rows counted by status per destination, and the
    agent's verdict from device_op_support.csv on whether the destination runs the op at all."""
    verdicts = {r["device_op"]: r for r in support}
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(op_key(row), []).append(row)
    out = []
    for n, (device_op, members) in enumerate(groups.items(), 1):
        entry = {
            "id": str(n), "device_op": device_op,
            "program_factories": "; ".join(dict.fromkeys(m["program_factory"] for m in members if m["program_factory"])),
            "ttnn_apis": "; ".join(dict.fromkeys(m["ttnn_api"] for m in members if m["ttnn_api"])),
            "op_table_ids": ",".join(m["id"] for m in members),
            f"{source}:launches": str(sum(int(m[f"{source}:launches"] or 0) for m in members)),
        }
        verdict = verdicts.get(device_op, {})
        for t in destinations:
            statuses = collections.Counter(m[f"{t}:status"] for m in members)
            entry |= {f"{t}:{f}": str(statuses[s]) for s, f in STATUS_COUNT_FIELDS.items()}
            entry[f"{t}:op_support"] = verdict.get(f"{t}:op_support") or NOT_CHECKED
            entry[f"{t}:op_evidence"] = verdict.get(f"{t}:op_evidence", "")
        out.append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--measured", type=pathlib.Path)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    rows = read_csv(args.run_dir / "op_table.csv")
    measured = args.measured or (pathlib.Path(run["measured_ops"]) if run.get("measured_ops") else None)
    counts = measured_counts(measured, int(run.get("measured_inferences", 1)))
    source, destinations = run_targets(run)
    node_results = {}
    for t in (source, *destinations):
        path = args.run_dir / f"kernel_test_results_{t}.csv"
        if t in TEST_TARGETS and path.exists():
            node_results[t] = {r["node_id"]: r["result"] for r in read_csv(path)}
    graph_path = args.run_dir / "graph_ops.csv"
    graph = read_csv(graph_path) if graph_path.exists() else None
    out = aggregate(rows, source, destinations, counts, node_results, graph)
    write_csv(args.run_dir / "kernel_ops.csv", kernel_ops_columns(source, destinations), out)
    graph_only = sum(e["origin"] == "graph" for e in out)
    print(f"{len(out)} kernel ops" + (f", {graph_only} seen only by the graph capture" if graph_only else ""))
    if destinations:
        support_path = args.run_dir / "device_op_support.csv"
        support = read_csv(support_path) if support_path.exists() else []
        ops = device_ops(rows, source, destinations, support)
        write_csv(args.run_dir / "device_ops.csv", device_ops_columns(source, destinations), ops)
        unchecked = sum(o[f"{t}:op_support"] == NOT_CHECKED for o in ops for t in destinations)
        print(f"{len(ops)} device ops" + (f", {unchecked} destination verdicts not checked yet "
                                          "(device_op_support.csv)" if unchecked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
