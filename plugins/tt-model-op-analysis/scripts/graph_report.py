"""Turn a ttnn graph capture of the entry test into graph_ops.csv and graph_host_transfers.csv.

graph_ops.csv has one row per (op code, program factory) the run launched, with the top-level ttnn
calls that launched it and launches per inference. The factory is the one the device op really
selected for those arguments, so this is the exact kernel-op list of the run.

graph_host_transfers.csv lists the host transfers (to_torch, from_torch, from_device, to_device,
copy_host_to_device, copy_device_to_host) between the first and the last launch, with the launches
around each. A model
whose forward is entirely on device shows none there except the next inference's input upload and
the previous one's readback; anything else is host work inside the forward.

Records the capture in run.json `graph_capture`. Only launches that go through ttnn dispatch are
captured: a replayed device trace launches nothing new, so pass the number of forwards the test
dispatched (not replayed) as --inferences.

With --entry-pass the entry test passed in the captured run: like merge_measured.py --entry-pass,
it records the pass in run.json `entry_test_passes` and decides every source `⚠️` row of
op_table.csv again from the op codes the run launched. --entry-fail records a failed run.

`graph_report.py <run dir> --decide` does only that decision, from the graph_ops.csv and the pass
already recorded, without the capture: for a preflight capture taken before op_table.csv existed.

Peak DRAM is the largest total of DRAM buffers allocated at once on one device during the capture,
which starts after the device opens; buffers allocated before that are not counted.

The capture is parsed whole, so the host needs several times its size in memory.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from merge_measured import add_build_argument, decide_source_statuses, record_entry_result  # noqa: E402
from opa_schema import (  # noqa: E402
    GRAPH_HOST_TRANSFERS_COLUMNS, GRAPH_OPS_COLUMNS, ci_job, latest_entry_failure, op_table_columns, read_csv,
    run_targets, write_csv,
)

FACTORY = "program_factory_type"
# The ttnn Python calls first, then the C++ calls they make. ttnn.to_torch runs Tensor::cpu and the
# torch conversion one after the other, so only the outermost transfer node is counted.
TRANSFERS = {
    "ttnn.to_torch": "to_torch", "ttnn.from_device": "from_device", "ttnn.from_torch": "from_torch",
    "ttnn.to_device": "to_device", "ttnn.copy_host_to_device_tensor": "copy_host_to_device",
    "ttnn.copy_device_to_host_tensor": "copy_device_to_host",
    "tt::tt_metal::detail::convert_tt_tensor_to_framework_tensor": "to_torch",
    "Tensor::cpu": "from_device",
    "ttnn::convert_python_tensor_to_tt_tensor": "from_torch",
    "Tensor::to_device": "to_device",
    "tt::tt_metal::copy_to_device": "copy_host_to_device",
}
# Older captures name the device-op node generically; the device op is then its parent.
GENERIC_DEVICE_NODE = "Device Operation"


def short_factory(full: str) -> str:
    """Drop leading namespaces (lower-case components) outside template brackets.

    `ttnn::operations::pool::Pool2D::MultiCore` -> `Pool2D::MultiCore`, keeping a factory nested in
    its operation class the way op_table.csv names it.
    """
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(full):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == ":" and depth == 0 and full[i:i + 2] == "::":
            parts.append(full[start:i])
            start = i + 2
    parts.append(full[start:])
    parts = [p for p in parts if p]
    while len(parts) > 1 and parts[0][:1].islower():
        parts.pop(0)
    return "::".join(parts)


def _graph(document) -> list[dict]:
    return document["graph"] if isinstance(document, dict) else document


def walk(nodes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Launches and host transfers in capture order."""
    launches, transfers, stack = [], [], []
    for node in nodes:
        kind = node.get("node_type")
        if kind == "function_end":
            if stack:
                stack.pop()
            continue
        if kind != "function_start":
            continue
        params = node.get("params", {})
        name = params.get("name", "")
        if FACTORY in params:
            device_op = next((s for s in reversed(stack) if s.startswith("ttnn::prim::")), "")
            api = next((s for s in stack if s.startswith("ttnn.")), stack[0] if stack else name)
            op_code = name if name != GENERIC_DEVICE_NODE else device_op.removeprefix("ttnn::prim::")
            hit = params.get("program_cache_hit")
            launches.append({"op_code": op_code, "device_op": device_op, "factory": params[FACTORY],
                             "api": api, "cache_hit": hit is True or hit == "true"})
        elif name in TRANSFERS and not any(s in TRANSFERS for s in stack):
            transfers.append({"transfer": TRANSFERS[name], "position": len(launches)})
        stack.append(name)
    return launches, transfers


def peak_dram(nodes: list[dict]) -> dict[str, int]:
    """Peak bytes of DRAM buffers live at once, per device id, over the capture."""
    live: dict[tuple[str, str], int] = {}
    current: dict[str, int] = {}
    peak: dict[str, int] = {}
    for node in nodes:
        kind = node.get("node_type")
        if kind not in ("buffer_allocate", "buffer_deallocate"):
            continue
        params = node.get("params", {})
        if params.get("type") != "DRAM":
            continue
        device, key = str(params.get("device_id", "0")), (str(params.get("device_id", "0")), str(params.get("address")))
        if kind == "buffer_allocate":
            size = int(params.get("size", 0))
            live[key] = size
            current[device] = current.get(device, 0) + size
            peak[device] = max(peak.get(device, 0), current[device])
        elif key in live:
            current[device] -= live.pop(key)
    return peak


def per_inference(count: int, inferences: int) -> str:
    return str(count // inferences) if count % inferences == 0 else f"{count / inferences:.2f}"


def ops_rows(launches: list[dict], inferences: int) -> tuple[list[dict], list[str]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for launch in launches:
        groups.setdefault((launch["op_code"], launch["factory"]), []).append(launch)
    rows, uneven = [], []
    for n, ((op_code, factory), members) in enumerate(groups.items(), 1):
        if len(members) % inferences:
            uneven.append(f"{op_code} / {short_factory(factory)}: {len(members)} launches")
        device_ops = sorted({m["device_op"] for m in members if m["device_op"]})
        rows.append({
            "id": str(n), "op_code": op_code, "device_op": "; ".join(device_ops),
            "program_factory": short_factory(factory), "program_factory_type": factory,
            "ttnn_apis": "; ".join(dict.fromkeys(m["api"] for m in members)),
            "launches": str(len(members)), "launches_per_inference": per_inference(len(members), inferences),
            "cache_misses": str(sum(not m["cache_hit"] for m in members)),
        })
    return rows, uneven


def transfer_rows(launches: list[dict], transfers: list[dict]) -> list[dict]:
    inside = [t for t in transfers if 0 < t["position"] < len(launches)]
    counts = collections.Counter(
        (t["transfer"], launches[t["position"] - 1]["op_code"], launches[t["position"]]["op_code"]) for t in inside)
    return [{"id": str(n), "transfer": kind, "after_op_code": after, "before_op_code": before, "count": str(c)}
            for n, ((kind, after, before), c) in enumerate(counts.items(), 1)]


def decide(run_dir: pathlib.Path) -> int:
    run_path, op_path, graph_path = run_dir / "run.json", run_dir / "op_table.csv", run_dir / "graph_ops.csv"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    capture, passes = run.get("graph_capture"), run.get("entry_test_passes") or {}
    if not capture or not graph_path.exists() or not op_path.exists():
        print("ERROR: --decide needs run.json graph_capture, graph_ops.csv and op_table.csv")
        return 2
    if not passes.get("result") or latest_entry_failure(run):
        print("ERROR: run.json records no local pass of the entry test that is newer than its last failure")
        return 2
    source, destinations = run_targets(run)
    launched = [{"op_type": "tt_dnn_device", "op_code": g["op_code"]}
                for g in read_csv(graph_path) for _ in range(int(g["launches"]))]
    ops = read_csv(op_path)
    for warning in decide_source_statuses(ops, launched, source, capture["inferences"], passes, ci_job(run),
                                          how="under a graph capture"):
        print(f"WARNING: {warning}")
    write_csv(op_path, op_table_columns(source, destinations), ops)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=pathlib.Path, nargs="?", help="graph capture JSON written by opa_graph_capture")
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--decide", action="store_true",
                        help="decide source rows from graph_ops.csv and the recorded pass; no capture")
    parser.add_argument("--inferences", type=int,
                        help="forward passes the test dispatched through ttnn (trace replays excluded)")
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="date of the captured run")
    parser.add_argument("--entry-pass", metavar="RESULT",
                        help="the entry test passed in this run; RESULT is its pass line (for example the PCC)")
    parser.add_argument("--entry-fail", metavar="RESULT",
                        help="the entry test failed in this run; RESULT is its failure line")
    add_build_argument(parser)
    args = parser.parse_args()
    if args.decide:
        return decide(args.run_dir)
    if args.capture is None or args.inferences is None:
        print("ERROR: give the capture file and --inferences, or --decide")
        return 2
    if args.entry_pass and args.entry_fail:
        print("ERROR: give --entry-pass or --entry-fail, not both")
        return 2
    if (args.entry_pass or args.entry_fail) and not args.build:
        print("ERROR: --entry-pass and --entry-fail need --build, the tt-metal build type the test ran on")
        return 2
    if args.inferences < 1:
        print("ERROR: --inferences must be at least 1")
        return 2
    with args.capture.open(encoding="utf-8") as fh:
        document = json.load(fh)
    launches, transfers = walk(_graph(document))
    dram = peak_dram(_graph(document))
    if not launches:
        print("ERROR: the capture has no device-op launch with a program factory; it was taken from a "
              "tt-metal without factory identity in graph capture, or the test launched nothing")
        return 2
    run_path = args.run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    commit = run.get("tt_metal_commit", "")
    if metadata.get("git_sha") and commit and metadata["git_sha"] != commit:
        print(f"ERROR: the capture was taken at {metadata['git_sha']}; the run is pinned to {commit}")
        return 2
    rows, uneven = ops_rows(launches, args.inferences)
    moves = transfer_rows(launches, transfers)
    write_csv(args.run_dir / "graph_ops.csv", GRAPH_OPS_COLUMNS, rows)
    write_csv(args.run_dir / "graph_host_transfers.csv", GRAPH_HOST_TRANSFERS_COLUMNS, moves)
    run["graph_capture"] = {
        "path": str(args.capture.resolve()), "date": args.date, "inferences": args.inferences,
        "launches": len(launches), "kernel_ops": len(rows),
        "host_transfers_inside": sum(int(m["count"]) for m in moves),
        "git_sha": metadata.get("git_sha", ""), "dirty": metadata.get("dirty"),
        "peak_dram_mib": round(max(dram.values()) / 2**20, 1) if dram else None,
        "peak_dram_mib_per_device": {d: round(b / 2**20, 1) for d, b in sorted(dram.items())},
    }
    source, destinations = run_targets(run)
    record_entry_result(run, source, args.date, "source:graph", args.entry_pass, args.entry_fail, args.build)
    op_path = args.run_dir / "op_table.csv"
    if args.entry_pass and op_path.exists():
        ops = read_csv(op_path)
        launched = [{"op_type": "tt_dnn_device", "op_code": launch["op_code"]} for launch in launches]
        for warning in decide_source_statuses(ops, launched, source, args.inferences, run["entry_test_passes"],
                                              ci_job(run), how="under a graph capture"):
            print(f"WARNING: {warning}")
        write_csv(op_path, op_table_columns(source, destinations), ops)
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    peak = run["graph_capture"]["peak_dram_mib"]
    print(f"{len(launches)} launches, {len(rows)} (op code, factory) pairs, "
          f"{run['graph_capture']['host_transfers_inside']} host transfers between the first and last launch, "
          f"peak DRAM {'not recorded' if peak is None else f'{peak} MiB'}")
    for line in uneven:
        print(f"WARNING: not a multiple of {args.inferences} inferences: {line}")
    if metadata.get("dirty"):
        print("WARNING: the capturing tt-metal checkout had local modifications; name them in summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
