"""Column names, targets and status values shared by every tt-model-op-analysis script."""

from __future__ import annotations

import csv
import pathlib
import re

# A run analyses one source implementation on one profile and, optionally, projects its kernel ops
# onto destinations. A destination is a profile running the same code, or `quasar-bf16`: the code
# with block-float dtypes turned into bfloat16 and mainline calls routed to Quasar forks.
PROFILES = ("p100", "p150", "n150", "n300-1chip", "quasar")
TARGET_LABELS = {
    "p100": "P100", "p150": "P150", "n150": "N150", "n300-1chip": "N300, one chip", "quasar": "Quasar",
    "quasar-bf16": "Quasar, BF16 projection",
}
TARGETS = tuple(TARGET_LABELS)
SOURCE_TARGETS = PROFILES
# Destinations whose launches are traced: profiles that run the same code with their own grid.
LAUNCH_PROFILES = ("p100", "p150", "n150", "n300-1chip")
TARGET_PROFILE = {t: t for t in PROFILES} | {"quasar-bf16": "quasar"}

STATUS_VALUES = ("✅", "⚠️", "❌", "n/a")
CONFIDENCE_VALUES = ("verified", "unverified")
# How the sheet shows `confidence`: whether the code alone fixes the factory a call selects.
CONFIDENCE_DISPLAY = {"verified": "fixed by the code", "unverified": "decided at run time"}
KERNEL_OPS_TAB = "Device ops by factory"
DEVICE_OPS_TAB = "Device op support"
# Whether a destination runs a device op at all, for any arguments (target-status.md, "Device op
# support"), as opposed to the row statuses, which judge the model's own calls.
OP_SUPPORT_VALUES = ("✅", "⚠️", "❌", "n/a")
OP_SUPPORT_LABELS = {"✅": "supported", "⚠️": "untested", "❌": "not supported", "n/a": "not executed"}
NOT_CHECKED = "⏳"
# device_ops.csv row-count field per row status.
STATUS_COUNT_FIELDS = {"✅": "rows_ok", "⚠️": "rows_untested", "❌": "rows_failing", "n/a": "rows_na"}
ALWAYS = "always"
ORIGIN_VALUES = ("static", "measured", "static+measured")
NOT_TRACED = "not traced"
# Targets whose status is decided per kernel op from Quasar tests (references/target-status.md).
TEST_TARGETS = ("quasar", "quasar-bf16")
TEST_RESULT_VALUES = ("pass", "fail", "xfail", "error", "timeout", "skip", "not run")
# A failing result gives ❌ and a pass ✅; an undecided one leaves the status as the code decided it.
# Each class is ordered most telling first: an error is fixable, a timeout is not.
FAILING_RESULTS = ("fail", "xfail")
UNDECIDED_RESULTS = ("error", "timeout")
RESULT_STATUS = {"pass": "✅", "fail": "❌", "xfail": "❌"}
# Plugin 0.5.0 recorded a runner timeout as a hang.
LEGACY_RESULTS = {"hang": "timeout"}

OP_TABLE_BASE = (
    "id", "stage", "origin", "ttnn_api", "op_code", "device_op", "program_factory",
    "call_site", "launch_condition", "confidence", "dispatch_evidence", "grid_dependency",
)
CALL_TRACE_COLUMNS = ("id", "profile", "stage", "ops", "repeats", "launches_per_repeat", "notes")
PORT_BLOCKERS_COLUMNS = ("id", "blocker", "affected_ids", "evidence", "required_change")
HOST_OPS_COLUMNS = ("id", "host_work", "when", "counts_as_host_op", "reason")
HOST_OP_VALUES = ("yes", "no")
MATCH_KEY = ("stage", "ttnn_api", "device_op", "program_factory")

# A file:line reference, and the grid_dependency values that are not a hardcoded grid.
CALL_SITE = re.compile(r"[\w./-]+\.(?:py|cpp|hpp|cc|h):\d+")
GRID_KEYWORDS = ("automatic", "none")


# tt-metal build types (build_metal.sh --build-type). CMakeLists.txt defines DEBUG, and so compiles
# TT_ASSERT in, only for Debug and RelWithDebInfo; a pass on Release can hide a failing assertion.
BUILD_TYPES = ("Release", "RelWithDebInfo", "Debug", "ASan", "TSan", "ASanCoverage")
BUILD_ASSERTS = {"Release": "TT_ASSERT off", "RelWithDebInfo": "TT_ASSERT on", "Debug": "TT_ASSERT on"}


def build_label(build: str) -> str:
    if not build:
        return "build not recorded"
    return f"{build} build, {BUILD_ASSERTS[build]}" if build in BUILD_ASSERTS else f"{build} build"


def latest_entry_failure(run: dict) -> dict | None:
    """The recorded local failure of the entry test when it is not older than the latest local pass."""
    failure, passes = run.get("entry_test_failure"), run.get("entry_test_passes") or {}
    if failure and (not passes.get("date") or failure.get("date", "") >= passes["date"]):
        return failure
    return None


# Whether the model runs on a smaller core grid and with fewer layers (static-phase.md, step 6).
CONFIG_KEYS = ("configurable_grid", "configurable_layers")
CONFIG_VALUES = ("yes", "no", "partial", "n/a")
# Whether a CI job runs the entry test on the source profile, recorded at setup (run-setup.md).
# n/a: a Quasar source, whose entry test is never run.
CI_KEY = "entry_test_ci"
CI_VALUES = ("yes", "no", "n/a")


def ci_job(run: dict) -> str:
    """The CI job that runs the entry test on the source profile, or "" when none does."""
    entry = run.get(CI_KEY)
    if isinstance(entry, dict) and entry.get("value") in CI_VALUES:
        return str(entry.get("evidence", "")) if entry["value"] == "yes" else ""
    # Runs set up before entry_test_ci existed name the job in their pass record.
    return (run.get("entry_test_passes") or {}).get("ci", "")

# The four ways a run learns what the model launches, least to most exact. A graph capture names
# every launch's device op and program factory with the call's real arguments; a kernel cache holds
# the compiled kernels themselves. Tracy sees op codes, not factories; the static trace sees code.
SOURCE_METHODS = ("static", "tracy", "graph", "cache")
METHOD_LABELS = {
    "static": "Static trace of the model code",
    "tracy": "Tracy profile of the entry test",
    "graph": "ttnn graph capture of the entry test",
    "cache": "Kernel cache of the entry test (elfray)",
}
GRAPH_OPS_COLUMNS = ("id", "op_code", "device_op", "program_factory", "program_factory_type", "ttnn_apis",
                     "launches", "launches_per_inference", "cache_misses")
GRAPH_HOST_TRANSFERS_COLUMNS = ("id", "transfer", "after_op_code", "before_op_code", "count")
# Same columns as the elfray Cache API Analyzer's compute and data-movement tables, so a sheet and
# the extension's view of one cache can be compared cell by cell. "TTNN Ops" is elfray's name for
# the enclosing compute-API call (for example pack_tile), not a ttnn operation.
CACHE_COMPUTE_COLUMNS = ("id", "API", "Kernels", "TTNN Ops", "Op Args", "Input Data Formats",
                         "Output Data Formats", "Tile Dims", "Math Fidelity", "Math Approx", "FP32 Dest Accum",
                         "Dst Sync Mode")
CACHE_DATAFLOW_COLUMNS = ("id", "API", "Kernels", "Layer", "Processor(s)", "Op Args", "Enclosing Dataflow API")
CACHE_KERNELS_COLUMNS = ("id", "kernel", "kind", "compile_hash", "api_calls", "errors")
LLK_GAP_VERDICTS = ("gap", "renamed", "not a gap")
LLK_GAPS_COLUMNS = ("id", "kind", "name", "quasar_status", "review", "review_note", "template_args",
                    "missing_template_args",
                    "quasar_template_args", "missing_runtime_args", "similar_quasar_apis", "quasar_definitions",
                    "configs", "cache_rows", "kernels")


# Why a source measurement item failed. Only `test` says something about the entry test itself; `tool`
# (a profiler timeout from instrumented compilation, an unreadable capture) and `not-run` do not.
FAILURE_CAUSES = ("test", "tool", "not-run")

# The source measurement item of each measured method and the run.json key its script records.
METHOD_ITEMS = {"tracy": ("source:tracy", "measured_ops"), "graph": ("source:graph", "graph_capture"),
                "cache": ("source:cache", "kernel_cache")}


def methods_used(run: dict) -> list[str]:
    """Methods with results in this run, in SOURCE_METHODS order."""
    done = {"static": run.get("kind") == "static"} | {m: bool(run.get(key)) for m, (_, key) in METHOD_ITEMS.items()}
    return [m for m in SOURCE_METHODS if done[m]]


def confidence(methods: list[str]) -> str:
    """Run-level confidence in the kernel-op list (device op and program factory per launch).

    Only a graph capture names the factory each launch used. Tracy confirms op codes and the kernel
    cache confirms compiled kernels, neither per factory; the static trace is code read, not run.
    """
    if "graph" in methods:
        return "high"
    if "tracy" in methods or "cache" in methods:
        return "medium"
    return "low"

# kernel_tests.py prefixes a row's evidence with its TTsim result; the written evidence follows the
# separator. Only that prefix is split off, so a quoted C++ `||` in the evidence stays intact.
TTSIM_PREFIX = "TTsim run: "
EVIDENCE_SEPARATOR = " || "
# A row number or stage at the start of evidence repeats what the table already shows.
ROW_PREFIX = re.compile(r"^Row \d+\b(?:\s*\([^)]*\))?[\s:.,-]*")


def split_evidence(text: str) -> tuple[str, str]:
    """(TTsim result or '', written evidence) of an evidence cell."""
    text = text.strip()
    if text.startswith(TTSIM_PREFIX) and EVIDENCE_SEPARATOR in text:
        measured, written = text.split(EVIDENCE_SEPARATOR, 1)
        return measured, written.strip()
    return "", text


def run_targets(run: dict) -> tuple[str, list[str]]:
    return run["source"], list(run.get("destinations", []))


def has_launches(target: str, is_source: bool) -> bool:
    return is_source or target in LAUNCH_PROFILES


def target_columns(target: str, is_source: bool) -> list[str]:
    columns = [f"{target}:shapes", f"{target}:launches"] if has_launches(target, is_source) else []
    if target in TEST_TARGETS:
        columns += [f"{target}:tests", f"{target}:test_result"]
    return columns + [f"{target}:status", f"{target}:evidence"]


def op_table_columns(source: str, destinations: list[str]) -> list[str]:
    columns = list(OP_TABLE_BASE) + target_columns(source, True)
    for destination in destinations:
        columns += target_columns(destination, False)
    return columns


def kernel_ops_columns(source: str, destinations: list[str]) -> list[str]:
    columns = ["id", "device_op", "program_factory", "op_code", "origin", "sources", "op_table_ids",
               "static_launches", "static_launches_op_code", "measured_launches_op_code", "graph_launches"]
    for target in [source, *destinations]:
        columns.append(f"{target}:statuses")
        if target in TEST_TARGETS:
            columns += [f"{target}:tests", f"{target}:test_results"]
    return columns


def op_key(row: dict) -> str:
    """Device op of an op-table row; a measured-only row without one is keyed by its op code."""
    return row["device_op"] or row["op_code"]


def device_op_support_columns(destinations: list[str]) -> list[str]:
    columns = ["device_op"]
    for target in destinations:
        columns += [f"{target}:op_support", f"{target}:op_evidence"]
    return columns


def device_ops_columns(source: str, destinations: list[str]) -> list[str]:
    columns = ["id", "device_op", "program_factories", "ttnn_apis", "op_table_ids", f"{source}:launches"]
    for target in destinations:
        columns += [f"{target}:{f}" for f in STATUS_COUNT_FIELDS.values()]
        columns += [f"{target}:op_support", f"{target}:op_evidence"]
    return columns


# Spreadsheet presentation: tab order, readable headers and column widths in pixels.
# Tabs from the static trace have plain names; tabs from a measurement start with the method's name.
OP_TABLE_TAB = "ttnn calls"
CALL_TRACE_TAB = "Launch order"
TTSIM_TESTS_TAB = "TTsim: tests"
LLK_GAPS_TAB = "LLK gaps (Quasar)"
SHEET_TABS = (
    ("op_table", OP_TABLE_TAB), ("device_ops", DEVICE_OPS_TAB), ("kernel_ops", KERNEL_OPS_TAB),
    ("kernel_test_results", TTSIM_TESTS_TAB),
    ("call_trace", CALL_TRACE_TAB), ("port_blockers", "Port blockers"),
    ("host_ops", "Host work in code"), ("measured_ops", "Tracy: ops"), ("host_fallback", "Tracy: host fallback"),
    ("footprint", "Tracy: resource counts"), ("diff", "Static vs Tracy"), ("graph_ops", "Graph: launches"),
    ("graph_host_transfers", "Graph: host transfers"), ("cache_compute", "Cache: compute APIs"),
    ("cache_dataflow", "Cache: dataflow APIs"), ("cache_kernels", "Cache: kernels"),
    ("llk_gaps_quasar", LLK_GAPS_TAB),
)
# The method whose run writes each tab's CSV, for saying why a tab is missing.
TAB_METHOD = {"op_table": "static", "device_ops": "static", "kernel_ops": "static", "call_trace": "static", "port_blockers": "static",
              "host_ops": "static", "kernel_test_results": "TTsim", "measured_ops": "tracy", "host_fallback": "tracy",
              "footprint": "tracy", "diff": "tracy", "graph_ops": "graph", "graph_host_transfers": "graph",
              "cache_compute": "cache", "cache_dataflow": "cache", "cache_kernels": "cache", "llk_gaps_quasar": "cache"}
FIELD_LABELS = {
    "id": "#", "stage": "Stage", "origin": "Origin (static / measured)", "ttnn_api": "ttnn API", "op_code": "Op code (Tracy)",
    "device_op": "Device op", "program_factory": "Program factory", "call_site": "Call site (file:line)",
    "launch_condition": "Launch condition", "confidence": "Factory choice",
    "dispatch_evidence": "Dispatch evidence", "grid_dependency": "Grid dependency",
    "launches_per_repeat": "Launches per repeat", "affected_ids": "Affected ttnn call rows",
    "required_change": "Required change", "op_table_ids": "ttnn call rows", "program_factories": "Program factories",
    "static_launches": "Static launches",
    "static_launches_op_code": "Static launches (all factories of this op code)",
    "measured_launches_op_code": "Measured launches (op code)", "host_work": "Host work", "counts_as_host_op": "Counts as host op",
    "sources": "Seen by", "graph_launches": "Graph launches per inference",
    "program_factory_type": "Program factory (full type)", "ttnn_apis": "ttnn API",
    "launches_per_inference": "Launches per inference", "cache_misses": "Program cache misses",
    "after_op_code": "Previous launch (op code)", "before_op_code": "Next launch (op code)", "compile_hash": "Compile hash",
    "api_calls": "API calls", "quasar_status": "Quasar (by name)", "review": "Review",
    "review_note": "Review note", "template_args": "Template args (cache)",
    "missing_template_args": "Not taken on Quasar", "quasar_template_args": "Quasar template args",
    "missing_runtime_args": "Runtime args not taken on Quasar", "similar_quasar_apis": "Similar Quasar APIs",
    "quasar_definitions": "Quasar definitions", "cache_rows": "Cache rows",
}
TARGET_FIELD_LABELS = {
    "shapes": "shapes and dtypes", "launches": "launches per inference",
    "status": "status", "evidence": "evidence and gap", "tests": "kernel-op tests", "test_result": "TTsim result",
    "statuses": "statuses (rows)", "test_results": "TTsim results",
    "rows_ok": "✅ rows", "rows_untested": "⚠️ rows", "rows_failing": "❌ rows", "rows_na": "n/a rows",
    "op_support": "device op supported", "op_evidence": "device op evidence",
}
FIELD_WIDTHS = {
    "id": 40, "stage": 120, "origin": 110, "tests": 300, "test_result": 90, "ttnn_api": 180, "op_code": 200, "device_op": 200, "program_factory": 260,
    "call_site": 220, "launch_condition": 180, "confidence": 100, "dispatch_evidence": 320,
    "grid_dependency": 220, "shapes": 260, "launches": 90, "status": 80, "evidence": 360,
    "ops": 220, "notes": 300, "blocker": 260, "required_change": 300, "host_work": 260, "reason": 320,
    "profile": 70, "repeats": 70, "launches_per_repeat": 90,
    "program_factory_type": 360, "ttnn_apis": 220, "API": 420, "Kernels": 260, "Op Args": 300,
    "TTNN Ops": 200, "errors": 300,
}
DEFAULT_WIDTH = 160


def display_header(column: str) -> str:
    if ":" in column:
        target, field = column.split(":", 1)
        return f"{TARGET_LABELS.get(target, target)}: {TARGET_FIELD_LABELS.get(field, field.replace('_', ' '))}"
    if column != column.lower():
        # Already a display name (the elfray cache columns).
        return column
    return FIELD_LABELS.get(column, column.replace("_", " ").capitalize())


def column_width(column: str) -> int:
    return FIELD_WIDTHS.get(column.split(":", 1)[-1], DEFAULT_WIDTH)


# utf-8-sig: reading tolerates a BOM, writing adds one so Excel and Numbers decode the
# status emoji as UTF-8 instead of Mac Roman.
CSV_ENCODING = "utf-8-sig"


def read_header(path: pathlib.Path) -> list[str]:
    with pathlib.Path(path).open(newline="", encoding=CSV_ENCODING) as fh:
        return next(csv.reader(fh), [])


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with pathlib.Path(path).open(newline="", encoding=CSV_ENCODING) as fh:
        return list(csv.DictReader(fh))


def write_csv(path: pathlib.Path, columns: list[str], rows: list[dict]) -> None:
    with pathlib.Path(path).open("w", newline="", encoding=CSV_ENCODING) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: pathlib.Path) -> list[dict]:
    """kernel_test_results_<target>.csv with legacy result values renamed."""
    rows = read_csv(path)
    for row in rows:
        row["result"] = LEGACY_RESULTS.get(row["result"], row["result"])
    return rows
