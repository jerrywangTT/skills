"""Reader-facing descriptions for the spreadsheet: one per tab (listed on Info) and one per column
(a note on its header cell).

A note says what a column holds and where its value comes from, in the words of
references/schemas.md, so a reader of the sheet needs no other document.
"""

from __future__ import annotations

STATIC, TRACY, GRAPH, CACHE, TTSIM = ("static trace", "Tracy", "graph capture", "kernel cache", "TTsim")

# Tab title (or prefix for tabs with a target suffix) -> (what it holds, the methods that fill it).
TAB_DESCRIPTIONS = {
    "Info": ("Run identity, bring-up checks, analysis methods and confidence, counts and the summary.", "all"),
    "Device op support": ("One row per device op: per destination, how many of the model's calls are supported, "
                          "untested or failing, and whether the destination runs the op at all, with the reason.",
                          f"{STATIC} and a search of the destination's tests; {TTSIM} results update it"),
    "Device ops by factory": ("One row per device op and the program factory it selected (the kernels that run): "
                              "launches, the methods that saw it, and its status and evidence per target.", f"{STATIC}; {TRACY} and {GRAPH} confirm launches"),
    "ttnn calls": ("One row per launching variant of a ttnn call: call site, dispatch, grid dependency, shapes and "
                 "status per target.", f"{STATIC}; {TRACY} adds measured-only rows"),
    "TTsim: tests": ("Quasar kernel-op tests run on the simulator, one row per test with its result.", TTSIM),
    "Launch order": ("Launch order per model block, with repeats; its launch total must equal the ttnn calls tab's.", STATIC),
    "Port blockers": ("What stops the code from running on a destination, written once and linked to the rows it "
                      "affects.", STATIC),
    "Host work in code": ("Work the model does on the host, and whether it counts as host time inside the forward.",
                 STATIC),
    "Tracy: ops": ("Every op the profiled run executed, in order, with device and host time. Attributes longer "
                   "than 300 characters are cut; measured_ops.csv in the run directory has them whole.", TRACY),
    "Tracy: host fallback": ("Device and host op counts and times in the profiled window.", TRACY),
    "Tracy: resource counts": ("Launches, the largest core count and peak DRAM per op code.", TRACY),
    "Static vs Tracy": ("Where the static trace and the Tracy run disagree, per op code.", f"{STATIC} and {TRACY}"),
    "Graph: launches": ("One row per op code and program factory the captured run launched, with launches per "
                      "inference.", GRAPH),
    "Graph: host transfers": ("Host transfers between the first and the last launch: next input upload, previous "
                             "readback, or host work inside the forward.", GRAPH),
    "Cache: compute APIs": ("Every distinct LLK call in the compute kernels the run compiled, with template "
                            "arguments, data formats and math config (the elfray Cache API Analyzer table).", CACHE),
    "Cache: dataflow APIs": ("Every distinct data-movement call in the run's kernels (elfray table).", CACHE),
    "Cache: kernels": ("One row per compiled kernel build, with errors if elfray could not read it.", CACHE),
    "LLK gaps (Quasar)": ("Each LLK API and data format of the cache compared with the Quasar LLK sources, "
                          "with the reviewed verdict per candidate.", f"{CACHE} and the Quasar sources"),
    "Changes": ("What changed in a table between two runs of the same model.", "comparison of two runs"),
}

STATUS_NOTES = {"✅": "Supported: a test or recorded run shows it.",
                "⚠️": "The code path exists but nothing tests it.",
                "❌": "Shown not to work.",
                "n/a": "Never launched on this target."}
STATUS_NOTE = " ".join(f"{k}{':' if k == 'n/a' else ''} {v[0].lower()}{v[1:]}" for k, v in STATUS_NOTES.items())

# Notes by column field, shared by every tab unless a tab overrides it below.
FIELD_NOTES = {
    "id": "Row number. Other columns and tabs refer to rows by it.",
    "stage": "Model block, module and role of the call, from the model's structure.",
    "origin": "static: found by the static trace. measured: seen only in the Tracy run. static+measured: both. "
              "graph: launched in the graph capture but missing from the static trace.",
    "ttnn_api": "The ttnn call in the model code that launches this.",
    "op_code": "Device-operation class name, as Tracy and the graph capture report it.",
    "device_op": "The ttnn::prim device operation.",
    "program_factory": "The C++ program factory the device op selects for this call: the kernels that run.",
    "call_site": "Where the model makes the call, as file:line at the pinned tt-metal commit.",
    "launch_condition": "always, or the runtime condition under which the launch happens.",
    "confidence": "Whether the code alone fixes which program factory this call selects. fixed by the code: the "
                  "static trace followed the dispatch to this factory. decided at run time: the choice depends on "
                  "something known only when it runs (free L1, buffer addresses); a graph capture shows the factory "
                  "that was used.",
    "dispatch_evidence": "The C++ file and function that select the device op and factory, and any runtime "
                         "alternative.",
    "grid_dependency": "Hardcoded grid, shard spec or program config this op depends on, with file:line. "
                       "automatic: derived from the device grid. none: no core grid. Hardcoded values block a "
                       "smaller grid.",
    "shapes": "Input and output shapes, dtype, layout and sharding on this target.",
    "launches": "Launches per inference on this target.",
    "status": STATUS_NOTE,
    "evidence": "The decisive fact behind this target's status, with file:line.",
    "tests": "Quasar kernel-op tests (pytest node ids) that decide the status. For each row the static phase picks "
             "the tests that run the same kernel op: device op, program factory, layout and sharding, dtype. Batch, "
             "core grid and shard sizes are not part of the match; the evidence names where a test differs from "
             "the row. A TTsim run of these tests outweighs the static reading: a fail or xfail makes the status "
             "❌, ✅ needs every test to pass, and timeout, error or skip decide nothing.",
    "test_result": "Simulator result of those tests: pass, fail, xfail, timeout, error, skip, or not run.",
    "statuses": "Number of rows in the ttnn calls tab with each status. " + STATUS_NOTE,
    "test_results": "Simulator result of each kernel-op test.",
    "program_factories": "Program factories the model's calls of this device op select.",
    "rows_ok": "Rows of the ttnn calls tab for this device op with ✅ on this destination: a test or recorded run "
               "on the destination covers that exact call.",
    "rows_untested": ("Model calls of this device op that no test on this destination covers. Yellow: the op runs "
                      "here, so these calls may work as they are. Orange: the op has code here but no test, so they "
                      "may work once it is tested. Red: the op does not run here."),
    "rows_failing": ("Model calls of this device op that fail on this destination (failing check, xfail, hang). "
                     "Orange: the op runs here with other arguments, so these calls must move to a supported variant. "
                     "Red: the op itself needs work here first."),
    "rows_na": "Rows with n/a on this destination: the destination never launches that call.",
    "op_support": "Whether the destination runs this device op at all, for any arguments, not only the model's. "
                  "✅ supported: a test on the destination runs the op (any parameters). ⚠️ untested: the op has "
                  "code for the destination but no test of any variant. ❌ not supported: a check that fails "
                  "whatever the arguments (no factory or fork, every factory stops on this architecture). n/a: the "
                  "destination never executes it. ⏳ not checked. After the icon: how this was decided. The icon sets the colour "
                  "of the ⚠️ and ❌ row counts.",
    "sources": "Methods that saw this device op and factory. tracy (op code): Tracy saw the op code, which several "
               "rows here share, so it does not confirm this factory.",
    "op_table_ids": "The rows of the ttnn calls tab this device op and factory cover.",
    "static_launches": "Launches per inference of these rows, from the static trace.",
    "static_launches_op_code": "Static launches of every factory of this op code.",
    "measured_launches_op_code": "Launches per inference of this op code in the Tracy run (Tracy does not name "
                                 "factories).",
    "graph_launches": "Launches per inference of this device op and factory in the graph capture.",
    "profile": "Target the launches are counted for.",
    "ops": "Device ops of the block, in launch order.",
    "repeats": "How many times the block runs per inference.",
    "launches_per_repeat": "Launches per run of the block.",
    "notes": "What the trace decided here and why.",
    "blocker": "What stops the code from running on the destination.",
    "affected_ids": "Rows of the ttnn calls tab the blocker affects.",
    "required_change": "What the port has to change.",
    "host_work": "Work done on the host.",
    "when": "When and how often it runs.",
    "counts_as_host_op": "⚠️ yes: runs on the host once per inference inside the forward, so it adds to host time. "
                         "✅ no: outside the forward (test setup, weight preparation, reference computation).",
    "reason": "Why it counts or not.",
    "op_type": "tt_dnn_device for a device op; tt_dnn_cpu or python_fallback for host work.",
    "attributes": "Op attributes Tracy recorded (program config and others).",
    "inputs": "Input tensors: shape, layout, dtype, memory.",
    "outputs": "Output tensors: shape, layout, dtype, memory.",
    "core_count": "Cores the op ran on.",
    "device_kernel_ns": "Device kernel duration, ns.",
    "host_ns": "Host duration, ns.",
    "scope": "What the row summarizes.",
    "window": "warm: between signposts, compiled kernels; cold: whole test, compilation included.",
    "device_ops": "Device ops in the window.",
    "host_ops": "Host ops in the window.",
    "device_op_time_ms": "Total device kernel time, ms.",
    "host_op_time_ms": "Total host op time, ms.",
    "total_time_ms": "Wall time of the window, ms.",
    "host_pct_of_total": "Host op time as a share of the window's wall time.",
    "max_core_count": "Largest core count any launch of this op code used.",
    "peak_dram_mb": "Peak DRAM, only when measured.",
    "category": "Kind of disagreement.",
    "measured_launches": "Launches the Tracy run shows.",
    "static_ids": "Rows of the ttnn calls tab on the static side.",
    "detail": "What differs.",
    "program_factory_type": "Full C++ type of the factory the graph capture recorded.",
    "ttnn_apis": "Top-level ttnn calls that launched it.",
    "launches_per_inference": "Launches divided by the forwards the test dispatched.",
    "cache_misses": "Launches that built a new program (the first of each configuration).",
    "transfer": "to_torch, from_torch, from_device, to_device, copy_host_to_device or copy_device_to_host.",
    "after_op_code": "The device-op launch that ran just before the transfer.",
    "before_op_code": "The device-op launch that ran just after the transfer.",
    "count": "How many times this transfer happened between these launches.",
    "node_id": "Pytest node id of the kernel-op test.",
    "result": "Simulator result: pass, fail, xfail, timeout, error or skip.",
    "duration_s": "Run time, s.",
    "log": "Path of the pytest log.",
    "environment": "The environment the test ran under.",
    "change": "added, removed or changed.",
    "key": "Row identity the runs are matched on.",
    "column": "Changed column.",
    "old": "Value in the earlier run.",
    "new": "Value in the later run.",
    "API": "API name with the template arguments it was instantiated with.",
    "Kernels": "Kernels that make this call.",
    "TTNN Ops": "The enclosing compute-API call (for example pack_tile); elfray's column name, not a ttnn op.",
    "Op Args": "Runtime arguments: a constant value, or =? when the value is only known at run time.",
    "Input Data Formats": "Data format per input circular buffer (cbN=format).",
    "Output Data Formats": "Data format per output circular buffer.",
    "Tile Dims": "Tile shape per circular buffer.",
    "Math Fidelity": "Math fidelity the kernel was compiled with.",
    "Math Approx": "Approximate math mode.",
    "FP32 Dest Accum": "Whether the destination register accumulates in FP32.",
    "Dst Sync Mode": "Destination register sync mode.",
    "Layer": "dataflow_api (user-facing) or noc_api (lower-level NOC helpers).",
    "Processor(s)": "Data-movement processors that make the call (brisc, ncrisc).",
    "Enclosing Dataflow API": "The dataflow API call this one is inlined into.",
    "kernel": "Kernel name (its source file).",
    "compile_hash": "Compile hash: one per distinct compile-time configuration of the kernel.",
    "api_calls": "API calls read from this build in the selected layers.",
    "errors": "Why elfray could not read part of the build.",
    "name": "LLK API or data format name.",
    "quasar_status": "Found by name in the Quasar sources: present, signature differs, missing; not an LLK for "
                     "other calls.",
    "review": "Verdict after reading the Quasar headers: ❌ gap, ✅ renamed, ✅ not a gap, ⏳ not reviewed yet. "
              "Empty when the API was found by name.",
    "review_note": "The Quasar source that decides the verdict.",
    "template_args": "Template arguments the kernels instantiated.",
    "missing_template_args": "Those the closest Quasar definition does not take.",
    "quasar_template_args": "Template arguments of the closest Quasar definition.",
    "missing_runtime_args": "Runtime arguments the closest Quasar definition does not take.",
    "similar_quasar_apis": "Quasar LLK names sharing the longest prefix, for an API not found by name.",
    "quasar_definitions": "file:line of each Quasar definition.",
    "configs": "Distinct template configurations the cache holds.",
    "cache_rows": "Cache table rows that use it.",
    "kernels": "Kernels that use it.",
}

# Fields whose meaning differs on one tab.
TAB_FIELD_NOTES = {
    "Graph: launches": {"launches": "Launches in the whole capture."},
    "Device op support": {"launches": "Launches per inference on the source over all rows of this device op, "
                                      "measured-only rows included, as in the Info counts.",
                          "op_table_ids": "The rows of the ttnn calls tab of this device op, all factories."},
    "Tracy: resource counts": {"launches": "Launches of this op code in the Tracy window."},
    "Cache: kernels": {"kind": "compute or data_movement."},
    "LLK gaps (Quasar)": {"kind": "llk_api or data_format."},
}


# Analysis method (opa_schema.SOURCE_METHODS) -> a note on its Info label: what it does, what it is for,
# what it sees and misses, and the confidence it gives.
METHOD_NOTES = {
    "static": (
        "What it does: reads the model code at the pinned tt-metal commit and follows every ttnn call through "
        "its C++ dispatch to the device op and the program factory chosen for the call's shapes, data types, "
        "layouts and memory configs. Nothing runs.\n"
        "What it is for: the only method that says why each op is launched and where the code sets a grid, "
        "shard spec or program config. Every status per target, the grid blockers and the configurability "
        "checks rest on it, so it is always done.\n"
        "Sees: every ttnn call site, its arguments as written, dispatch conditions, hardcoded grids.\n"
        "Misses: values known only at run time (a factory chosen from input shapes is marked 'decided at run "
        "time'), how often an op runs when that depends on run-time values, host work outside the code it "
        "follows.\n"
        "Confidence: low on its own."),
    "tracy": (
        "What it does: runs the entry test once on the source device under the Tracy profiler, calling the "
        "forward live; a replayed device trace would hide host work.\n"
        "What it is for: confirms which ops really launch and how often per inference, measures device time "
        "per op, and counts ops that ran on the host instead of the device (host fallback).\n"
        "Sees: op code per launch, launch counts, device and host op time, op attributes and tensor shapes.\n"
        "Misses: the program factory (it names only the op code), the kernels compiled, memory use.\n"
        "Confidence: medium; program factories still come from the static trace."),
    "graph": (
        "What it does: runs the entry test once on the source device with ttnn graph capture, which records "
        "every device op call, the program factory it used, each buffer allocation and free, and each "
        "host-device transfer.\n"
        "What it is for: the exact list of kernel ops (device op and program factory per launch), peak DRAM "
        "allocated at once, and transfers inside the forward.\n"
        "Sees: device op, program factory, launch counts, program cache misses, DRAM buffer sizes, copies to "
        "and from the device.\n"
        "Misses: time, kernel source and the LLK calls inside kernels.\n"
        "Confidence: high."),
    "cache": (
        "What it does: runs the entry test once on the source device with an empty kernel cache "
        "(TT_METAL_CACHE) and debug info (TT_METAL_RISCV_DEBUG_INFO=1), then reads every compiled kernel ELF "
        "with elfray.\n"
        "What it is for: which compute (LLK) and dataflow APIs each kernel calls, with template arguments, "
        "data formats and math config; the Quasar LLK gap check compares these with the Quasar LLK.\n"
        "Sees: every kernel compiled during the test, the APIs inlined into it, data formats, tile "
        "dimensions, math fidelity.\n"
        "Misses: how often each kernel launches, which program factory built it, time.\n"
        "Confidence: medium; it confirms compiled kernels, not the factory of each launch."),
}

CI_NOTE = ("Whether a CI workflow or pipeline list runs the entry test on the source profile at the pinned commit. "
           "✅ yes: the job is named. ⚠️ no: nothing in CI guards this test on this profile, so a pass rests on "
           "local runs only, and a later commit can break it unnoticed. n/a: a Quasar source, whose entry test "
           "is read, never run.")
ISSUES_NOTE = ("Open tt-metal issues that report this entry test failing (any build, any profile), from a search at "
               "setup. While one is open the end-to-end test shows ⚠️ even after a local pass: a Release build "
               "compiles TT_ASSERT out and can pass where an assertion-enabled build fails.")
AGENT_NOTE = ("The coding agents that produced this analysis (for example Claude Code or Codex), with the model, "
              "the work each did and when.")
E2E_NOTE = ("The latest local run of the entry test and the tt-metal build it ran on. Release compiles TT_ASSERT out, "
            "so a pass there can hide an assertion that Debug and RelWithDebInfo builds hit.")
CONFIDENCE_NOTE = ("Confidence in the kernel-op list (device op and program factory per launch). high: a graph "
                   "capture ran. medium: Tracy or the kernel cache ran, no graph capture. low: only the static "
                   "trace.")


def tab_description(title: str) -> tuple[str, str] | None:
    for prefix, description in TAB_DESCRIPTIONS.items():
        if title == prefix or title.startswith(prefix + " (") or (prefix == "Changes" and title.startswith("Changes: ")):
            return description
    return None


def column_note(tab: str, column: str) -> str:
    """Note for a CSV column (a field, or `<target>:<field>`) on the given tab."""
    field = column.split(":", 1)[-1]
    overrides = TAB_FIELD_NOTES.get(tab) or TAB_FIELD_NOTES.get(tab.split(" (")[0], {})
    return overrides.get(field) or FIELD_NOTES.get(field, "")
