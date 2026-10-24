# Measured source phase

Adds what the source device actually launches and compiles, using the run's entry test on the run's
commit. For a Blackhole or Wormhole source there are three items, from at most two runs of the entry
test:

| Item | What it records | Files |
|---|---|---|
| `source:tracy` | op codes launched, with device and host times (Tracy) | `measured_ops.csv`, `host_fallback.csv`, `footprint.csv` |
| `source:graph` | every launch's device op and the program factory it selected, with the ttnn call that launched it (ttnn graph capture) | `graph_ops.csv`, `graph_host_transfers.csv` |
| `source:cache` | every kernel the run compiled and the LLK and dataflow API calls inside it, with template arguments, data formats and math config (kernel cache, read with elfray) | `cache_compute.csv`, `cache_dataflow.csv`, `cache_kernels.csv` |

The graph capture and the kernel cache are the most exact: they record what the device did, while
Tracy names op codes only and the static trace reads code. The graph capture and the kernel cache
share one run. Tracy gets its own: graph capture adds host work to every call, which Tracy would
count as host time, and Tracy's device profiler adds profiling code to every kernel, which the cache
would then hold. Recommend all three; the user may pick fewer.

A Quasar source has no source item: its entry test never runs, on a device or on TTsim. A
whole-model run on TTsim can spend most of an hour in one program with no way to tell slow from
hung. Its rows' kernel-op tests run on TTsim one node at a time (measured-destination.md).

The preflight run (run-setup.md, "Entry test preflight") is this phase's capture run, taken before
the static phase. When `run.json` has `graph_capture` or `kernel_cache` from it, those items are
done: do not run the capture again. The static phase's gate already let the pass decide the source
rows (`graph_report.py <run dir> --decide`).

## Kernel cache: the elfray engine

`cache_report.py` reads kernel ELFs through the elfray engine (`elfray --inlined-calls`,
`elfray --enum`) and builds the same compute and dataflow tables as the elfray VS Code extension's
Cache API Analyzer. It finds the engine in `--elfray`, `$ELFRAY`, `$ELFRAY_BIN`, `PATH`,
`~/.local/bin/elfray` or a VS Code extension install (`~/.vscode-server/extensions/elfray.elfray-*`);
exit code 3 means none was found. elfray is a private repository built for Linux x86_64 only.
Install it on the device host, with the user's approval, through
`gh api repos/tenstorrent/tt-elfray/contents/install.sh -H 'Accept: application/vnd.github.raw' | bash`
(needs `gh auth` with access to `tenstorrent/tt-elfray`). Without the engine there is no cache
analysis: the skill does not read DWARF itself. `source_methods.py` then shows the cache as
"available, read later": build it in the device run anyway (it costs two environment variables),
leave `source:cache` pending, and run `cache_report.py` once the engine is installed, on this or
another host, or open the cache in the extension ("Cache API Analyzer: Analyze a kernel cache").

## Steps

1. Read the build type of the checkout the runs will use: `CMAKE_BUILD_TYPE` in
   `<checkout>/build/CMakeCache.txt` (`build` links to `build_<type>`). Every `--entry-pass` and
   `--entry-fail` below passes it as `--build <type>`. tt-metal compiles `TT_ASSERT` only in Debug
   and RelWithDebInfo builds (`-DDEBUG`, CMakeLists.txt), so a Release pass can hide an assertion
   an issue reports; the sheet shows the build next to the result.
   Then run `python3 <plugin-root>/scripts/same_commit.py <run dir> --tt-metal <checkout>`. On failure,
   offer: check out the recorded commit and rebuild, or start a new static run. Never measure on
   another commit.
2. Probe the device: `python3 <plugin-root>/scripts/probe_device.py` through `tt-device-mcp`. The
   probed architecture and grid must equal the run's source profile in `run.json`; if the broker is
   unavailable, show the command and ask the user to run it in the built tt-metal environment.
3. For `source:cache`, ask where the kernel cache goes: an empty directory outside the tt-metal
   checkout and outside the run directory (a cache holds hundreds of MB to GB), recommended
   `<workspace>/.tt-metal-cache-opa/<model>-<source>-<sha7>`. If the directory exists and is not
   empty, ask for another; never delete it without the user naming it. For `source:graph`, the
   capture goes to `<workspace>/.tt-metal-cache-opa/<model>-<source>-<sha7>.graph_capture.json`,
   beside the cache directory when there is one (a long run writes GB). If the entry
   test runs in a CI workflow, record its `pytest_timeout` in `run.json` as `ci_pytest_timeout_s`.
4. Run the entry test from `run.json` (`test`) through `tt-device-mcp`, once for each run the
   picked items need, one after the other: start the next run only when the previous one has
   finished and the device reports healthy, never two at once:

   | Run | Items | Command |
   |---|---|---|
   | capture | `source:graph`, `source:cache` | `TT_METAL_OPERATION_TIMEOUT_SECONDS=900 TT_METAL_CACHE=<cache dir> TT_METAL_RISCV_DEBUG_INFO=1 OPA_GRAPH_CAPTURE=<capture file> PYTHONPATH=<plugin-root>/pytest_plugins:$PYTHONPATH pytest -p opa_graph_capture --timeout <s> <node id>` |
   | profile | `source:tracy` | `python -m tracy -p -r -v -m "pytest --timeout <s> <node id>"` (needs a profiler build), live |

   With only `source:cache`, drop `OPA_GRAPH_CAPTURE`, `PYTHONPATH` and `-p opa_graph_capture`;
   with only `source:graph`, drop `TT_METAL_CACHE` and `TT_METAL_RISCV_DEBUG_INFO`.

   A cold capture or cache run must set `TT_METAL_OPERATION_TIMEOUT_SECONDS=900`, overriding any
   shorter generic device-run default while preserving its triage callback. Compilation and queued
   execution before a small host readback can legitimately exceed 30 seconds; the 15-minute
   operation watchdog prevents that backlog from being misclassified as a device hang. This is
   separate from the pytest timeout below, which still bounds the whole test.

   Run both from the tt-metal checkout with `TT_METAL_HOME` set to it in the job's environment:
   the broker does not set it, and without it `tracy` looks for its capture tools under the wrong
   root and exits with "Tracy tools were not found" before the test starts.

   Always pass `--timeout <s>`. Without it pytest-timeout applies tt-metal's `pytest.ini` default
   (300 s), and both runs compile every kernel from nothing: the capture run into the empty cache
   with debug info and graph capture on every call, the profile run because the device profiler
   changes every kernel. A whole model can spend more than 300 s in its first forward; RT-DETR on a
   P100 timed out inside the backbone. Use 1400, or `ci_pytest_timeout_s` when it is larger but
   still below the broker's 1500 s job limit, so pytest reports where it stopped instead of the
   broker killing it. A capture run that times out leaves a partial cache and capture file:
   ask the user before deleting them, then rerun into an empty directory.

   Tracy runs the forward **live**: the test calls the model's forward in the profiled window. A test
   that captures a device trace and replays it never re-enters the forward during the replay, so Tracy
   then shows no host dispatch and no host fallback, whatever the model does. When the entry test
   has a live and a trace variant (`-k live`), pick the live one; when it only replays a trace,
   profile the model's live test instead (a PCC test of the same implementation) or an approved
   temporary copy that calls the forward directly (run-setup.md), and record which. The sheet's host
   fallback comes from this run only.

   The empty cache directory makes every kernel compile in the capture run, so the cache holds
   this test's kernels only. Debug info adds DWARF to the ELFs and does not change the kernel
   code. If the test has `op-analysis-start` / `op-analysis-end` signposts the Tracy window is warm;
   otherwise it is cold and labelled so (host times then include compilation). Do not combine
   watcher or DPRINT with profiling. If the test fails only with the graph capture on (for example
   inside a device-trace region), run it again without it for the cache and mark `source:graph`
   failed with the error. Revert any local test edit afterwards and say so. Record each run's pass
   line (for example the PCC) and date.
5. **Tracy** (`source:tracy`). Use the newest `ops_perf_results_*.csv` under
   `$TT_METAL_HOME/generated/profiler/reports/`:
   `python3 <plugin-root>/scripts/tracy_report.py <csv> <run dir> [--start op-analysis-start --end op-analysis-end]`
   writes `measured_ops.csv`, `host_fallback.csv` and `footprint.csv`. Pass `--peak-dram-mb` only
   with a measured value. Record the raw CSV path in `run.json` (`ops_csv`).
6. `python3 <plugin-root>/scripts/merge_measured.py <run dir> <run dir>/measured_ops.csv --tracy-mode live --inferences <n>`
   (`--tracy-mode trace` only to record an old traced run; its host fallback is then shown as not measured),
   where `<n>` is the number of forward passes in the measured window (1 for a warm signposted
   window; a cold window over the whole test counts every forward the test runs). Static rows seen
   in Tracy become `static+measured`; op codes only Tracy shows become `measured` rows after the
   static row that ran just before them, with launches per inference, `call_site` and
   `grid_dependency` `not traced`, and no launch count on any destination.
   When the entry test passed, add `--entry-pass "<pass line>"` (for example `PCC 0.990856`),
   `--date <YYYY-MM-DD>` of the run and `--build <type>` (step 1). The pass is then a recorded run (`target-status.md`): the
   script adds it to `run.json` `entry_test_passes` and decides every source `⚠️` row again, since
   that `⚠️` only said no passing run was found. A row the test launched becomes `✅`; a
   conditional row whose op code ran fewer times than its static rows allow stays `⚠️` (Tracy does
   not say which call site ran); a row it never launched stays `⚠️`. Each gets an evidence text
   saying that no CI workflow runs the entry test on this profile and that it passed on a local
   device under Tracy, with the pass line and date. `✅`, `❌` and `n/a` rows keep their status; a
   `❌` row the test launched is printed as a warning to resolve by hand. When the entry test failed,
   add `--entry-fail "<failure line>"` instead: the failure is recorded in `run.json`
   `entry_test_failure` and shown as ❌ on the sheet; statuses stay as they are.
   Then `python3 <plugin-root>/scripts/diff_reports.py --static <run dir>/op_table.csv --measured <run dir>/measured_ops.csv --profile <source> --out <run dir>/diff.csv`
   writes where the static trace and Tracy disagree per op code (the sheet's Static vs Tracy tab).
7. For each `measured` row, fill every destination's status per `target-status.md` (kernel-op
   lookup) and, for Quasar targets, its `<t>:tests` column, then rerun `validate_static.py`.
8. **Graph capture** (`source:graph`).
   `python3 <plugin-root>/scripts/graph_report.py <capture file> <run dir> --inferences <n> --date <YYYY-MM-DD>`,
   where `<n>` is the number of forwards the test dispatched through ttnn. A replayed device trace
   launches nothing new, so a test that compiles, captures a trace and then replays it counts the
   compile and capture forwards only; read the test to count them. The script refuses a capture
   taken at another commit. When the entry test passed in the capture run, add
   `--entry-pass "<pass line>"`: as in step 6, the pass is recorded and source `⚠️` rows the run
   launched become `✅` (this decides them when Tracy was not picked); `--entry-fail` records a
   failure as in step 6. The script also records the peak DRAM allocated at once on a device
   during the capture (`peak_dram_mib`), which the sheet shows. A WARNING that a count is
   not a multiple of `<n>` means `<n>` is wrong or a launch happens once per test, in setup such
   as weight upload, which the capture includes; say which in `summary.md`. The script parses the
   whole capture, so a multi-GB capture needs a host with several times its size in memory.
   Read `graph_host_transfers.csv`: a transfer between two launches is the next inference's input
   upload, the previous one's readback, or host work inside the forward. Add the last kind to
   `host_ops.csv` (`counts_as_host_op` `yes`) and to `summary.md`.
9. **Kernel cache** (`source:cache`).
   `python3 <plugin-root>/scripts/cache_report.py <cache dir> <run dir> --date <YYYY-MM-DD>`
   (`--elfray <path>` when the engine is not found; `--layers` to change the default
   `llk_api,dataflow_api`, for example `llk_api,llk_core` for an LLK coverage check). Each WARNING
   names a kernel build the engine could not read; builds without DWARF mean the run missed
   `TT_METAL_RISCV_DEBUG_INFO=1`. The cache is per kernel and API call, not per kernel op: it does
   not change `op_table.csv` or `kernel_ops.csv`. `summary.md` gives the counts from the script and
   the distinct data formats and math fidelities in `cache_compute.csv`.
10. **Quasar LLK gap check**, after the cache, whether or not the run has a Quasar destination.
   `python3 <plugin-root>/scripts/llk_gaps.py check <run dir>/cache_compute.csv --tt-metal <checkout> --run-dir <run dir>`
   compares every LLK API in the cache with its definitions in the Quasar LLK sources at the run's
   commit, and every data format with the Quasar DataFormat enum, into `llk_gaps_quasar.csv`
   (`present`, `signature differs`, `missing`). The match is by name, so it over-reports: an API
   can exist under another name (`similar_quasar_apis`) and a template argument can be renamed
   (`quasar_template_args`). Read the Quasar header for every `missing` and `signature differs`
   row with `git show <sha>:<path>` and record the verdict:
   `python3 <plugin-root>/scripts/llk_gaps.py review <run dir>/llk_gaps_quasar.csv <id> <gap|renamed|not a gap> --note "<file:line and what it shows>"`.
   A wrapper that exists only in the Blackhole API layering (`*_impl`) is `not a gap` when its
   caller exists on Quasar. With many candidates, give one subagent each API family (unpack, math,
   pack, sync), under `dispatch-tracing.md`'s contract. `summary.md` lists the confirmed gaps first,
   then the renames, with the kernels that need each. The same check runs on an elfray extension
   export: `llk_gaps.py check <compute.csv> --tt-metal <checkout> --ref <sha> --out <csv>`.
11. `python3 <plugin-root>/scripts/kernel_ops.py <run dir>` (it reads Tracy counts from run.json `measured_ops`)
   shows static, Tracy and graph launches side by side and lists in `sources` which methods saw
   each kernel op; it reads `graph_ops.csv` itself. A kernel op with origin `graph` was launched but
   is missing from the static trace; a measured-only kernel op takes its factory from the graph
   capture. `tracy (op code)` in `sources` means Tracy saw the op code, which several kernel ops
   share, so it does not confirm this factory. Explain every difference in `summary.md` (conditional moves,
   runtime reroutes, ops the static trace missed). Do not change static rows to match a
   measurement.
12. After `--entry-pass`, rewrite every sentence in `summary.md` (and any port-blocker or host-op
   text) that still says no passing source run exists or that the entry test is unverified on the
   source: state that no CI workflow runs it on this profile and that it passed on a local device
   under Tracy, with the pass line, and recount the source statuses from `kernel_ops.py`. Rerun
   `validate_static.py`; it warns about measured, unconditional source rows still `⚠️` while
   `entry_test_passes` is set.

When an item cannot run or its output is unusable (no device, no profiler build, no elfray engine,
test failure), record it with
`python3 <plugin-root>/scripts/measurements.py mark <run dir> <item> failed --cause <cause> --note "<reason>"`
so the next session shows why. A successful script run marks its item done by itself. `--cause` says
what the failure means for the entry test:

- `test`: the entry test itself hung, failed or missed its PCC in this run (a device hang on a
  blocking read, a TT_FATAL, a PCC below threshold), even when another run passed. The sheet then
  shows the end-to-end test as ⚠️ "passed in 1 of N runs", not ✅: a pass that does not repeat is not
  a reliable pass.
- `tool`: the instrument stopped a run the test would otherwise finish (Tracy's instrumented cold
  compilation running past the timeout, an unwritable capture file, an elfray error). Listed on the
  end-to-end row, without downgrading it.
- `not-run`: the run never started (no device, no profiler build, no engine).

Decide `test` versus `tool` from where it stopped: a run that stopped inside the model's forward on a
device wait, with nothing of the tool's own on the stack, is `test`.

Silicon numbers come from the scripts, never from reading a CSV by eye. Do not claim a host
percentage or footprint the scripts did not produce.
