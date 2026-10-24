# tt-model-op-analysis

Kernel-op analysis of one implementation of a tt-metal model on the architecture it runs on, and
how its kernel ops would port to other architectures.

```bash
/plugin install tt-model-op-analysis@tenstorrent-skills     # Claude Code
codex plugin add tt-model-op-analysis@tenstorrent-skills    # Codex
```

## How the analysis works

The question is: which kernel ops does this model run, and can each of them run on another
architecture? The skill answers it in two steps.

1. **Whole model, on the source architecture.** The model's end-to-end entry test is analysed on
   the architecture it already runs on (for example P150). Every ttnn call is traced statically
   to the kernel op it launches: device op plus program factory, with its layout, dtype and launch
   count. Optionally, runs of the same test on a source device confirm what actually ran, three
   ways: a Tracy profile (op codes), a ttnn graph capture (the device op and program factory of
   every launch) and the kernel cache it compiled, decomposed with the elfray engine into the LLK
   and dataflow API calls of every kernel. Graph capture and cache share one run; Tracy has its own. The result is the list of kernel ops the
   model needs, with the methods that saw each one and a confidence for the whole list.
2. **Kernel op by kernel op, on the destination.** The model is not run end to end on the
   destination. Instead each kernel op from step 1 is checked on its own. For a Quasar
   destination, the Quasar kernel-op tests that exercise that op are run on TTsim, and the result
   (pass, fail, xfail) sets the op's status; a timeout or error decides nothing. Without a TTsim
   run the status comes from the test markers and the Quasar fork code. For a Blackhole or Wormhole destination there is no simulator
   step: the status comes from CI and the profile's code paths.

The per-kernel-op table (`kernel_ops.csv`) is the answer: for every kernel op of the model, whether
it runs on the source and on each destination, and what that rests on.

## Skills

- `model-op-analysis`: runs the analysis and writes the results to a local run directory.
- `upload-op-analysis`: uploads finished runs to Google Sheets. It asks where the results are,
  which run to send, then which Drive folder; a run uploaded before refreshes its spreadsheet.

## Runs

A run is one implementation of a model, its entry test, a source profile and a pinned tt-metal
commit, stored in `op-analysis/<model>/<impl-slug>/<source>/<sha7>/` with a `run.json`. A session starts by listing the model's existing runs; it continues one of them
or starts a new one, and never repeats work a run already has.

- **Source profiles:** `p100`, `p150` (Blackhole), `n150`, `n300-1chip` (one of the two
  Wormhole chips of an N300 card; single-device tests only) and `quasar` (TTsim).
- **Static analysis**, once per run: every ttnn call of the entry test traced to its device op and
  program factory, with support decided per kernel op. No device.
- **Destinations**, optional and added any time without re-tracing: another profile running the
  same code, or `quasar-bf16`, a Quasar projection with block-float dtypes as BF16 and mainline
  calls routed to their Quasar forks.
- **Measurements**, as items picked one or more at a time and resumable across sessions: from one
  run of the entry test on a local source device, its ttnn graph capture (`source:graph`) and its
  kernel cache (`source:cache`, needs the elfray engine); from a second run, its Tracy profile
  (`source:tracy`); and TTsim runs of Quasar kernel-op tests, one node at a time, grouped by row
  status (`<target>:kernel-tests ❌`). An entry test never runs on TTsim. Each item shows its state and a time estimate.

Outputs in the run directory:

- `kernel_ops.csv`: one row per kernel op (device op and program factory) with launches and
  statuses per target; the summary starts with it;
- `device_ops.csv` (with destinations): one row per device op with its rows counted by status per
  destination and whether the destination runs the op at all, decided in `device_op_support.csv`;
- `op_table.csv`: one row per launching device-op variant, marked `static`, `measured` or
  `static+measured`;
- `call_trace.csv`, `port_blockers.csv`, `host_ops.csv`, `summary.md`;
- Tracy: `measured_ops.csv`, `host_fallback.csv`, `footprint.csv`;
- graph capture: `graph_ops.csv` (launches per device op and program factory), `graph_host_transfers.csv`;
- kernel cache: `cache_compute.csv`, `cache_dataflow.csv` (the elfray Cache API Analyzer tables), `cache_kernels.csv`;
- Quasar LLK gap check: `llk_gaps_quasar.csv`, every LLK API and data format of the cache compared
  with the Quasar LLK sources at the run's commit, with a reviewed verdict per candidate;
- TTsim: `kernel_test_results_<target>.csv` with pass, fail, xfail, timeout or error per kernel-op test.

Comparing two implementations of one model (for example the Blackhole model and its Quasar port)
is a separate step on two runs.

Setup asks one question at a time, using native choice controls when the host exposes them. Counts,
totals, schema checks, merges and test runs come from the scripts in `scripts/`, not from the agent.
The source device run needs a device of the source architecture (routed through `tt-device-mcp`),
Tracy a profiler build, and the kernel cache the elfray engine (Linux x86_64); Quasar items need
a local TTsim build, which gives execution and correctness evidence, never performance. Upload needs
the host's Google Drive and Google Sheets connectors.
