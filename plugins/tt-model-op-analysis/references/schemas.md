# CSV schemas

`<t>` is a target from `targets.md`: the run's `source` and each of its `destinations` from
`run.json`.

All CSVs are UTF-8 with a BOM (`utf-8-sig`) so Excel and Numbers show the status emoji;
write them with `opa_schema.write_csv` or open files with `encoding="utf-8-sig"`. Column order is
`opa_schema.py`; `validate_static.py` rejects missing and unexpected columns.

## op_table.csv (static, one row per unique launching device-op variant)

| Column | Content |
|---|---|
| `id` | 1..N in row order |
| `stage` | `<block> <module> <role>` from the model's structure, not from op names: `stem fold`, `layer1 m1 conv1`, `layer3 m2-m6 conv2`, `head fc`. The first word is one of `run.json` `stage_blocks` (the model's top-level blocks); at least two words. Variants inside one role (halo, move, conv of one conv layer) share the stage and differ by `device_op`. Measured-only rows use `measured only <op code>` |
| `origin` | `static` (traced from source only), `measured` (seen only in the source Tracy run; `call_site` and `grid_dependency` are `not traced`) or `static+measured` |
| `ttnn_api` | the Python-facing call |
| `op_code` | Tracy OP CODE, e.g. `MatmulDeviceOperation`; empty only when unknown |
| `device_op` | `ttnn::prim::` name |
| `program_factory` | canonical factory type name only; implementation notes go in `dispatch_evidence` |
| `call_site` | every call site as `<path>:<line>` at the pinned SHA, `; `-separated; add the C++ dispatch line (`<file>.cpp:<line>`) when the launch comes from inside a composite |
| `launch_condition` | exactly `always`, or the runtime condition under which the launch happens (`skipped when the new buffer lands at the same address`) |
| `confidence` | `verified` \| `unverified`: whether the dispatch chain to this factory is fully followed (`dispatch-tracing.md`) |
| `dispatch_evidence` | file and function that select the device op and factory, and every runtime-dependent alternative; at most 300 characters, shared by rows with one dispatch chain |
| `grid_dependency` | the hardcoded grid, shard spec or program config this op depends on, with its value and `<file>:<line>` (`8x4 CoreGrid, <model>.py:<line>`); `automatic` when derived from the device grid, including a model constant the code clamps to the device grid (`min(64, device cores)`), since a smaller grid shrinks it without a code change; `none` when the op uses no core grid (interleaved DRAM) |

Then, for the source and then each destination in `run.json` order, the columns from
`opa_schema.target_columns`:

| Column | Targets | Content |
|---|---|---|
| `<t>:shapes` | the source; p100, p150, n150, n300-1chip as destinations | input and output shapes, dtype, layout and sharding on that profile; `not traced` on a destination for a measured-only row |
| `<t>:launches` | as `<t>:shapes` | launches per inference on that profile; empty on a destination for a measured-only row |
| `<t>:tests` | quasar, quasar-bf16 | pytest node ids of the Quasar kernel-op tests that decide the status, `; `-separated |
| `<t>:test_result` | quasar, quasar-bf16 | `not run` until `kernel_tests.py` runs them; then `pass`, `fail`, `xfail`, `timeout`, `error` or `skip` |
| `<t>:status` | all | `✅`, `⚠️`, `❌`, `n/a`, defined per target in `target-status.md` |
| `<t>:evidence` | all | evidence for this target's status only: the decisive fact and its `<file>:<line>`, at most 300 characters, no row-number prefix; rows with one signature share it (`target-status.md`) |

Do not add rows for calls proven to launch no device operation. Record those decisions in the
inventory and in `call_trace.csv` notes or `summary.md`. If launch versus no-launch cannot be
settled statically, keep the potential device-op variant, set `launch_condition` to the deciding
condition and count the launch; the validator reports the total as a range from the `always`
launches to all launches. Combine multiple call sites only when the full dispatch signature and
every target status match; otherwise use stable substage names so
`stage,ttnn_api,device_op,program_factory` remains unique.

## call_trace.csv (static, execution order)

`id`, `profile`, `stage`, `ops` (device ops in order), `repeats`, `launches_per_repeat`, `notes`.
`profile` is each run target that has launches: the source and every p100, p150, n150 or n300-1chip
destination. For each, the sum of
`repeats * launches_per_repeat` must equal the sum of `<t>:launches` in `op_table.csv`.

## port_blockers.csv (static)

`id`, `blocker`, `affected_ids` (comma-separated `op_table.csv` ids, or `all`), `evidence` (the
check at the pinned SHA), `required_change` (what the port must change). Lists what stops the
source code from running on a destination: unsupported dtypes, kernel generations, grids larger
than a destination grid. Name the destination in `blocker` when a run has more than one. Empty (header only) when nothing blocks.

## host_ops.csv (static)

`id`, `host_work`, `when`, `counts_as_host_op`, `reason`. `counts_as_host_op` is `yes` when the
work runs on the host once per inference inside the measured forward, such as torch calls in model
code, so it must be added by hand to the host op time of `host_fallback.csv`, which Tracy does not
see; `no` for test setup, weight preparation and reference computation. `reason` explains the
choice.

Also list per-inference Python control or metadata construction when it is material. Mark it `yes`
only when it occurs inside the forward and a future measured report would otherwise omit it; give
its frequency in `when`. Do not label TTNN lifetime calls as hidden host compute merely because
they launch no device kernel.

## measured_ops.csv

`id`, `op_code`, `op_type`, `attributes`, `inputs`, `outputs`, `core_count`, `device_kernel_ns`,
`host_ns`.

Written by `tracy_report.py` for the `source:tracy` item; `merge_measured.py` reads it.

## kernel_ops.csv (one row per kernel op)

`id`, `device_op`, `program_factory`, `op_code`, `origin` (`static`, `measured`, `static+measured`,
or `graph` for a kernel op only the graph capture saw), `sources` (the methods that saw it, from
`static`, `tracy`, `graph`; `tracy (op code)` when several kernel ops share the op code Tracy
saw), `op_table_ids` (the op_table rows of this kernel op), `static_launches`
(traced launches of these rows), `static_launches_op_code` and `measured_launches_op_code` (all
launches of this op code, traced and measured; Tracy reports op codes, not factories, so the
measured count is per op code and repeats on every kernel op that shares it), `graph_launches`
(launches per inference of this device op and factory in the graph capture), then per target (source, then destinations) `<t>:statuses` (count of each status over
its rows), and for Quasar targets `<t>:tests` and `<t>:test_results`. Written by `kernel_ops.py`;
`summary.md` starts with this table and the spreadsheet shows it as the Device ops by factory tab.

## device_op_support.csv (destinations, written by the agent)

`device_op`, then per destination `<t>:op_support` (`✅`, `⚠️`, `❌`, `n/a`) and `<t>:op_evidence`
(at most 300 characters). One row per device op of `op_table.csv` (`device_op`, or `op_code` when it
is empty). Whether the destination runs the op at all, for any arguments; defined in
`target-status.md`, "Device op support". Only runs with destinations have it.

## device_ops.csv (one row per device op, destinations only)

`id`, `device_op`, `program_factories`, `ttnn_apis`, `op_table_ids`, `<source>:launches`, then per
destination `<t>:rows_ok`, `<t>:rows_untested`, `<t>:rows_failing`, `<t>:rows_na` (op-table rows
with `✅`, `⚠️`, `❌`, `n/a`), `<t>:op_support` and `<t>:op_evidence` from `device_op_support.csv`
(`⏳` and empty when it has no verdict). Written by `kernel_ops.py`; the spreadsheet shows it as the
Device op support tab, second after Info. There a non-zero row count is coloured by the device op's
verdict: with `✅`, ⚠️ rows yellow (an untested call may run as it is) and ❌ rows orange (the call moves to
a supported variant); with `⚠️`, ⚠️ rows orange (they may need nothing once the op's tests pass) and ❌ rows
red; with `❌`, both red (the op needs work on the destination first).

## graph_ops.csv and graph_host_transfers.csv (graph capture)

Written by `graph_report.py` for the `source:graph` item. `graph_ops.csv`: `id`, `op_code` (the
device-operation class, as Tracy names it), `device_op` (`ttnn::prim::` name), `program_factory`
(the factory type without leading namespaces, as op_table.csv names it), `program_factory_type`
(the full C++ type), `ttnn_apis` (top-level ttnn calls that launched it), `launches` (in the whole
capture), `launches_per_inference`, `cache_misses` (launches that built a new program).

`graph_host_transfers.csv`: `id`, `transfer` (`to_torch`, `from_torch`, `from_device`, `to_device`,
`copy_host_to_device`, `copy_device_to_host`; a transfer inside another counts once, as the outer
one), `after_op_code` and `before_op_code` (the launches around it), `count`. Only transfers between
the first and the last launch are listed.

## cache_compute.csv, cache_dataflow.csv and cache_kernels.csv (kernel cache)

Written by `cache_report.py` for the `source:cache` item, with the columns of the elfray VS Code
extension's Cache API Analyzer tables so the two can be compared cell by cell.

`cache_compute.csv`, one row per distinct compute API call across kernels: `id`, `API` (name and
template arguments), `Kernels`, `TTNN Ops` (elfray's name for the enclosing compute-API call, such
as `pack_tile`; not a ttnn operation), `Op Args` (constant runtime arguments; `=?` when dynamic),
`Input Data Formats` and `Output Data Formats` (per circular buffer), `Tile Dims`, `Math Fidelity`,
`Math Approx`, `FP32 Dest Accum`, `Dst Sync Mode`.

`cache_dataflow.csv`, one row per distinct data-movement call: `id`, `API`, `Kernels`, `Layer`,
`Processor(s)` (`brisc`, `ncrisc`), `Op Args`, `Enclosing Dataflow API`.

`cache_kernels.csv`, one row per kernel build: `id`, `kernel`, `kind` (`compute`,
`data_movement`), `compile_hash`, `api_calls` (calls collected in the selected layers), `errors`.

## llk_gaps_quasar.csv (Quasar LLK gap check)

Written by `llk_gaps.py check` from a compute table; `llk_gaps.py review` sets the verdicts. One row
per LLK API and per data format: `id`, `kind` (`llk_api`, `data_format`), `name`, `quasar_status`
(`present`, `signature differs`, `missing`, or `not an LLK` for a non-LLK call), `review` (`gap`,
`renamed`, `not a gap`; empty until read), `review_note` (the Quasar source that decides it),
`template_args` (template argument names the kernels instantiated), `missing_template_args` (those
the closest Quasar overload does not take), `quasar_template_args` (that overload's),
`missing_runtime_args`, `similar_quasar_apis` (Quasar LLK names sharing the longest prefix, for a
missing API), `quasar_definitions` (`file:line` of each Quasar definition), `configs` (distinct
template configurations), `cache_rows`, `kernels`.

## Bring-up checks (Info tab)

`sheet_spec.py` derives them, never by hand: end-to-end test (✅ from `entry_test_passes`, ❌ from a
later `entry_test_failure`, each with its build and whether `TT_ASSERT` was compiled in; ⚠️ instead
of ✅ while `entry_test_issues` lists an open issue or a source measurement failed with cause `test`;
every failed run of the entry test is listed with its cause), entry test issues (`entry_test_issues`: ✅ none
open, ⚠️ the open ones, ⏳ not checked), host fallback (host op count and time share from `host_fallback.csv` of a
live Tracy run, `run.json` `tracy_mode`: ✅ 0% of ops on the host, ⚠️ up to 5%, ❌ above), test hardware
(`profile_values`), peak DRAM (`graph_capture.peak_dram_mib`, else a measured `footprint.csv`
value), blockers to a smaller grid (op_table rows whose `grid_dependency` is a hardcoded value, with
the places they are set), and `configurable_grid` and `configurable_layers` from `run.json` (✅ yes
or n/a, ⚠️ partial, ❌ no, ⏳ not checked). A note on each of the last two labels names the setting:
`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` for the grid, `configurable_layers.setting` for the layers.

## Analysis methods and confidence

Four methods tell a run what the model launches, from least to most exact: the static trace (code
read at the pinned commit), Tracy (op codes the entry test launched), the graph capture (device op
and program factory of every launch, with the call's real arguments) and the kernel cache (the
compiled kernels and the APIs inside them). `opa_schema.methods_used` lists the ones a run has;
`opa_schema.confidence` gives the run's confidence in its list of kernel ops (device op and program
factory per launch): `high` with a graph capture, the only method that names the factory of each
launch; `medium` with Tracy or the kernel cache, which confirm op codes or compiled kernels but not
factories; `low` with the static trace only. The
spreadsheet's Info tab and `summary.md` both state the methods and the confidence, and a note on each method's
Info label (`sheet_notes.METHOD_NOTES`) says what the method does, what it is for, what it sees and
misses, and the confidence it gives. Info also carries notes on the configurability labels (the
setting that changes each) and on each `<target>: <status>` count (that target's definition from the
summary's `Status legend`, which Info does not repeat as a section).

## kernel_tests_plan_<target>.csv and kernel_test_results_<target>.csv (TTsim)

One pair per Quasar target. The plan, written by `kernel_tests.py plan`: `id`, `node_id`,
`op_table_ids`, `row_statuses` (`;`-separated statuses of the rows citing the node, used to select
`<target>:kernel-tests <status>` items).

The results: `id`, `node_id`, `op_table_ids`, `result` (`pass`, `fail`, `xfail`, `timeout`, `error`,
`skip`), `duration_s`, `log` (path of the pytest log), `environment` (`run.json` or the
`--env-json` file the node ran under). Written by `kernel_tests.py run`, one row per plan node that
has run; batches add to it. `measurements.py` derives item states and estimates from it.

## host_fallback.csv

`scope`, `window`, `device_ops`, `host_ops`, `device_op_time_ms`, `host_op_time_ms`,
`total_time_ms`, `host_pct_of_total`. Measured values only; no threshold.

## footprint.csv

`op_code`, `launches`, `max_core_count`, `peak_dram_mb` (`not measured` unless supplied).

## diff.csv (Tracy, `diff_reports.py`)

`category`, `op_code`, `static_launches`, `measured_launches`, `static_ids`, `detail`.

## changes_<table>.csv

`change` (`added` | `removed` | `changed`), `key`, `column`, `old`, `new`.
