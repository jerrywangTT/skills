# Static phase

Reads code only; never touches a device. Every count must survive
`<plugin-root>/scripts/validate_static.py`, and every dispatch claim carries evidence.

## Steps

1. **Source and blocks.** Load `<plugin-root>/references/targets.md`, confirm the source profile's
   values at the pinned SHA, and add `kind` (`static`), `destinations` (empty) and
   `profile_values` to the `run.json` written at setup, and `entry_test_passes` (`{}`) only when
   it is absent. Never overwrite what the preflight recorded (`entry_test_passes`,
   `entry_test_failure`, `graph_capture`, `kernel_cache`, `measurements`; run-setup.md, "Entry
   test preflight"). Write `stage_blocks`: the model's
   top-level blocks in forward order, from its structure (for ResNet50 `stem`, `layer1` …
   `layer4`, `head`).
2. **Inventory.** Read the model file and the test once, by range, noting line numbers. List every
   ttnn call on the forward path (`<file>:<line>`, stage, how many times it runs per inference),
   including calls that become views, equality no-ops or untaken branches, and every predicate from
   targets.md the code reads. Note host-side work for `host_ops.csv`. Keep zero-launch decisions in
   the inventory and summary/call-trace notes, not as device-op rows in `op_table.csv`.
3. **Trace.** Load `<plugin-root>/references/dispatch-tracing.md`. Small model: trace yourself.
   Larger model: one subagent per block with the subagent contract in that file. Collect rows for
   `op_table.csv` and `call_trace.csv` per `<plugin-root>/references/schemas.md`, then normalize
   duplicate call sites and zero-launch decisions yourself before writing either table. Every
   traced row has `origin` `static`. Stages follow `<block> <module> <role>` (schemas.md), so runs
   of two implementations of one model align by stage.
4. **Source status.** Load `<plugin-root>/references/target-status.md` and fill the source's
   `<t>:status` and `<t>:evidence`. For a Quasar source fill `<t>:tests` with the node ids of the
   tests that decide each status, in the form pytest prints them, including the ids fixtures add
   (ttsim.md, "Node IDs"), and set `<t>:test_result` to `not run`. Never derive parametrized ids
   only from decorators: repository conftest hooks may add an architecture prefix. The measured
   phase runs `kernel_tests.py check --fix-node-ids`, which safely normalizes a single added prefix
   and stops on ambiguous mismatches before any TTsim run (measured-destination.md, step 3).
   Research subagents follow the subagent contract in dispatch-tracing.md.
5. **Re-check evidence.** Reopen the pinned source for every `verified` row and every status. A
   status follows its definition in target-status.md; Quasar targets decide per kernel op (device
   op, factory, layout, dtype) from the Quasar tests and fork code. Dispatch uncertainty
   (runtime-dependent factory, conditional launch) belongs in `confidence` and `launch_condition`;
   do not move a status to `⚠️` for it. `verified` requires the complete dispatch chain for the
   exact variant. Apply the same status to rows sharing the target-relevant signature unless
   row-specific evidence distinguishes them.
6. **Configurability.** Record in `run.json` whether the model can run on a smaller core grid and
   with fewer layers, each as `{"value": "yes" | "no" | "partial" | "n/a", "evidence": "<file:line
   and the decisive fact>"}`:
   - `configurable_grid`: `yes` when every grid, shard spec and program config the model uses is
     derived from the device grid, so the runtime override `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`
     shrinks it with no code change (every row's `grid_dependency` is `automatic` or `none`); `no`
     when grids are hardcoded (give how many places and one example); `partial` when some are.
   - `configurable_layers`: `yes` when the model or its test reads a layer count it can be given,
     such as an environment variable `<MODEL>_NUM_LAYERS` or a test parameter; `partial` when that
     count covers some of the repeated stacks but not all (an encoder but not the decoder); `no` when the
     model has a repeated layer stack with a fixed count; `n/a` when it has no repeated layer stack.
     For `yes` and `partial` add `"setting"`: the environment variable or test parameter that sets
     the count (`HY_NUM_LAYERS`, `test parameter num_layers`). The sheet puts it in a note on the
     Info row.
7. **Gate.** Run `python3 <plugin-root>/scripts/validate_static.py <run dir>`. Fix and rerun until
   it prints `OK`. Resolve every warning (shorten the evidence, drop row-number prefixes). No
   output, summary or upload before that. When the preflight's capture run passed, run
   `python3 <plugin-root>/scripts/graph_report.py <run dir> --decide`: it reads `graph_ops.csv`
   and the recorded pass, not the capture, and decides the source rows the run launched. Then the
   gate again.
8. **Kernel ops and summary.** Run `python3 <plugin-root>/scripts/kernel_ops.py <run dir>`. Write
   `summary.md` for a reader who opens it cold; the spreadsheet's Info tab is this file plus the
   run details from `run.json` and counts computed from the CSVs, so do not repeat those here.
   A `#` title, the per-kernel-op table from `kernel_ops.csv` (kernel op, launches, op-table rows,
   statuses per target), then these `##` sections in this order, as `- **Label**: text` bullets or
   short paragraphs. Every bullet that names something (a finding, a count, a verdict group) starts
   with a bold label, so the sheet puts the label in its own column:
   - `Method`: one paragraph: what was traced, from which entry test, how statuses were decided,
     what was run and what was not.
   - `Targets`: one bullet per target: the question it answers (for a destination, what it
     assumes) and the profile facts that change dispatch (grid, dtypes, batch).
   - `Status legend`: one bullet per status value used, defined for this run's targets as
     `- <status>: <target> - <definition>. <target> - <definition>.` The sheet leaves this section off
     Info and puts each target's definition in a note on its status count.
   - `Findings`: three to six bullets a reader acts on, each with its numbers: launches per
     inference as the validator's range and the tighter executable range when known, status counts
     where they matter, the port blockers that decide the outcome. For a Quasar target, how many
     `✅` rest on a TTsim run, a release document, or test markers alone (target-status.md).
   - `Cleanup found on the way` (only when there is one): inefficiencies in the model code noticed
     while tracing, with the call site and what would remove them.
   - `Open items`: `unverified` rows, runtime-dependent launches, measurements not run yet.

   No section that explains CSV columns: the sheet headers are readable and `schemas.md` defines
   them. Define a term only where a reader cannot infer it. Update `summary.md` whenever a
   destination or a measurement is added.

## Rules

- Read sources as `git show <sha>:<path>` with line ranges; never the working tree. The only
  exception is an approved temporary test in the run directory (run-setup.md, section 1).
- In steps 2-5, read no file of an earlier run under `op-analysis/`: an earlier run is not
  evidence. Only `compare_runs.py` opens one, on request (run-setup.md).
- A status of `✅` needs evidence a reader can open. Without a matching test or run, `⚠️` with
  what is missing.
- Do not upgrade confidence or status because a likely branch, familiar operation name or
  analogous test looks convincing.
- `validate_static.py` is a structural/count gate, not evidence that a factory choice or support
  status is correct. The manual re-check in step 5 remains mandatory after it prints `OK`.

## Adding a destination

A destination reuses the source trace; it never re-traces the source. Ask for it in one question
listing every choice but the source, likely ones first: `p100`, `p150`, `n150`, `n300-1chip`,
`quasar` (the code as written, on Quasar: block-float dtypes and mainline calls stay as they are) and,
for a non-Quasar source, `quasar-bf16` (a BF16 projection: block-float dtypes become BF16 and
mainline calls use their Quasar forks).

A destination is always a projection of the source rows: one to one (the other profile, or `quasar`
with the code as written) or with the BF16 change (`quasar-bf16`). Each row's kernel op, with the
parameters the source model passes there, is judged on the destination. Branches the code takes on
the destination's architecture (`is_quasar()`, `is_wormhole_b0()`, arch checks in the forks) do not
add, drop or rewrite rows and never make a row `n/a`; never offer a separate run instead of a
destination. `n/a` is only for a row whose source launch does not happen on that profile under the
same branch.

1. Append the destination to `run.json` `destinations`.
2. For a Blackhole or Wormhole destination running the same code, evaluate the profile predicates
   from targets.md for every source row and fill `<dest>:shapes` and `<dest>:launches`; add its
   `call_trace.csv` rows with `profile` set to it. Rows the destination does not launch get 0
   launches and `n/a`. Never change the source's columns or rows.
3. Fill `<dest>:status` and `<dest>:evidence` (and for Quasar targets `<dest>:tests` as in step 4
   of the static phase, with `<dest>:test_result` `not run`) per target-status.md; add the
   destination's blockers to `port_blockers.csv`. Measured-only rows get the destination's status from a kernel-op lookup.
4. Decide per device op whether the destination runs it at all, for any arguments, and write
   `device_op_support.csv` per target-status.md, "Device op support". Do this for every device op,
   also those with `✅` rows; it matters most where no row is `✅`.
5. Rerun `validate_static.py` and `kernel_ops.py`, then update `summary.md`: under `Findings`,
   name the device ops with no `✅` row and say for each whether the op is missing on the
   destination (`❌`), untested (`⚠️`) or runs with other arguments (`✅`).
